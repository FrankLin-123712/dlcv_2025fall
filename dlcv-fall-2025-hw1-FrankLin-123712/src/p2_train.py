import argparse, sys, os 
from PIL import Image
from pathlib import Path
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
from collections import OrderedDict

from p2_model import U_Net, SegmentationModel

################## args ###################
def parse_args():
    ap = argparse.ArgumentParser("P2 Training segmentation model (UNet or DeepLabv3)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data_root", type=str, default='./data_2025/p2_data')
    ap.add_argument("--ckpt_dir", type=str, default='./ckpts_p2')
    ap.add_argument("--num_epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=7)
    ap.add_argument("--arch", choices=["unet", "deeplabv3"], required=True)
    ap.add_argument("--skip_idx", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=4e-2)
    ap.add_argument("--wd", type=float, default=0)
    ap.add_argument("--lr_factor", type=float, default=0.7)
    ap.add_argument("--patience", type=int, default=5)

    return ap.parse_args()

################## Dataset, DataLoader ################## 
class SatelliteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image = [f for f in os.listdir(root_dir) if f.endswith('_sat.jpg')]
    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        img_name = self.image[idx]
        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(self.root_dir, img_name.replace('_sat.jpg', '_mask.png'))

        img = Image.open(img_path).convert('RGB')
        img_mask = Image.open(mask_path)
        
        if self.transform:
            img = self.transform(img)

        img_mask = torch.from_numpy(np.array(img_mask))
        img_mask = (img_mask >= 128).int()
        img_mask = 4*img_mask[:, :, 0] + 2*img_mask[:, :, 1] + img_mask[:, :, 2]

        mask = torch.zeros_like(img_mask)
        cls_mapping = {3:0, 6:1, 5:2, 2:3, 1:4, 7:5, 0:6}
        for _color, _cls in cls_mapping.items():
            mask[img_mask == _color] = _cls

        mask = mask.long()

        return img, mask

def get_satellite_loader(data_root: str, batch_size=32, num_workers=4):

    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'validation')

    train_dataset = SatelliteDataset(root_dir=train_dir, transform=transform)
    val_dataset = SatelliteDataset(root_dir=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader

################## Model ##################
def get_model(num_classes=7, arch: str = "", skip_idx: int = -1):
    if arch == 'unet':
        model = U_Net(n_class=num_classes, skip_idx=skip_idx)
    elif arch == 'deeplabv3':
        model = SegmentationModel(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model arch: {arch!r}"
            f"Valid options: ['unet', 'deeplabv3']"
        )
    
    return model 
    
################## Train ################
def train_one_epoch(model: nn.Module, ep, args, train_loader, criterion, optim, device):
    model.train()
    _loss, _iou, n = 0.0, 0.0, 0

    train_bar = tqdm(train_loader, desc=f"[INFO] Train Epoch : {ep}/{args.num_epochs}", leave=True)

    for step, (imgs, masks) in enumerate(train_bar, 1):
        # move tensor into device
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        
        # calculate gradient and move one step
        optim.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optim.step()

        # calculate loss
        _loss += loss.item()

        # calculate train_miou
        preds = torch.argmax(outputs, dim=1)
        for i in range(6):
            intersection = ((preds == i) & (masks == i)).sum().float()
            union = ((preds == i) | (masks == i)).sum().float()
            iou = (intersection + 1e-6) / (union + 1e-6)
            _iou += iou.item()

        # logging
        if (step%20 == 0) or (step == len(train_loader)):
            train_bar.set_postfix(loss=f"{_loss:.4f}")

    train_loss = _loss/len(train_loader)        
    train_miou = _iou/(len(train_loader)*6)
    return train_loss, train_miou

################## Validation ################
@torch.no_grad()
def validation(model: nn.Module, ep, args, val_loader, device):
    model.eval()
    _iou = 0.0

    val_bar = tqdm(val_loader, desc=f"[INFO] Val Epoch : {ep}/{args.num_epochs}", leave=True)

    for step, (imgs, masks) in enumerate(val_bar, 1):
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        for i in range(6):
            intersection = ((preds == i) & (masks == i)).sum().float()
            union = ((preds == i) | (masks == i)).sum().float()
            iou = (intersection + 1e-6) / (union + 1e-6)
            _iou += iou.item()
        
    val_miou = _iou / (len(val_loader)*6)
    return val_miou

################## checkpoint ################
def save_ckpt(model_info: dict, ckpt_dir: str, filename: str):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model_info, os.path.join(ckpt_dir, filename))
    print(f"[INFO] save model to {ckpt_dir}")
    

################## main ###################
def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # [1] dataloader, dataset
    train_loader, val_loader = get_satellite_loader(data_root=args.data_root, 
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers)

    # [2] model
    model = None
    if args.arch == "deeplabv3":
        model = get_model(num_classes=args.num_classes, arch=args.arch)
        print("[INFO] Construct the deeplabv3_resnet101.")
    elif args.arch == "unet":
        model = get_model(num_classes=args.num_classes, arch=args.arch, skip_idx=args.skip_idx)
        print("[INFO] Construct UNet that remove one skip connection.")
        print(f"[INFO] Remove the {args.skip_idx}th skip connection")
    
    model.to(device)
    print(f"[INFO] move model to device : {device}")

    # [3] loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_sche = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=args.lr_factor, 
                                                   patience=args.patience, verbose=True)

    # [4] training loop
    best_miou = 0.0
    history = {'train_loss': [], 'train_miou': [], 'val_miou': []}
    num_epochs = args.num_epochs

    for ep in range(1, num_epochs+1):

        train_loss, train_miou = train_one_epoch(model, ep, args, train_loader, criterion, opt, device)
        val_miou = validation(model, ep, args, val_loader, device)

        lr_sche.step(val_miou)

        history["train_loss"].append(train_loss)
        history["train_miou"].append(train_miou)
        history["val_miou"].append(val_miou)

        print(f'[INFO] Epoch {ep+1}/{num_epochs}, Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}, Val mIoU: {val_miou:.4f}')

        if val_miou > best_miou:
            best_miou = val_miou
            model_info = {"epoch" : ep, "model": model.state_dict(), "args": vars(args), "mIoU": val_miou}
            save_ckpt(model_info=model_info, ckpt_dir=f"{args.ckpt_dir}_{args.arch}", filename='best.pth')
        
        if ep == num_epochs:
            model_info = {"epoch" : ep, "model": model.state_dict(), "args": vars(args), "mIoU": val_miou}
            save_ckpt(model_info=model_info, ckpt_dir=f"{args.ckpt_dir}_{args.arch}", filename='last.pth')
            
    print(f"[DONE] train {args.arch}. Best mIoU : {best_miou}")

if __name__ == '__main__':
    main()