import os, csv, random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import OrderedDict
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class OfficeHomeDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        assert split in ("train", "val")
        self.img_dir = os.path.join(root_dir, split)
        self.csv_path = os.path.join(root_dir, f"{split}.csv")
        self.transform = transform
        self.rows = []

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rid = r.get("id", r.get("image_id"))
                self.rows.append({
                    "id": rid, 
                    "filename": r["filename"],
                    "label": int(r["label"]),
                })
        if not self.rows:
            raise RuntimeError(f"No items in {self.csv_path}")
        self.num_classes = max(r["label"] for r in self.rows) + 1

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(os.path.join(self.img_dir, r["filename"])).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, r["label"]

def get_officehome_loaders(data_root, batch_size=64, num_workers=4):
    
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4,0.4,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_set = OfficeHomeDataset(data_root, "train", train_tf)
    val_set   = OfficeHomeDataset(data_root, "val",   val_tf)

    if train_set.num_classes != 65:
        print(f"[warn] CSV shows {train_set.num_classes} classes (but expected 65). Continuing.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_set.num_classes

@torch.no_grad()
def evaluate_cls(model, loader, device, log_intervals=20):
    model.eval()
    correct = total = 0
    val_bar = tqdm(loader, desc="val", leave=False)

    for step, (x, y) in enumerate(val_bar, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred, _ = model(x)
        pred = pred.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()

        if (step % log_intervals == 0) or (step == len(loader)):
            acc = correct / total
            val_bar.set_postfix(acc=f"{acc:.2%}")
    
    acc = correct / max(1, total)
    return acc

def save_ckpt(state, out_dir, is_best=False, filename="last.pth"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, os.path.join(out_dir, filename))
    if is_best: torch.save(state, os.path.join(out_dir, "best.pth"))

def load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # new PyTorch
    except TypeError:
        return torch.load(path, map_location="cpu")  # older PyTorch

def unwrap_state_dict(ckpt):
    sd = ckpt
    if isinstance(sd, (dict, OrderedDict)) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, (dict, OrderedDict)) and "student" in sd and isinstance(sd["student"], (dict, OrderedDict)):
        sd = sd["student"].get("backbone", sd["student"])
    return sd

def peek_state_dict(sd, limit=30):
    if not isinstance(sd, (dict, OrderedDict)):
        print(f"[INFO] Not a dict-like state_dict, got: {type(sd).__name__}")
        return
    keys = list(sd.keys())
    print(f"[INFO] Entries: {len(keys)} (showing first {min(limit, len(keys))})")
    for i, k in enumerate(keys[:limit]):
        v = sd[k]
        if hasattr(v, "shape"):
            print(f"{i:03d} {k:40s} shape={tuple(v.shape)} dtype={getattr(v, 'dtype', None)}")
        else:
            print(f"{i:03d} {k:40s} type={type(v).__name__}")

def load_dino_backbone_weights(backbone: nn.Module, ckpt_path: str):
    """Load backbone weights from a DINO checkpoint into a torchvision ResNet50 (fc=Identity)."""

    # load checkpoint and unwrap the state dict
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = unwrap_state_dict(ckpt)
    print("[INFO] The state dict of checkpoint : ")
    peek_state_dict(sd, limit=100000)  # adjust limit to your tastet
    print(sd)

    # strip prefixes and drop non-backbone keys
    clean_sd = {}
    for k, v in sd.items():
        if k.startswith("module."): k = k[len("module."):]
        if k.startswith("backbone."): k = k[len("backbone."):]
        if k.startswith("encoder."): k = k[len("encoder."):]
        # skip DINO head 
        if k.startswith("fc.") or k.startswith("head") or "proj" in k:
            continue
        clean_sd[k] = v
    
    model_sd = backbone.state_dict()
    print("[INFO] The state dict of backbone : ")
    peek_state_dict(model_sd, limit=10000)
    print(model_sd)
    loadable = {k: v for k, v in clean_sd.items() 
                if k in model_sd and model_sd[k].shape == v.shape}

    msg = backbone.load_state_dict(loadable, strict=True)
    print(msg) # showing missing/unexpected tensor


if __name__ == '__main__':
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    load_dino_backbone_weights(backbone=backbone, ckpt_path="./ckpts_r50_mini_300ep_3e-2/checkpoint.pth")

    new_model_sd = backbone.state_dict()
    print("[INFO] The new model state dict : ")
    peek_state_dict(new_model_sd, limit=10000)
    print(new_model_sd)


    

        

