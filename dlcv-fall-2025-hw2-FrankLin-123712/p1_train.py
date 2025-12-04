import os, datetime, logging, argparse, sys, random, csv
from PIL import Image
import numpy as np

import torch
import torch.nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image


from tqdm import tqdm
from p1_model import DDPM, ContextUnet

############### Utils ############### 
def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

############### Dataset ###############  
class DigitsDataset(Dataset):
    def __init__(self, data_path, mnistm_tf=None, svhn_tf=None):

        logging.info("creating DigitDataset...")
        self.data_path = data_path # expected to be ./hw2_data/digits/

        self.imgs_name = []
        self.imgs_label = []

        self.mnistm_tf = None
        self.svhn_tf = None

        if mnistm_tf:
            self.mnistm_tf = mnistm_tf
        else:
            self.mnistm_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ]) 

        if svhn_tf:
            self.svhn_tf = svhn_tf
        else:
            self.svhn_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])

        mnistm_csv_path = os.path.join(self.data_path, "mnistm/train.csv")
        svhn_cvs_path   = os.path.join(self.data_path, "svhn/train.csv")

        # pick only even digit in mnistm dataset
        with open(mnistm_csv_path, "r", newline="") as mnistm_file:
            reader = csv.reader(mnistm_file, delimiter=",")
            next(reader)
            for row in reader:
                img_name, label = row
                if int(label)%2==0:
                    self.imgs_name.append(os.path.join("mnistm/data", img_name))
                    self.imgs_label.append(torch.tensor(int(label)))

        # pick only odd digits in svhn dataset
        with open(svhn_cvs_path, "r", newline="") as svhn_file:
            reader = csv.reader(svhn_file, delimiter=",")
            next(reader)
            for row in reader:
                img_name, label = row
                if int(label)%2==1:
                    self.imgs_name.append(os.path.join("svhn/data", img_name))
                    self.imgs_label.append(torch.tensor(int(label)))
        
    def __len__(self):
        return len(self.imgs_name)

    def __getitem__(self, idx):


        img_path = os.path.join(self.data_path, self.imgs_name[idx])
        _img = Image.open(img_path)
        img = None
        if self.imgs_name[idx].split("/")[0] == "mnistm":
            img = self.mnistm_tf(_img)
        else:
            img = self.svhn_tf(_img)
        label = self.imgs_label[idx]

        return img, label
        
        

############### DataLoader ############### 
def get_digits_loader(data_path, batch_size=32, num_workers=4):

    logging.info("creating digit loader...")
    train_set = DigitsDataset(data_path=data_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    return train_loader

############### Model #################
def get_model(in_channels=3, n_feat=256, n_classes=10, 
              betas=[1e-4, 2e-2], n_T=1000, device="cuda", drop_prob=0.1):
    
    logging.info("creating DDPM model...")

    nn_model = ContextUnet(
        in_channels=in_channels, 
        n_feat=n_feat, 
        n_classes=n_classes
    )
    
    ddpm = DDPM(
        nn_model=nn_model, 
        betas=betas, 
        n_T=n_T, 
        device=device, 
        drop_prob=drop_prob
    )
    
    return ddpm

################# Train ####################
def train_one_epoch(ep, n_ep, model, device, train_loader, opt, sche):
    total_loss = 0.0
    model.train()
    train_bar = tqdm(train_loader, desc=f"[INFO] Train Epoch : {ep}/{n_ep}", leave=True)

    for step, (x, c) in enumerate(train_bar, 1):
        
        x, c= x.to(device, non_blocking=True), c.to(device, non_blocking=True) 
        opt.zero_grad(set_to_none=True)
        loss = model(x, c)
        loss.backward()
        opt.step()
        _loss = loss.item()
        total_loss += _loss
        if step%20==0 or (step==len(train_loader)): 
            train_bar.set_postfix(loss=f"{_loss:.4f}")
        sche.step()
        
    train_loss = total_loss/len(train_loader)
    return train_loss
    
    

################# Validation #################
def validation(model, guide_w, device, save_dir, ep=0):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Disable gradient calculations for inference
    with torch.no_grad():
        # We need 10 different noise instances for each of the 10 digits.
        n_sample = 100
        n_classes = 10

        # Assuming single-channel 28x28 images (like MNIST).
        # This can be adjusted if your model uses a different size.
        img_size = (3, 28, 28)
        C, H, W = img_size

        # Generate 100 images. Based on the provided sample() function,
        # the output will be ordered as: [d0, d1..d9, d0, d1..d9, ...]
        logging.info("Generating validation images...")
        x_gen, _ = model.sample(n_sample, img_size, device, guide_w)
        logging.info("Image generation complete.")

        # Reorder the images to group them by class for the grid.
        # The desired order is: [10 of d0, 10 of d1, ..., 10 of d9]
        
        # 1. Reshape to (num_noise_instances, num_classes, C, H, W)
        # Since n_sample=100 and n_classes=10, we have 10 noise instances.
        x_gen_reshaped = x_gen.view(n_sample // n_classes, n_classes, C, H, W)
        
        # 2. Permute dimensions to (num_classes, num_noise_instances, C, H, W)
        # This swaps the order, grouping images by their class label.
        x_gen_permuted = x_gen_reshaped.permute(1, 0, 2, 3, 4)
        
        # 3. Reshape back to a flat tensor (100, C, H, W) now in the correct order.
        x_gen_reordered = x_gen_permuted.reshape(n_sample, C, H, W)

        # # Create the grid with 10 columns (nrow=10).
        # # We use `1 - x_gen_reordered` to invert the colors for a
        # # white background with black digits, which is standard for MNIST display.
        # grid = make_grid(1 - x_gen_reordered, nrow=10)
        
        # clamp the output to the valid range of [-1, 1].
        x_gen_clamped = torch.clamp(x_gen_reordered, -1, 1)
        # normalize the clamped tensor from [-1, 1] to [0, 1] for saving.
        x_gen_normalized = (x_gen_clamped + 1) / 2
        # create the grid
        grid = make_grid(x_gen_normalized, nrow=10)
        
        # Save the final image grid.
        save_path = os.path.join(save_dir, f"validation_epoch_{ep}.png")
        save_image(grid, save_path)
        
        logging.info(f"Validation grid successfully saved to {save_path}")
    

################# Save ckpts #################
def save_ckpt(model_info, ckpt_path, filename):
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(model_info, os.path.join(ckpt_path, filename))
    logging.info(f"save model to {os.path.join(ckpt_path, filename)}")


################# Parse parameter ####################
def parse_args():
    ap = argparse.ArgumentParser("P1 training conditional DDPM on digits (MINST/SVHN).")
    ap.add_argument("--debug", action="store_true", help="enable debug mode")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data_path", type=str, default="./hw2_data/digits")
    ap.add_argument("--ckpt_path", type=str, default="./ckpts/p1")

    ap.add_argument("--in_channels", type=int, default=3)
    ap.add_argument("--n_feat", type=int, default=256)
    ap.add_argument("--n_classes", type=int, default=10)
    ap.add_argument("--betas", type=float, nargs=2, default=[1e-4, 2e-2], help="two float specifying the beta_start and beta_end for noise scheduling.")
    ap.add_argument("--n_T", type=int, default=1000)
    ap.add_argument("--drop_prob", type=float, default=0.1, help="the probabilty of droping out the context to generate the image without guidance.")
    ap.add_argument("--guide_w", type=float, default=2.0)

    ap.add_argument("--n_ep", type=int, default=100)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--worker", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)

    return ap.parse_args()

def main():
    ########### parse arguments ##############
    args = parse_args()
    ########### set up gpu ###########
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ############ set up logging #################
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("./log/p1", exist_ok=True)
    LOG_FILE = f"./log/p1_train_{time}.log"
    LOG_LEVEL = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"device setted to {device}")
    logging.info(f"log saved to {LOG_FILE}")
    logging.info(f"log level : {LOG_LEVEL}")
    logging.info(f"parameters : {vars(args)}")
    ############ Model #############
    ddpm = get_model(
        in_channels=args.in_channels,
        n_feat=args.n_feat,
        n_classes=args.n_classes,
        betas=args.betas,
        n_T=args.n_T,
        device=device,
        drop_prob=args.drop_prob
    ).to(device)
    logging.info("DDPM model constructed.")

    ############ DataLoader #############
    dataloader = get_digits_loader(
        data_path=args.data_path, 
        batch_size=args.bs, 
        num_workers=args.worker
    )
    logging.info("dataloader constructed.")

    ############ Optimizer ##############
    opt = optim.Adam(params=ddpm.parameters(), lr=args.lr, weight_decay=args.wd)
    logging.info("optimizer constructed.")

    ############ Scheduler ##############
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=opt, 
        T_max=args.n_ep*len(dataloader),
        eta_min=0
    )
    logging.info("scheduler constructed.")

    ############ Train ##############
    logging.info("Start to train ddpm")
    minimum_loss = float('inf')
    for ep in range(1, args.n_ep+1):
        
        train_loss = train_one_epoch(ep=ep, n_ep=args.n_ep, model=ddpm, device=device, train_loader=dataloader, opt=opt, sche=scheduler)
        if ep % 5 == 0: # Example: validate every 5 epochs
            validation(
                model=ddpm, guide_w=args.guide_w, device=device, 
                save_dir=os.path.join(args.ckpt_path, time), ep=ep
            )
        logging.info(f"Epoch : {ep}/{args.n_ep} with Loss : {train_loss:.4f}")
        
        if minimum_loss > train_loss:
            minimum_loss = train_loss
            model_info = {"epoch": ep, "model": ddpm.state_dict(), "args" : vars(args), "loss" : train_loss}
            save_ckpt(model_info, ckpt_path=os.path.join(args.ckpt_path, time), filename=f"best.pth")

        if ep == args.n_ep:
            model_info = {"epoch": ep, "model": ddpm.state_dict(), "args" : vars(args), "loss" : train_loss}
            save_ckpt(model_info, ckpt_path=os.path.join(args.ckpt_path, time), filename=f"last.pth")
    
    
    

if __name__ == '__main__':
    main()

    
    ########## check the image size #############
    # file_path = os.getcwd()
    # file_path = os.path.join(file_path, "hw2_data/digits/mnistm/data/00000.png")

    # with Image.open(file_path) as img:
    #     to_tensor = transforms.ToTensor()
    #     img_tensor = to_tensor(img)
    #     print(f"The image size is {img.size}")
    #     print(f"The image size casted to np is {img_tensor.shape}")

    ############ check out the dataset correctness ###########
    # data_path = "./hw2_data/digits"
    # dataset = DigitsDataset(data_path)
    # mx, mc, mpath = dataset.__getitem__(0)
    # sx, sc, spath = dataset.__getitem__(-1)
    # print(f"the No.{0} image is at {mpath} with label {mc} and shape {mx.shape}")
    # print(f"the value of tensor is : {mx} \r")

    # print(f"the No.{-1} image is at {spath} with label {sc} and shape {sx.shape}")
    # print(f"the value of tensor is : {sx} \r")
    # for i in range(len(dataset)):
    #     x, c, path = dataset.__getitem__(i)
    #     print(f"the No.{i} image is at {path} with label {c} and shape {x.shape}")


