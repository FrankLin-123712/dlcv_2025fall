import os, sys, logging, argparse
import random, math, datetime
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


# --------------------------
# Utilities
# --------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f"set torch.seed to {seed}.")

# --------------------------
# Dataset 
# --------------------------
class Fill50KDataset(Dataset):
    """
    Expects structure:
      {root}/
        source/   (control maps; filenames referenced in prompt.json as "source/xxx.png")
        target/   (RGB target images; referenced as "target/xxx.png")
        prompt.json   # JSON Lines; each line is {"source": "...", "target": "...", "prompt": "..."}

    Returns:
      image:  tensor float in [-1,1], Bx3xHxW   (TARGET image -> VAE encoder)
      hint:   tensor float in [0,1],  BxCxHxW   (CONTROL image -> ControlNet hint encoder)
      text:   str prompt
      name:   sample id (derived from target filename stem)
    """
    def __init__(self, root, img_size=512, hint_mode="rgb", prompts_file="prompt.json"):
        self.root = Path(root)
        self.img_size = img_size
        self.hint_mode = hint_mode
        self.prompts_path = self.root / prompts_file

        if not self.prompts_path.is_file():
            logging.error(f"Missing prompt file: {self.prompts_path}")
            raise FileNotFoundError(f"Missing prompt file: {self.prompts_path}")

        # read JSON Lines
        self.records = []
        with open(self.prompts_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # ensure required keys
                    if not all(k in obj for k in ("source", "target", "prompt")):
                        logging.error("record missing keys")
                        raise ValueError("record missing keys")
                    self.records.append(obj)
                except Exception as e:
                    logging.error(f"{self.prompts_path}: line {ln} is not valid JSON: {e}")
                    raise ValueError(f"{self.prompts_path}: line {ln} is not valid JSON: {e}")

        if len(self.records) == 0:
            logging.error(f"No samples found in {self.prompts_path}")
            raise RuntimeError(f"No samples found in {self.prompts_path}")

        self.to_img = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.records)

    def _open_hint(self, path: Path):
        img = Image.open(path)
        if self.hint_mode == "gray":
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx):
        rec = self.records[idx]
        # paths in file are relative (e.g., "source/0.png")
        src_p = (self.root / rec["source"]).resolve()
        tgt_p = (self.root / rec["target"]).resolve()

        if not src_p.is_file():
            raise FileNotFoundError(f"hint not found: {src_p}")
        if not tgt_p.is_file():
            raise FileNotFoundError(f"target not found: {tgt_p}")

        # load images
        target_img = Image.open(tgt_p).convert("RGB")
        hint_img = self._open_hint(src_p)

        x = self.to_img(target_img) * 2.0 - 1.0   # [-1,1] for VAE
        h = self.to_img(hint_img)                 # [0,1] for ControlNet hint

        name = Path(rec["target"]).stem
        return {"image": x, "hint": h, "text": rec["prompt"], "name": name}

# --------------------------
# model helper
# --------------------------
def load_ldm_from_cfg(cfg_path, sd_ckpt=None, device="cuda"):
    cfg = OmegaConf.load(cfg_path)
    model = instantiate_from_config(cfg.model)
    if sd_ckpt:
        sd = torch.load(sd_ckpt, map_location="cpu")
        # Many SD ckpts store under 'state_dict'; strict=False to allow ControlNet new params
        model.load_state_dict(sd.get("state_dict", sd), strict=False)
        logging.info(f"Loaded base weights from {sd_ckpt} (strict=False).")
    model.to(device)
    return model

@torch.no_grad()
def encode_to_latents(model, img_bchw):
    # img in [-1,1], Bx3xHxW
    z = model.encode_first_stage(img_bchw)
    z = model.get_first_stage_encoding(z)
    return z  # Bx4xH/8xW/8

def save_ckpt(model, ckpt_path, filename, epoch=None, loss=None, args=None):
    """
    Save ONLY ControlNet weights for Problem 3.
    The TA will separately provide Stable-Diffusion (UNet/VAE/CLIP) weights.
    """
    os.makedirs(ckpt_path, exist_ok=True)

    # 1) pure ControlNet state_dict
    control_sd = model.control_model.state_dict()

    # 2) wrap with a minimal header (nice for bookkeeping)
    payload = {
        "epoch": epoch,
        "loss": loss,
        "args": dict(args) if args is not None else None,
        "controlnet_state_dict": control_sd,   # <- what we will load at inference
        "_format": "dlcv2025_hw2_p3_controlnet_only_v1"
    }
    outpath = os.path.join(ckpt_path, filename)
    torch.save(payload, outpath)
    logging.info(f"saved ControlNet weights to {outpath}")
# --------------------------
# train
# --------------------------
def train_1ep(args, ep, model, device, dl, opt):

    total_loss = 0.0
    model.train()
    tbar = tqdm(dl, desc=f"[INFO] Train Epoch : {ep}/{args.ep}", leave=True)

    for it, batch in enumerate(tbar, 1):
        img = batch["image"].to(device, non_blocking=True)  # [-1,1], Bx3xHxW
        hint = batch["hint"].to(device, non_blocking=True)  # [0,1],  BxCxHxW
        text = batch["text"]

        with torch.no_grad():
            x0 = encode_to_latents(model, img)                      # Bx4xH/8xW/8
            B = x0.size(0)
            noise = torch.randn_like(x0)
            t = torch.randint(0, model.num_timesteps, (B,), device=x0.device).long()
            x_noisy = model.q_sample(x_start=x0, t=t, noise=noise)
            c = model.get_learned_conditioning(text)                # (B,T,D)

        cond = {'c_crossattn': [c], 'control': hint}

        opt.zero_grad(set_to_none=True)

        eps_pred = model.apply_model(x_noisy, t, cond)          # predict ε
        loss = F.mse_loss(eps_pred, noise, reduction='mean')

        loss.backward()

        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=args.grad_clip
            )

        opt.step()

        _loss = loss.item()
        total_loss += _loss
        
        avg_loss = total_loss/it
        if (it%20==0) or (it==len(dl)):
            tbar.set_postfix(loss=f"{avg_loss:.4f}")
    
    train_loss = total_loss/len(dl)
    return train_loss

# --------------------------
# parse args
# --------------------------
def parse_args():
    ap = argparse.ArgumentParser("P3 Control Net training")
    ap.add_argument("--debug", action="store_true", help="enable debug mode")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data_path", type=str, default="./hw2_data/fill50k/training")
    ap.add_argument("--ckpt_path", type=str, default="./ckpts/p3/")
    ap.add_argument("--config", type=str, default="configs/v1-inference.yaml",
                    help="YAML with control_stage_config added")
    ap.add_argument("--sd-ckpt", dest="sd_ckpt", type=str, required=True,
                    help="pretrained SD checkpoint (UNet/VAE/Text); will be loaded with strict=False")
    
    ap.add_argument("--img_size", type=int, default=512, help="training crop/resize size (pixels)")
    ap.add_argument("--hint_mode", choices=["rgb", "gray"], default="rgb",
                    help="how to read hint images")
    ap.add_argument("--prompts", type=str, default="prompt.json",
                    help="optional prompts.txt (one line per image); else empty prompts")
    
    ap.add_argument("--ep", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--worker", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=None)
    
    # ap.add_argument("--seed", type=int, default=123)

    return ap.parse_args()

# --------------------------
# main
# --------------------------
def main():

    ########### parse arguments ##############
    args = parse_args()
    ########### set up gpu ###########
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ############ set up logging #################
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("./log/p3", exist_ok=True)
    LOG_FILE = f"./log/p3/train_{time}.log"
    LOG_LEVEL = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"set device as {device}")
    logging.info(f"save log to {LOG_FILE}")
    logging.info(f"logging level - {LOG_LEVEL}")
    logging.info(f"parameters : {vars(args)}")
    
    ################ set seed ################
    # logging.info(f"set seed to {args.seed}")
    # set_seed(args.seed)
    
    ################ dataset, dataloader ##################
    logging.info("preparing dataset and datalaoder...")

    ################ model ##################
    logging.info("constructing model...")    
    # model
    model = load_ldm_from_cfg(args.config, args.sd_ckpt, device=device)
    assert getattr(model, "control_model", None) is not None, \
        "control_model not found. Did you add control_stage_config in your YAML?"
    # freeze UNet; train only ControlNet
    for p in model.model.diffusion_model.parameters():  # UNet inside DiffusionWrapper
        p.requires_grad = False
    for p in model.control_model.parameters():
        p.requires_grad = True
    model.train()
    
    ############### loss, optimizer, scheduler #################
    logging.info("instantiate loss, optimizer, scheduler...")
    # data
    ds = Fill50KDataset(
        args.data_path,
        img_size=args.img_size,
        hint_mode=args.hint_mode,
         prompts_file=args.prompts
    )
    dl = DataLoader(
        ds, batch_size=args.bs,
        shuffle=True,
        num_workers=args.worker,
        pin_memory=True
    )
    if args.debug:
        with torch.no_grad():  # turn off grads for the check
            batch0 = next(iter(dl))
            img  = batch0["image"].to(device)          # [-1,1], (B,3,H,W)
            hint = batch0["hint"].to(device)           # [0,1],  (B,C,H,W)
            c    = model.get_learned_conditioning(["test"] * img.size(0))  # (B,T,D)
            z    = encode_to_latents(model, img)       # (B,4,H/8,W/8)
            B    = z.size(0)
            t    = torch.randint(0, model.num_timesteps, (B,), device=z.device).long()
            noise= torch.randn_like(z)
            x_noisy = model.q_sample(x_start=z, t=t, noise=noise)
            eps  = model.apply_model(x_noisy, t, {'c_crossattn':[c], 'control': hint})
            assert eps.shape == x_noisy.shape, f"Mismatch: {eps.shape} vs {x_noisy.shape}"
        logging.debug("[debug] smoke test passed: shapes are consistent.")

    # opt
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd
    )
    
    ############### train ##################
    logging.info("start to train...")
    
    ckpt_dir = Path(args.ckpt_path) / time; ckpt_dir.mkdir(parents=True, exist_ok=True)
    minimum_loss = float('inf')
    
    for ep in range(1, args.ep+1):
        train_loss = train_1ep(args, ep, model, device, dl, opt)

        logging.info(f"Epoch : {ep}/{args.ep} with loss : {train_loss:.4f}")
            
        if minimum_loss > train_loss:
            minimum_loss = train_loss
            save_ckpt(model=model, ckpt_path=ckpt_dir, filename="best.pth", epoch=ep, loss=train_loss, args=vars(args))

        if ep == args.ep:
            save_ckpt(model=model, ckpt_path=ckpt_dir, filename="best.pth", epoch=ep, loss=train_loss, args=vars(args))
    

if __name__ == '__main__':
    main()

    
    ########## check the image size #############
    # file_path = os.getcwd()
    # file_path = os.path.join(file_path, "./hw2_data/fill50k/training/source/0.png")

    # with Image.open(file_path) as img:
    #     to_tensor = transforms.ToTensor()
    #     img_tensor = to_tensor(img)
    #     print(f"The image size is {img.size}")
    #     print(f"The image size casted to np is {img_tensor.shape}")
