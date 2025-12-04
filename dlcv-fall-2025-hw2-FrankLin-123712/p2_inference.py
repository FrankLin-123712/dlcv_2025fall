import os, sys, argparse, datetime
import logging
import numpy as np

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

from tqdm import tqdm

from UNet import UNet
from utils import beta_scheduler

class DDIM(nn.Module):
    def __init__(self, nn, device, n_T, betas, n_steps=50, eta=0.0):
        super(DDIM, self).__init__()
        self.nn = nn.to(device)
        self.device = device
        self.n_T = n_T

        # obtain coefficents required for ddim
        self.beta = beta_scheduler(n_T, betas[0], betas[1])
        self.alpha = (1.0 - self.beta)
        log_a = torch.log(self.alpha)
        self.ab = torch.cumsum(log_a, dim=0).exp()

        self.n_steps = n_steps
        self.step_size = n_T//n_steps
        self.eta = torch.tensor(eta, dtype=torch.float32, device=device)

        # obtain time steps for ddim
        self.ts_seq = None
        self.pts_seq = None
        self.gen_ts()

    def gen_ts(self):
        """
        generate the current and previous timesteps with uniform step size.
        """
        self.ts_seq = np.array(list(range(0, self.n_T, self.step_size))) + 1
        self.pts_seq = np.append(np.array([0]), self.ts_seq[:-1])

    @torch.no_grad()
    def ddim_step(self, x_t, t, clip_denoised=True):
        """
        one step ddim.
        """
        pts = self.pts_seq[np.where(self.ts_seq == t)[0][0]]
        
        ab = self.ab[t-1]  # \bar{\alpha}
        abp = self.ab[pts-1] if pts > 0 else torch.tensor([1.0], dtype=torch.float64, device=self.device) # previous step of \bar{\alpha}

        sq_ab = torch.sqrt(ab) # \sqrt{\bar{\alpha}}
        sq_abp = torch.sqrt(abp)  # previous step of \sqrt{\bar{\alpha}}

        sq_mab = torch.sqrt(1.0 - ab) # \sqrt{1-\bar{\alpha}}
        sq_mabp = torch.sqrt(1.0 - abp) # previous step of \sqrt{1-\bar{\alpha}}

        B = x_t.shape[0]
        temb = torch.full((B,), t, device=self.device, dtype=torch.float32)

        eps = self.nn(x_t, temb)

        # compute x0
        x0 = (x_t - sq_mab * eps) / sq_ab

        if clip_denoised:
            x0 = torch.clamp(x0, -1.0, 1.0)
        
        # compute sigma_t 
        logging.debug(f"######### x_t : {t} ###########")
        logging.debug(f"self.eta.device : {self.eta.device}")
        logging.debug(f"sq_mabp.device : {sq_mabp.device }")
        logging.debug(f"sq_mab.device : {sq_mab.device}")
        logging.debug(f"ab.device : {ab.device}")
        logging.debug(f"abp.device : {abp.device}")
        sigma_t = self.eta * torch.sqrt(
            ((1.0 - abp)/(1.0 - ab)) * (1.0 - (ab/abp))
        )

        # compute x_pt
        dir_xt = torch.sqrt(1.0 - abp - sigma_t**2) * eps
        x_prev = sq_abp * x0 + dir_xt + sigma_t * torch.randn_like(x0)
        
        return x_prev

    
    @torch.no_grad()
    def ddim_sample(self, x_t):
        """
        generate n_sample of images by n_steps of ddim backward process.
        """
        logging.info("start ddim sampling process")
        for i, t in enumerate(reversed(self.ts_seq)):
            x_t = self.ddim_step(x_t, t, False)
        
        return x_t

def slerp(z0: torch.Tensor, z1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between z0 and z1.
    z0, z1: shape (B, C, H, W) or (1, C, H, W)
    alpha:  shape (B,) or (1,) in [0, 1]
    """
    # flatten per-sample for角度計算，但最後仍用原張量組合
    v0 = z0.view(z0.shape[0], -1)
    v1 = z1.view(z1.shape[0], -1)

    # cosθ = <v0, v1> / (||v0|| ||v1||)
    dot = torch.sum(v0 * v1, dim=1) / (v0.norm(dim=1) * v1.norm(dim=1) + 1e-12)
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.acos(dot)  # shape (B,)

    # 避免 sinθ≈0 時數值不穩，退化成線性插值
    sin_theta = torch.sin(theta)
    small = (sin_theta.abs() < 1e-6).float()

    # slerp 係數
    a = alpha.view(-1)                    # (B,)
    w0 = torch.sin((1.0 - a) * theta) / (sin_theta + 1e-12)
    w1 = torch.sin(a * theta) / (sin_theta + 1e-12)

    # 線性插值係數（for small-angle fallback）
    lw0 = (1.0 - a)
    lw1 = a

    # 係數升維以配張量形狀
    while w0.ndim < z0.ndim:
        w0 = w0.unsqueeze(-1); w1 = w1.unsqueeze(-1)
        lw0 = lw0.unsqueeze(-1); lw1 = lw1.unsqueeze(-1)

    # small==1 用 lerp，否則用 slerp
    out = (1 - small)[:, None, None, None] * (w0 * z0 + w1 * z1) + \
          small[:, None, None, None] * (lw0 * z0 + lw1 * z1)
    return out

def norm(img):
    min = torch.min(img)
    max = torch.max(img)
    norm_img = (img - min) / (max - min + 1e-9)

    return norm_img

def parse_args():
    ap = argparse.ArgumentParser('P2 DDIM algorithm for generating face images')
    
    ap.add_argument("--debug", action="store_true", help="enable debug logging")
    ap.add_argument("--mode", type=int, choices=[1, 2, 3], required=True, 
                    help="The mode select for the format of images you want to show.")
    ap.add_argument("--gpu", type=int, default=0)

    ap.add_argument("--gt", type=str, default="./hw2_data/face/GT")
    ap.add_argument("--noise", type=str, default="./hw2_data/face/noise")
    ap.add_argument("--model", type=str, default="./hw2_data/face/UNet.pt")
    
    ap.add_argument("--save_path", type=str, default="./output/p2/")
    ap.add_argument("--eta", type=float, default=0.0)
    
    return ap.parse_args()

def main():

    torch.manual_seed(42)

    ########### parse arguments ##############
    args = parse_args()

    ########### set up gpu ###########
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ############ set up logging #################
    # time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # LOG_FILE = f"./log/p2/ddim_{time}.log"
    # LOG_LEVEL = logging.DEBUG if args.debug else logging.INFO
    # os.makedirs("./log/p2", exist_ok=True)
    # logging.basicConfig(
    #     level=LOG_LEVEL,
    #     format="%(asctime)s [%(levelname)s] %(message)s",
    #     handlers=[
    #         logging.FileHandler(LOG_FILE),
    #         logging.StreamHandler(sys.stdout)
    #     ]
    # )
    logging.info(f"device setted as {device}")
    # logging.info(f"save log to {LOG_FILE}")
    # logging.info(f"log level : {LOG_LEVEL}")
    logging.info(f"parameters : {vars(args)}")

    ############## create DDIM model #################
    unet = UNet().to(device)
    weights = torch.load(args.model, map_location=device)
    logging.info(f"load model weights from {args.model}")
    msg = unet.load_state_dict(weights, strict=True)
    unet.eval()
    logging.info(f"{msg}")
    
    ############## start ddim sampling ##############
    ddim = DDIM(
        nn=unet,
        device=device,
        n_T=1000,
        betas=[1e-4, 2e-2],
        n_steps=50,
        eta=args.eta
    ).to(device)
    ddim.eval()
    logging.info(f"ddim constructed and moved to {device}")
    
    os.makedirs(args.save_path, exist_ok=True)
    ############# mode 1 : generate images for 00.pt ~ 09.pt with eta=0 #################
    if(args.mode == 1):
        logging.info("generate face images for 00.pt ~ 09.pt with eta=0.")
        for i in range(10):
            noise_path = os.path.join(args.noise, f"{i:02d}.pt")
            if not os.path.exists(noise_path):
                logging.error(f"{noise_path} does not eixst.")
                sys.exit(1)

            noise_tensor = torch.load(noise_path, map_location=device)
            logging.info(f"load noise from {noise_path}")
            gen_img = ddim.ddim_sample(noise_tensor)

            ############# store the generated images ################
            min = torch.min(gen_img)
            max = torch.max(gen_img)
            norm_img = (gen_img - min) / (max - min + 1e-9)
            img_path = os.path.join(args.save_path, f"{i:02d}.png")
            save_image(norm_img, img_path)

            logging.info(f"save generated image with noise {i:02d}.pt to {img_path}.")

    ############# mode 2 : generate report images ########
    elif(args.mode == 2):
        logging.info("generate face images of noise 00.pt ~ 03.pt with different eta.")
        face_img_tensor = []
        for i in range(5):
            eta_i = torch.tensor(args.eta*i, dtype=torch.float32, device=device)
            ddim.eta = eta_i
            logging.info(f"eta now set as {ddim.eta}")
            for k in range(4):
                noise_path = os.path.join(args.noise, f"{k:02d}.pt")
                if not os.path.exists(noise_path):
                    logging.error(f"{noise_path} does not eixst.")
                    sys.exit(1)
                noise_tensor = torch.load(noise_path, map_location=device)
                logging.info(f"load noise from {noise_path}")
                gen_img = ddim.ddim_sample(noise_tensor)
                gen_img = norm(gen_img)
                logging.info(f"the shape of gen_img : {gen_img.shape}")
                face_img_tensor.append(gen_img)
        img_grid = make_grid(torch.cat(face_img_tensor, dim=0), nrow=4)
        fname = f"face_grid.png"
        save_image(img_grid, os.path.join(args.save_path, fname))
        logging.info(f"save face grid to {os.path.join(args.save_path, fname)}")

    elif(args.mode == 3):
        logging.info("generate face image of interpolation of noise 00.pt ~ 01.pt")
        
        # read 00.pt 與 01.pt
        n0_path = os.path.join(args.noise, "00.pt")
        n1_path = os.path.join(args.noise, "01.pt")
        if (not os.path.exists(n0_path)) or (not os.path.exists(n1_path)):
            logging.error("00.pt or 01.pt does not exist under --noise folder.")
            sys.exit(1)
        z0 = torch.load(n0_path, map_location=device).to(device)
        z1 = torch.load(n1_path, map_location=device).to(device)
        logging.info(f"loaded {n0_path} and {n1_path} with shapes {tuple(z0.shape)} {tuple(z1.shape)}")

        # α=0.0 ~ 1.0（共 11 個點）
        alphas = torch.linspace(0.0, 1.0, 11, device=device)

        # 逐 α 產生 slerp 與 lerp 的插值噪聲 -> DDIM 反推 -> 收集圖片
        slerp_imgs = []
        lerp_imgs  = []
        for a in alphas:
            # 讓 batch 維一致；noise 檔通常是 (1,C,H,W)，保留 B=1 即可
            B = z0.shape[0]
            a_vec = torch.full((B,), float(a.item()), device=device)

            # slerp noise
            z_s = slerp(z0, z1, a_vec)
            x_s = ddim.ddim_sample(z_s)
            # lerp noise
            z_l = (1.0 - a) * z0 + a * z1
            x_l = ddim.ddim_sample(z_l)

            # 0-1 normalize 以利排圖
            for x, bag in [(x_s, slerp_imgs), (x_l, lerp_imgs)]:
                mn, mx = torch.min(x), torch.max(x)
                x = (x - mn) / (mx - mn + 1e-9)
                bag.append(x)

            logging.info(f"alpha={a.item():.1f} done")

        # 存兩張 grid：slerp 與 linear
        slerp_grid = make_grid(torch.cat(slerp_imgs, dim=0), nrow=len(alphas))
        lerp_grid  = make_grid(torch.cat(lerp_imgs,  dim=0), nrow=len(alphas))
        os.makedirs(args.save_path, exist_ok=True)
        slerp_path = os.path.join(args.save_path, "interp_slerp_00_01.png")
        lerp_path  = os.path.join(args.save_path, "interp_linear_00_01.png")
        save_image(slerp_grid, slerp_path)
        save_image(lerp_grid,  lerp_path)
        logging.info(f"saved slerp grid to {slerp_path}")
        logging.info(f"saved linear grid to {lerp_path}")
    else:
        logging.error("Invalid mode.")
        logging.error("Program aborted.")
        sys.exit(1)
    
    

    logging.info("DDIM completed.")
        
    
if __name__ == '__main__':
    main()