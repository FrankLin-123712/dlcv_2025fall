import os, sys, logging, argparse, datetime


import torch
from torchvision.utils import make_grid, save_image

from p1_model import DDPM, ContextUnet

@torch.no_grad()
def output_500_img(model, guide_w, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    mnistm_dir = os.path.join(save_dir, "mnistm")
    svhn_dir = os.path.join(save_dir, "svhn")
    os.makedirs(mnistm_dir, exist_ok=True)
    os.makedirs(svhn_dir, exist_ok=True)
    
    torch.manual_seed(42)
    n_sample = 500
    n_classes = 10
    n_per_class = n_sample // n_classes
    img_size = (3, 28, 28)
    C, H, W = img_size
    
    logging.info(f"Generating {n_sample} images with guide_w : {guide_w} and size {img_size}")
    x_gen, _ = model.sample(n_sample, img_size, device, guide_w)
    logging.info("Image generation complete.")
    
    # Reorder so images are grouped by digit class
    # 1. reshape → (num_noise_instances, num_classes, C, H, W)
    x_gen = x_gen.view(n_per_class, n_classes, C, H, W)
    # 2. permute → (num_classes, num_noise_instances, C, H, W)
    x_gen = x_gen.permute(1, 0, 2, 3, 4)
    # 3. flatten → (n_sample, C, H, W)
    x_gen = x_gen.reshape(n_sample, C, H, W)

    # normalize from [-1,1] to [0,1]
    x_gen = torch.clamp(x_gen, -1, 1)
    x_gen = (x_gen + 1) / 2.0

    # now first 50 are digit 0, next 50 are digit 1, etc.
    for digit in range(n_classes):
        start = digit * n_per_class
        end = (digit + 1) * n_per_class
        imgs = x_gen[start:end]

        out_dir = mnistm_dir if digit % 2 == 0 else svhn_dir
        for idx, img in enumerate(imgs, 1):
            fname = f"{digit}_{idx:03d}.png"
            save_image(img, os.path.join(out_dir, fname))

    logging.info(f"Saved 500 images under {save_dir}")

@torch.no_grad()
def show_100_img_make_grid(model, guide_w, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.manual_seed(42)

    n_sample = 100
    n_classes = 10
    img_size = (3, 28, 28)
    C, H, W = img_size

    logging.info(f"Generating {n_sample} images with guide_w : {guide_w} and size {img_size}")
    x_gen, x_buf = model.sample(n_sample, img_size, device, guide_w)

    x_gen = x_gen.view(n_sample // n_classes, n_classes, C, H, W)
    x_gen = x_gen.permute(1, 0, 2, 3, 4)
    x_gen = x_gen.reshape(n_sample, C, H, W)

    x_gen = torch.clamp(x_gen, -1, 1)
    x_gen = (x_gen + 1) / 2.0

    grid = make_grid(x_gen, nrow=10)
    grid_path = os.path.join(save_dir, f"digits_grid_{time}.png")
    save_image(grid, grid_path)
    logging.info(f"Saved 10×10 grid visualization to {grid_path}")

    x_buf = torch.tensor(x_buf)  # convert np.array back to torch tensor
    num_steps, total_samples, C, H, W = x_buf.shape
    x_buf = x_buf.view(num_steps, n_sample // n_classes, n_classes, C, H, W)
    x_buf = x_buf.permute(0, 2, 1, 3, 4, 5)  # (steps, class, noise, C, H, W)
    # now: x_buf[step_idx, digit, noise_idx, C, H, W]

    # Pick first “0” and “1” (noise_idx = 0)
    for digit in [0, 1]:
        imgs_per_step = x_buf[:, digit, 0]  # shape: (num_steps, 3, 28, 28)
        imgs_per_step = torch.clamp(imgs_per_step, -1, 1)
        imgs_per_step = (imgs_per_step + 1) / 2.0

        index = digit if digit == 0 else 10
        img_clean = x_gen[[index]].to(imgs_per_step.device)
        img_reverse_process = torch.cat([imgs_per_step, img_clean], dim=0)
        # Make a 1×6 grid to show progression from noise → clear image
        grid = make_grid(img_reverse_process, nrow=img_reverse_process.shape[0])
        fname = f"reverse_process_{digit}.png"
        save_image(grid, os.path.join(save_dir, fname))
        logging.info(f"Saved reverse process visualization for digit {digit} to {fname}")

    logging.info("All visualizations completed.")

def parse_args():
    ap = argparse.ArgumentParser("P1 Inference scripts.")
    ap.add_argument("--debug", action="store_true", help="enable debug mode")
    ap.add_argument("--mode", type=int, choices=[1, 2], required=True, 
                    help="The mode represents for the format of images you want to show.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model_path", type=str, default="./ckpts/p1/")
    ap.add_argument("--save_path", type=str, default="./output/p1")
    ap.add_argument("--guide_w", type=float, default=2.0)
    return ap.parse_args()

def main():

    ########### parse arguments ##############
    args = parse_args()
    ########### set up gpu ###########
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ############ set up logging #################
    # time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # os.makedirs("./log/p1", exist_ok=True)
    # LOG_FILE = f"./log/p1/inference_{time}.log"
    # LOG_LEVEL = logging.DEBUG if args.debug else logging.INFO
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
    # logging.info(f"logging level {LOG_LEVEL}")
    logging.info(f"parameters : {vars(args)}")
    
    ################ load model ##################
    # ----- Step 1. Load checkpoint -----
    ckpt = torch.load(args.model_path, map_location=device)

    # ----- Step 2. Rebuild model -----
    m_args = ckpt["args"]
    model = DDPM(
        nn_model=ContextUnet(
            in_channels=m_args["in_channels"],
            n_feat=m_args["n_feat"],
            n_classes=m_args["n_classes"]
        ),
        betas=m_args["betas"],
        n_T=m_args["n_T"],
        device=device,
        drop_prob=m_args["drop_prob"]
    ).to(device)

    # ----- Step 3. Load weights -----
    model.load_state_dict(ckpt["model"])
    model.eval()  # Set to evaluation mode
    logging.info(f"Loaded model with parameters : {m_args}")
    logging.info(f"Loaded model from {args.model_path}")

    ############### inference #################
    if(args.mode == 1):
        """ 
        mode 1 : sample 500 output images for 
        digits 0, 2, 4, 6, 8 with mnistm style 
        and digits 1, 3, 5, 7, 9 with svhn style.
        """
        logging.info("############ Mode 1 ##############")
        output_500_img(model, args.guide_w, device, args.save_path)

    elif(args.mode == 2):
        """
        mode 2 : sample 100 output images for digits 0~9 with different noise 
        and make a grid and visualize a total of six images  in the reverse 
        process of the first “0” and first “1” in your outputs grid.
        """
        show_100_img_make_grid(model, args.guide_w, device, args.save_path)

    else:
        logging.error(f"Invalid mode : {args.mode}")
        logging.error("Program terminated.")
        sys.exit(1)
        
    
    logging.info("P1 inference completed.") 


if __name__ == '__main__':
    main()
