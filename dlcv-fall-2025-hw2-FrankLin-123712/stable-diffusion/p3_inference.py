# ControlNet/p3_infer.py
import os, json, argparse
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

# repo-local imports (your tree)
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.controlnet.cldm import ControlNet  # we only need the ControlNet module

def load_ldm_from_cfg(cfg_path, sd_ckpt, device):
    cfg = OmegaConf.load(cfg_path)
    model = instantiate_from_config(cfg.model)
    # load SD weights
    sd = torch.load(sd_ckpt, map_location="cpu")
    if "state_dict" in sd:  # typical .ckpt structure
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ldm] loaded with {len(missing)} missing and {len(unexpected)} unexpected keys")
    model.eval().to(device)
    return model

def build_controlnet(device, hint_channels=3):
    """
    Build a ControlNet that matches SD v1.4 UNet topology.
    These hyperparams mirror cldm_v15 for SD1.4.
    """
    cn = ControlNet(
        image_size=64,          # latent spatial size for 512px images
        in_channels=4,          # latent channels
        model_channels=320,     # base channels of SD v1 UNet
        hint_channels=hint_channels,
        num_res_blocks=2,
        attention_resolutions=[4, 2, 1],
        dropout=0.0,
        channel_mult=[1, 2, 4, 4],
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=8,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=True,
        transformer_depth=1,
        context_dim=768,
        n_embed=None,
        legacy=False,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    )
    cn.to(device).eval()
    return cn

def load_controlnet_weights(controlnet, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    # be forgiving about different save formats
    if isinstance(sd, dict):
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "controlnet_state_dict" in sd:
            sd = sd["controlnet_state_dict"]
    missing, unexpected = controlnet.load_state_dict(sd, strict=False)
    print(f"[controlnet] loaded with {len(missing)} missing and {len(unexpected)} unexpected keys")

@torch.no_grad()
def run(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] args={vars(args)}")

    # 1) Load base SD (LatentDiffusion) from v1-inference.yaml
    model = load_ldm_from_cfg(args.config, args.sd_ckpt, device)

    # 2) Build ControlNet and load weights, then attach to model
    controlnet = build_controlnet(device, hint_channels=3)
    load_controlnet_weights(controlnet, args.control_ckpt)
    model.control_model = controlnet  # <-- this is the key line

    # we only need text embedder & VAE / UNet in eval mode
    model.cond_stage_model.eval()
    model.first_stage_model.eval()
    model.model.eval()

    # 3) Sampler
    sampler = DDIMSampler(model)

    # 4) IO prep
    os.makedirs(args.output_dir, exist_ok=True)
    # transforms for the control hint (source image)
    # Resize to latent resolution (512 -> 64), keep 3 channels, [0,1] float
    to_hint = T.Compose([
        T.Resize((args.img_size, args.img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.img_size),
        T.ToTensor(),  # keep [0,1], 3xHxW
    ])

    with open(args.json_path, "r") as f:
        lines = [json.loads(l) for l in f.read().splitlines() if l.strip()]

    if args.seed is not None and args.seed >= 0:
        torch.manual_seed(args.seed)

    print(f"Running DDIM Sampling with {args.steps+1} timesteps")
    # NB: DDIMSampler.sample expects "steps" (not steps+1). The printed string mirrors upstream logs.
    for rec in tqdm(lines, desc="[infer]"):
        src_name = rec["source"]
        tgt_name = rec["target"]
        prompt   = rec["prompt"]

        # 4.1) build conditioning
        cond_txt  = model.get_learned_conditioning([prompt]).to(device)
        uc_txt    = model.get_learned_conditioning([""]).to(device)

        # 4.2) load & prep control hint (source)
        src_path = os.path.join(args.input_dir, src_name)
        img = Image.open(src_path).convert("RGB")
        hint = to_hint(img).unsqueeze(0).to(device)  # (1,3,64,64)

        # 4.3) DDIM sample with CFG
        H = W = args.img_size // 8  # latent spatial
        shape = (model.channels, H, W)  # (4,64,64) for 512px

        cond   = {"c_crossattn": [cond_txt], "control": hint}
        uncond = {"c_crossattn": [uc_txt],   "control": hint}
        
        print("hint shape:", hint.shape)          # expect [1, 3, 512, 512]
        print("latent shape:", shape)             # expect (4, 64, 64)

        samples, _ = sampler.sample(
            S=args.steps,                  # number of DDIM steps
            batch_size=1,
            shape=shape,
            conditioning=cond,
            verbose=False,
            unconditional_guidance_scale=args.cfg,
            unconditional_conditioning=uncond,
        )

        # 4.4) Decode latents to image in [-1,1] -> save to PNG [0,255]
        x = model.decode_first_stage(samples)
        x = (x.clamp(-1, 1) + 1.0) / 2.0  # [0,1]
        x = (x * 255.0).round().squeeze(0).permute(1, 2, 0).cpu().numpy().astype("uint8")
        out_path = os.path.join(args.output_dir, tgt_name)
        Image.fromarray(x).save(out_path)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--json_path", type=str, default="./hw2_data/fill50k/testing/prompt.json")
    ap.add_argument("--input_dir", type=str, default="./hw2_data/fill50k/testing/source")
    ap.add_argument("--output_dir", type=str, default="./output/p3")
    ap.add_argument("--config", type=str, default="./stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
    ap.add_argument("--sd_ckpt", type=str, default="./stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt")
    ap.add_argument("--control_ckpt", type=str, default="./ckpt_p3.pth")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    run(parse_args())
