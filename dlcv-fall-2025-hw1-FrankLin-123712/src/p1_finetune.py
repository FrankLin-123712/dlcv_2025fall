import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.models as models
from tqdm import tqdm

from p1_utils4ft import set_seed, get_officehome_loaders, evaluate_cls, save_ckpt, load_dino_backbone_weights
from p1_model import build_resnet50_head



def parse_args():
    ap = argparse.ArgumentParser("P1 Finetune ResNet50 on Office-Home (A–E)")
    ap.add_argument("--setting", choices=["A","B","C","D","E"], required=True,
                    help="A: scratch+FT, B: officialDINO+FT, C: yourSSL+FT, D: officialDINO+LP, E: yourSSL+LP")
    ap.add_argument("--data_root", type=str, default="./data_2025/p1_data/office")
    ap.add_argument("--out", type=str, default="./ckpts_p1")
    ap.add_argument("--ckpt_official", type=str, default="./official_dino_model.pth")
    ap.add_argument("--ckpt_user", type=str, default="./checkpoint.pth")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)                 # FT epochs
    ap.add_argument("--linear_probe_epochs", type=int, default=10)    # LP epochs
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-2)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--unfreeze", choices=["all","layer4"], default="all")
    return ap.parse_args()


def resolve_setting(s):
    if s == "A": return "scratch", False
    if s == "B": return "official", False
    if s == "C": return "user",     False
    if s == "D": return "official", True
    if s == "E": return "user",     True
    raise ValueError(s)

def param_groups(backbone, head, lr_bb, lr_hd, wd):
    def wd_ok(n,p): return p.dim()>1 and "bias" not in n and "bn" not in n and "norm" not in n and "downsample.1" not in n
    bb_wd, bb_nwd, hd_wd, hd_nwd = [], [], [], []
    for n,p in backbone.named_parameters(): (bb_wd if wd_ok(n,p) else bb_nwd).append(p)
    for n,p in head.named_parameters():     (hd_wd if wd_ok(n,p) else hd_nwd).append(p)
    return [
        {"params": bb_wd,  "lr": lr_bb, "weight_decay": wd},
        {"params": bb_nwd, "lr": lr_bb, "weight_decay": 0.0},
        {"params": hd_wd,  "lr": lr_hd, "weight_decay": wd},
        {"params": hd_nwd, "lr": lr_hd, "weight_decay": 0.0},
    ]

@torch.no_grad()
def _pred_correct_count(logits, y):
    return (logits.argmax(1) == y).sum().item()


def train_one_epoch(model, loader, opt, device, log_interval=20, label_smoothing=0.1):
    # print(f"[INFO] current device : {device}")
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    train_bar = tqdm(loader, desc="Train", leave=False)
    
    for step, (x,y) in enumerate(train_bar, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        loss.backward()
        opt.step()

        bs = y.size(0)
        n += bs
        loss_sum += loss.item()*bs
        correct += _pred_correct_count(logits, y)

        if (step%log_interval == 0) or (step == len(loader)):
            avg_loss = loss_sum / n
            acc = correct / n
            train_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.2%}")

    avg_loss = loss_sum / max(1, n)
    return avg_loss

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, num_classes = get_officehome_loaders(
        args.data_root, args.batch_size, args.num_workers
    )
    
    model, backbone, head = build_resnet50_head(num_classes)

    # load checkpoints while everything is still on CPU
    load_mode, lp_only = resolve_setting(args.setting)
    if load_mode == "official":
        load_dino_backbone_weights(backbone, args.ckpt_official)
    elif load_mode == "user":
        load_dino_backbone_weights(backbone, args.ckpt_user)

    model.to(device)
    print(f"[INFO] model moved to device : {device}")

    # ---------- Linear probe (head only) ----------
    for p in backbone.parameters(): p.requires_grad = False
    opt_lp = optim.SGD([{"params": head.parameters(), "lr": args.lr_head, "weight_decay": args.wd}],
                       momentum=args.momentum, nesterov=True)

    best = 0.0
    for ep in range(1, args.linear_probe_epochs+1):
        tr = train_one_epoch(model, train_loader, opt_lp, device)
        acc = evaluate_cls(model, val_loader, device)
        if ep == 1:
            print("[INFO] Save model at epoch 1")
            save_ckpt({"epoch": 0, "model": model.state_dict(), "args": vars(args)}, args.out, is_best=False, filename="epoch1.pth")
        save_ckpt({"epoch": ep, "model": model.state_dict(), "args": vars(args)}, args.out, is_best=acc>best)
        best = max(best, acc)
        print(f"[LP] {ep:03d}/{args.linear_probe_epochs}  loss={tr:.4f}  val@1={acc:.2%}")

    if lp_only:
        print(f"[DONE] {args.setting}: linear-probe only. Best val@1={best:.4f}")
        return

    # ---------- Full finetune ----------
    for p in backbone.parameters(): p.requires_grad = True
    if args.unfreeze == "layer4":
        for n,p in backbone.named_parameters():
            p.requires_grad = ("layer4" in n) or ("bn" in n)

    opt = optim.SGD(param_groups(backbone, head, args.lr_backbone, args.lr_head, args.wd),
                    momentum=args.momentum, nesterov=True)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(1, args.epochs+1):
        tr = train_one_epoch(model, train_loader, opt, device)
        acc = evaluate_cls(model, val_loader, device)
        sch.step()
        save_ckpt({"epoch": args.linear_probe_epochs+ep, "model": model.state_dict(), "args": vars(args)}, args.out, is_best=acc>best)
        best = max(best, acc)
        lrs = [f"{g['lr']:.2e}" for g in opt.param_groups]
        print(f"[FT] {ep:03d}/{args.epochs}  loss={tr:.4f}  val@1={acc:.4f}  LRs={lrs}")

    print(f"[DONE] {args.setting}. Best val@1={best:.4f}")

if __name__ == "__main__":
    main()

    
