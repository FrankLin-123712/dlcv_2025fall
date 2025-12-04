import argparse, os, csv, sys
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from p1_model import build_resnet50_head

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --------------------------
# Dataset
# --------------------------
class CSVImageList(Dataset):
    def __init__(self, csv_path: str, img_dir: str, image_size: int = 224):
        self.rows = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            # tolerate either 'id,filename' or 'id,filename,label'
            for row in reader:
                self.rows.append({"id": row["id"], "filename": row["filename"], "label": row["label"]})
        self.img_dir = img_dir
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        rid = self.rows[idx]["id"]
        fname = self.rows[idx]["filename"]
        label = self.rows[idx]["label"]
        path = os.path.join(self.img_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # fallback: black image if unreadable (won't crash)
            print("ERROR : cannot load image !")
            img = Image.new("RGB", (224, 224), (0,0,0))
        return self.tf(img), rid, fname, label

# --------------------------
# Checkpoint loading (robust to different key prefixes)
# --------------------------
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

def load_weights(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # print(ckpt)
    sd = None
    # support common containers
    if isinstance(ckpt, (dict, OrderedDict)) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        print("[ERROR] model doesn't exist in ckpts:{ckpts_path}")
    peek_state_dict(sd, limit=10000)

    msg = model.load_state_dict(sd)
    print(msg)

def viz_tsne(features, labels, out_path):
    tsne = TSNE(n_components=2, random_state=42)
    embedded_features = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=labels, cmap='tab20')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of learned features')
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] t-SNE visualization saved to {out_path}")



# --------------------------
# Inference
# --------------------------
@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = build_resnet50_head(args.num_classes)
    load_weights(model, args.ckpt)
    model.eval().to(device)

    ds = CSVImageList(args.csv_path, args.img_dir, image_size=224)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, 
                        pin_memory=torch.cuda.is_available())

    total = correct = 0
    preds = []
    features = []
    filenames = []
    
    for imgs, ids, fnames, labels in loader:

        ## forward
        imgs = imgs.to(device, non_blocking=True)
        ## get prediction and the output of second last layer
        logits, _features = model(imgs)
        y = logits.argmax(1).cpu().tolist()
        features.append(_features.cpu().numpy())
        filenames.extend(fnames)

        ## calculate accuracy 
        for _id, _fn, _y in zip(ids, fnames, y):
            preds.append((_id, _fn, int(_y)))
        labels_list = labels.cpu().tolist() if torch.is_tensor(labels) else [int(x) for x in labels]
        correct += sum(p == t for p, t in zip(y, labels_list)) 
        total += len(y)

    acc = correct/total
    print("***********************************")
    print(f"[INFO] Model Accuracy : {acc:.2%}")
    print("***********************************")
    
    # visualize t-SNE
    if args.tsne_viz_enabled:
        features = np.vstack(features)
        labels = [int(f.split('_')[0]) for f in filenames]
        viz_tsne(features, labels, args.tsne_out_path)

    # write output: id,filename,label  (no spaces)
    Path(os.path.dirname(args.out_csv) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "filename", "label"])
        writer.writerows(preds)


def parse_args():
    p = argparse.ArgumentParser()
    # TA passes these via hw1_1.sh
    p.add_argument("csv_path", type=str)
    p.add_argument("img_dir", type=str)
    p.add_argument("out_csv", type=str)
    # you control the rest
    p.add_argument("--ckpt", type=str, default="ckpts_p1_C/best.pth", help="path to your finetuned P1 checkpoint")
    p.add_argument("--num-classes", type=int, default=65)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--tsne_viz_enabled", action='store_true', help="whether to perform t-SNE visualization")
    p.add_argument("--tsne-out-path", type=str, default="tsne.png", help="path to save t-SNE visualization")
    return p.parse_args()



if __name__ == "__main__":
    args = parse_args()
    run(args)
    # model, _, _ = build_model(num_classes=65)
    # print(model.state_dict())
    # load_weights(model, './ckpts_p1_C_300ep_4e-2/best.pth')
