import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from p2_model import ViLModel
from tokenization_qwen3 import Qwen3Tokenizer


LOGGER = logging.getLogger(__name__)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


@dataclass
class CaptionSample:
    image: torch.Tensor
    caption_ids: List[int]
    image_id: int
    file_name: str
    caption_text: str


class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer: Qwen3Tokenizer,
        transform: transforms.Compose,
        max_text_len: int,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_text_len = max_text_len

        annotation_path = self.data_dir / f"{split}.json"
        with open(annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        self.annotations = annotations["annotations"]
        self.id_to_name = {item["id"]: item["file_name"] for item in annotations["images"]}
        self.images_dir = self.data_dir / "images" / split

        self.pad_id = tokenizer.encoder["<|endoftext|>"]
        self.bos_id = tokenizer.encoder["<|im_start|>"]
        self.eos_id = tokenizer.encoder["<|im_end|>"]

    def __len__(self) -> int:
        return len(self.annotations)

    def _encode_caption(self, caption: str) -> List[int]:
        token_ids = [self.bos_id]
        token_ids.extend(self.tokenizer.encode(caption))
        token_ids.append(self.eos_id)

        if len(token_ids) > self.max_text_len:
            token_ids = token_ids[: self.max_text_len - 1] + [self.eos_id]
        return token_ids

    def __getitem__(self, idx: int) -> CaptionSample:
        ann = self.annotations[idx]
        image_id = ann["image_id"]
        file_name = self.id_to_name[image_id]

        image_path = self.images_dir / file_name
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        caption_text = ann["caption"]
        caption_ids = self._encode_caption(caption_text)

        return CaptionSample(
            image=image_tensor,
            caption_ids=caption_ids,
            image_id=image_id,
            file_name=file_name,
            caption_text=caption_text,
        )


def collate_samples(batch: List[CaptionSample], pad_id: int) -> Dict[str, torch.Tensor]:
    images = torch.stack([sample.image for sample in batch], dim=0)

    lengths = [len(sample.caption_ids) for sample in batch]
    max_len = max(lengths)
    captions = torch.full((len(batch), max_len), pad_id, dtype=torch.long)

    for i, sample in enumerate(batch):
        captions[i, : lengths[i]] = torch.tensor(sample.caption_ids, dtype=torch.long)

    return {
        "images": images,
        "caption_ids": captions,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "image_ids": [sample.image_id for sample in batch],
        "file_names": [sample.file_name for sample in batch],
        "captions": [sample.caption_text for sample in batch],
    }


def create_dataloaders(
    data_dir: str,
    tokenizer: Qwen3Tokenizer,
    batch_size: int,
    num_workers: int,
    max_text_len: int,
) -> Tuple[DataLoader, DataLoader]:
    transform = build_transform()

    train_dataset = ImageCaptionDataset(
        data_dir=data_dir,
        split="train",
        tokenizer=tokenizer,
        transform=transform,
        max_text_len=max_text_len,
    )
    val_dataset = ImageCaptionDataset(
        data_dir=data_dir,
        split="val",
        tokenizer=tokenizer,
        transform=transform,
        max_text_len=max_text_len,
    )

    pad_id = tokenizer.encoder["<|endoftext|>"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_samples(batch, pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_samples(batch, pad_id),
    )

    return train_loader, val_loader


def save_checkpoint(
    model: ViLModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: Path,
    model_config: dict,
    best: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable_state = model.get_trainable_state_dict()
    state = {
        "trainable": trainable_state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "model_config": model_config,
    }
    ckpt_path = output_dir / ("best.pt" if best else f"epoch_{epoch}.pt")
    torch.save(state, ckpt_path)
    LOGGER.info("Saved checkpoint to %s", ckpt_path)


def validate(model: ViLModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            captions = batch["caption_ids"].to(device, non_blocking=True)

            loss, logits = model(images, captions)

            token_mask = captions[:, 1:] != model.pad_token_id
            total_tokens += token_mask.sum().item()
            total_loss += loss.item() * token_mask.sum().item()

    return total_loss / max(total_tokens, 1)


def train(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    tokenizer = Qwen3Tokenizer("vocab.json", "merges.txt")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_text_len=args.max_text_len,
    )

    lora_targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())

    model = ViLModel(
        decoder_ckpt=args.decoder_ckpt,
        projector_out_dim=args.projector_dim,
        freeze_vision=args.freeze_vision,
        freeze_decoder=args.freeze_decoder,
        pad_token_id=tokenizer.encoder["<|endoftext|>"],
        bos_token_id=tokenizer.encoder["<|im_start|>"],
        eos_token_id=tokenizer.encoder["<|im_end|>"],
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=lora_targets,
        lora_train_bias=args.lora_train_bias,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("Trainable parameters: %.2fM", trainable_params / 1e6)

    model_config = {
        "vision_encoder_name": getattr(model, "vision_encoder_name", None),
        "projector_out_dim": model.projector.out_features,
        "decoder_ckpt": args.decoder_ckpt,
        "pad_token_id": model.pad_token_id,
        "bos_token_id": model.bos_token_id,
        "eos_token_id": model.eos_token_id,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_targets": list(lora_targets),
        "lora_train_bias": args.lora_train_bias,
        "freeze_vision": args.freeze_vision,
        "freeze_decoder": args.freeze_decoder,
    }

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    best_val = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        running_loss = 0.0
        running_tokens = 0

        for step, batch in enumerate(progress, start=1):
            images = batch["images"].to(device, non_blocking=True)
            captions = batch["caption_ids"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                loss, _ = model(images, captions)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            token_mask = captions[:, 1:] != model.pad_token_id
            tokens = token_mask.sum().item()
            running_tokens += tokens
            running_loss += loss.item() * tokens

            if step % args.log_interval == 0:
                avg_loss = running_loss / max(running_tokens, 1)
                progress.set_postfix({"loss": f"{avg_loss:.4f}"})

        val_loss = validate(model, val_loader, device)
        LOGGER.info("Epoch %d validation loss: %.4f", epoch, val_loss)

        save_checkpoint(model, optimizer, epoch, output_dir, model_config, best=False)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, output_dir, model_config, best=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train PEFT Vision-Language model for captioning.")
    parser.add_argument("--data_dir", type=str, default="hw3_data/p2_data")
    parser.add_argument("--output_dir", type=str, default="ckpt/p2")
    parser.add_argument("--decoder_ckpt", type=str, default="hw3_data/p2_data/decoder_model.bin")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_text_len", type=int, default=64)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projector_dim", type=int, default=None)
    parser.add_argument("--freeze_vision", dest="freeze_vision", action="store_true")
    parser.add_argument("--no_freeze_vision", dest="freeze_vision", action="store_false")
    parser.add_argument("--freeze_decoder", dest="freeze_decoder", action="store_true")
    parser.add_argument("--no_freeze_decoder", dest="freeze_decoder", action="store_false")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,v_proj",
        help="Comma separated list of attention projection names to apply LoRA.",
    )
    parser.add_argument("--lora_train_bias", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.set_defaults(freeze_vision=True, freeze_decoder=True)
    return parser.parse_args()


def setup_logging(enable_log: bool) -> None:
    level = logging.INFO if enable_log else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=level,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        LOGGER.setLevel(logging.DEBUG)

    train(args)


if __name__ == "__main__":
    main()
