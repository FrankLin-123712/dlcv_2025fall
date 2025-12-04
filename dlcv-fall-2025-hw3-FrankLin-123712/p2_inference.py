import argparse
import json
import logging
from pathlib import Path
from typing import List

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from p2_model import ViLModel
from p2_train import build_transform, collate_samples, ImageCaptionDataset
from tokenization_qwen3 import Qwen3Tokenizer


LOGGER = logging.getLogger(__name__)


class InferenceImageDataset(Dataset):
    """Dataset that loads images from a flat folder."""

    def __init__(self, image_dir: str, transform) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.transform = transform
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory {self.image_dir} not found.")
        valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_paths: List[Path] = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in valid_suffixes]
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found under {self.image_dir}.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image_tensor = self.transform(image)
        return {"image": image_tensor, "file_name": path.stem}


def collate_inference(batch: List[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch], dim=0)
    file_names = [item["file_name"] for item in batch]
    return {"images": images, "file_names": file_names}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("P2 Inference - Vision-Language Captioning")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="hw3_data/p2_data")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to a folder that directly contains images for inference.",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--decoder_ckpt", type=str, default=None)
    parser.add_argument("--output", type=str, default="p2_predictions.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--projector_dim", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default=None,
        help="Comma separated list of attention projection names to apply LoRA.",
    )
    parser.add_argument("--lora_train_bias", dest="lora_train_bias", action="store_true")
    parser.add_argument("--no_lora_train_bias", dest="lora_train_bias", action="store_false")
    parser.set_defaults(lora_train_bias=None)
    parser.add_argument("--log", action="store_true")
    return parser.parse_args()


def setup_logging(enable_log: bool) -> None:
    level = logging.INFO if enable_log else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=level,
    )


def load_checkpoint(ckpt_path: str) -> tuple[dict, dict]:
    state = torch.load(ckpt_path, map_location="cpu")
    if "trainable" in state:
        return state["trainable"], state.get("model_config", {})
    if "model" in state:
        LOGGER.warning("Checkpoint contains full model weights; using them as trainable state.")
        return state["model"], state.get("model_config", {})
    return state, {}


def decode_tokens(tokenizer: Qwen3Tokenizer, token_ids: torch.Tensor, bos_id: int, eos_id: int) -> str:
    tokens = token_ids.tolist()
    if tokens and tokens[0] == bos_id:
        tokens = tokens[1:]
    if eos_id in tokens:
        eos_idx = tokens.index(eos_id)
        tokens = tokens[:eos_idx]
    text = tokenizer.decode(tokens).strip()
    return text


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    tokenizer = Qwen3Tokenizer("vocab.json", "merges.txt")
    default_pad = tokenizer.encoder["<|endoftext|>"]
    default_bos = tokenizer.encoder["<|im_start|>"]
    default_eos = tokenizer.encoder["<|im_end|>"]

    trainable_state, saved_config = load_checkpoint(args.checkpoint)

    print(f"model config : \r {saved_config}")
    
    pad_id = saved_config.get("pad_token_id", default_pad)
    bos_id = saved_config.get("bos_token_id", default_bos)
    eos_id = saved_config.get("eos_token_id", default_eos)

    transform = build_transform()
    if args.image_dir:
        dataset = InferenceImageDataset(args.image_dir, transform)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_inference,
        )
    else:
        dataset = ImageCaptionDataset(
            data_dir=args.data_dir,
            split=args.split,
            tokenizer=tokenizer,
            transform=transform,
            max_text_len=max(args.max_new_tokens + 2, 64),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda batch: collate_samples(batch, pad_id),
        )

    projector_dim = args.projector_dim or saved_config.get("projector_out_dim")
    lora_rank = args.lora_rank if args.lora_rank is not None else saved_config.get("lora_rank", 16)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else saved_config.get("lora_alpha", 16.0)
    lora_dropout = args.lora_dropout if args.lora_dropout is not None else saved_config.get("lora_dropout", 0.0)
    saved_targets = saved_config.get("lora_targets", ["q_proj", "v_proj"])
    if isinstance(saved_targets, str):
        saved_targets = [t.strip() for t in saved_targets.split(",") if t.strip()]
    if args.lora_targets:
        lora_targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
    else:
        lora_targets = tuple(saved_targets)
    if args.lora_train_bias is None:
        lora_train_bias = saved_config.get("lora_train_bias", False)
    else:
        lora_train_bias = args.lora_train_bias

    decoder_ckpt = args.decoder_ckpt or saved_config.get("decoder_ckpt") or "hw3_data/p2_data/decoder_model.bin"
    vision_encoder_name = saved_config.get("vision_encoder_name", "vit_large_patch14_clip_224.openai")

    model = ViLModel(
        decoder_ckpt=decoder_ckpt,
        vision_encoder_name=vision_encoder_name,
        projector_out_dim=projector_dim,
        freeze_vision=True,
        freeze_decoder=True,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
        lora_train_bias=lora_train_bias,
    ).to(device)

    model.load_trainable_state_dict({k: v.to(device) for k, v in trainable_state.items()})
    model.eval()
    
    print(f"the shape of projector : {model.projector.weight.shape}")

    predictions = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images = batch["images"].to(device, non_blocking=True)
            token_ids = model.generate(images, max_new_tokens=args.max_new_tokens)

            for file_name, tokens in zip(batch["file_names"], token_ids):
                caption = decode_tokens(tokenizer, tokens, bos_id, eos_id)
                predictions[Path(file_name).stem] = caption

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    LOGGER.info("Predictions saved to %s", output_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.log)
    run_inference(args)


if __name__ == "__main__":
    main()
