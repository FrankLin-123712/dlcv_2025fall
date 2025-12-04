import argparse
import json
import os
import random
import zipfile
from pathlib import Path
from typing import Any, Dict, List
import logging

import torch
from PIL import Image
from tqdm import tqdm
from transformers.generation.stopping_criteria import StoppingCriteriaList

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Problem 1 inference with optional VCD")
    parser.add_argument("--model_path", type=str, required=True, help="Path or repo id of pretrained LLaVA weights.")
    parser.add_argument("--model_base", type=str, default=None, help="Base model path when loading delta weights.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Annotation json (e.g. hw3_data/p1_data/val.json).")
    parser.add_argument("--images_root", type=str, required=True, help="Directory containing validation images.")
    parser.add_argument("--output", type=str, default="outputs/p1_pred.json", help="Path to write predictions.")
    parser.add_argument("--conv_mode", type=str, default="llava_v1", help="Conversation template key.")
    parser.add_argument("--max_new_tokens", type=int, default=8, help="Generation length for yes/no answers.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy).")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p during sampling.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k during sampling.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cd", action="store_true", default=False, help="Enable visual contrastive decoding.")
    parser.add_argument("--noise_step", type=int, default=500, help="DDPM noise step for VCD.")
    parser.add_argument("--cd_alpha", type=float, default=1.0, help="Alpha hyper-parameter for VCD.")
    parser.add_argument("--cd_beta", type=float, default=0.1, help="Beta hyper-parameter for VCD.")
    parser.add_argument("--device", type=str, default=None, help="Computation device (e.g. cuda:0 or cpu).")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_annotations(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(question: str, conv_mode: str, uses_image_tokens: bool) -> str:
    text = question.strip()
    if uses_image_tokens:
        prefix = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n"
    else:
        prefix = f"{DEFAULT_IMAGE_TOKEN}\n"
    prompt_question = f"{prefix}{text}\nPlease answer this question with one word."

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt_question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def normalize_answer(text: str) -> str:
    cleaned = text.strip().lower()
    if cleaned.startswith("yes"):
        return "yes"
    if cleaned.startswith("no"):
        return "no"
    if "yes" in cleaned and "no" not in cleaned:
        return "yes"
    if "no" in cleaned:
        return "no"
    return cleaned.split()[0] if cleaned else ""


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.use_cd:
        print("[VCD Enabled]")
        print(f" cd_alpha : {args.cd_alpha} | cd_beta : {args.cd_beta}")
        evolve_vcd_sampling()

    disable_torch_init()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)

    model_path = os.path.expanduser(args.model_path)
    if os.path.isfile(model_path) and zipfile.is_zipfile(model_path):
        target_dir = Path(model_path).with_suffix("")
        if not target_dir.exists():
            print(f"Extracting model archive from {model_path} to {target_dir} ...")
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(model_path) as zf:
                zf.extractall(target_dir)
        model_path = str(target_dir)

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device_map=None,
        device=device_str,
    )
    model.eval()
    model.to(device=device)

    annotations = load_annotations(args.annotation_file)

    predictions: List[Dict[str, Any]] = []
    num_correct = 0

    for ann in tqdm(annotations, desc="Running inference"):
        image_id = ann["image_source"]
        question = ann["question"]
        answer = ann.get("answer", "").strip().lower()

        prompt = build_prompt(question, args.conv_mode, getattr(model.config, "mm_use_im_start_end", False))
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        input_ids = input_ids.to(device=device)

        image_path = Path(args.images_root) / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_tensor = image_tensor.to(device=device, dtype=model.dtype)

        image_tensor_cd = None
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor.detach().cpu(), args.noise_step)
            image_tensor_cd = image_tensor_cd.to(device=device, dtype=model.dtype)

        conv = conv_templates[args.conv_mode]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = (
            StoppingCriteriaList([KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)])
            if stop_str
            else None
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor.unsqueeze(0),
                images_cd=(image_tensor_cd.unsqueeze(0) if image_tensor_cd is not None else None),
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
                do_sample=args.use_cd,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )

        generated_tokens = output_ids[:, input_ids.shape[1]:]
        output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

        if stop_str and output_text.endswith(stop_str):
            output_text = output_text[: -len(stop_str)].strip()

        normalized = normalize_answer(output_text)
        if normalized not in {"yes", "no"}:
            lower = output_text.lower()
            if "yes" in lower and "no" not in lower:
                normalized = "yes"
            elif "no" in lower and "yes" not in lower:
                normalized = "no"

        predictions.append(
            {
                "image_source": image_id,
                "question": question,
                "predict": normalized,
                # "raw_output": output_text,
            }
        )

        if normalized == answer:
            num_correct += 1

    accuracy = num_correct / len(annotations) if annotations else 0.0
    print(f"Accuracy: {accuracy:.4f} ({num_correct}/{len(annotations)})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
