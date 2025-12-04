import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate POPE yes/no predictions")
    parser.add_argument("--gt_files", type=str, required=True, help="Ground truth json file (list of dicts).")
    parser.add_argument("--gen_files", type=str, required=True, help="Prediction json file (list of dicts).")
    parser.add_argument("--verbose", action="store_true", help="Print mismatched entries.")
    return parser.parse_args()


def load_json_lines(path: str) -> Iterable[dict]:
    file_path = Path(path)
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            if isinstance(data, dict):
                raise ValueError(f"{path} contains a JSON object but a list was expected.")
            return data
        return [json.loads(line) for line in f if line.strip()]


def extract_answer(entry: dict) -> Tuple[str, str]:
    image = entry.get("image_source") or entry.get("image") or entry.get("image_id") or entry.get("id")
    question = entry.get("question") or entry.get("text") or entry.get("prompt")
    if image is None or question is None:
        raise KeyError("Missing keys to identify a QA pair.")
    return str(image), str(question)


def get_label(entry: dict) -> str:
    for key in ("answer", "label", "predict", "text"):
        if key in entry:
            return str(entry[key]).strip().lower()
    raise KeyError("No label/prediction field found.")


def main() -> None:
    args = parse_args()

    gt_entries = list(load_json_lines(args.gt_files))
    pred_entries = list(load_json_lines(args.gen_files))

    gt_map: Dict[Tuple[str, str, int], str] = {}
    gt_counter: Dict[Tuple[str, str], int] = {}
    for item in gt_entries:
        key = extract_answer(item)
        occ = gt_counter.get(key, 0)
        gt_counter[key] = occ + 1
        gt_map[(key[0], key[1], occ)] = get_label(item)

    pred_map: Dict[Tuple[str, str, int], str] = {}
    pred_counter: Dict[Tuple[str, str], int] = {}
    for item in pred_entries:
        key = extract_answer(item)
        occ = pred_counter.get(key, 0)
        pred_counter[key] = occ + 1
        idx_key = (key[0], key[1], occ)
        if idx_key in pred_map:
            raise ValueError(f"Duplicate prediction for {idx_key}")
        pred_map[idx_key] = get_label(item)

    assert set(gt_map.keys()) == set(pred_map.keys()), "Prediction and GT sets differ."

    tp = tn = fp = fn = 0
    yes_predictions = 0
    unknown = 0

    for key, gt_answer in gt_map.items():
        pred_answer = pred_map[key]

        if gt_answer not in {"yes", "no"}:
            unknown += 1
            if args.verbose:
                print(f"Skipping unknown GT label {gt_answer} for {key}")
            continue

        pred_yes = "yes" in pred_answer
        pred_no = "no" in pred_answer

        if pred_yes and not pred_no:
            pred_label = "yes"
        elif pred_no and not pred_yes:
            pred_label = "no"
        elif pred_answer.startswith("yes"):
            pred_label = "yes"
        elif pred_answer.startswith("no"):
            pred_label = "no"
        else:
            pred_label = pred_answer

        if gt_answer == "yes":
            if pred_label == "yes":
                tp += 1
                yes_predictions += 1
            else:
                fn += 1
                if args.verbose:
                    print(f"FN {key}: pred={pred_answer}")
        else:
            if pred_label == "no":
                tn += 1
            else:
                fp += 1
                if pred_label == "yes":
                    yes_predictions += 1
                if args.verbose:
                    print(f"FP {key}: pred={pred_answer}")

    total = len(gt_map)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    yes_ratio = yes_predictions / total if total else 0.0
    unknown_ratio = unknown / total if total else 0.0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Yes proportion: {yes_ratio:.4f}")
    print(f"Unknown proportion: {unknown_ratio:.4f}")


if __name__ == "__main__":
    main()
