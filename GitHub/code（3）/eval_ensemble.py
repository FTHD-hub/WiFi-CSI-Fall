import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from custom_models import build_model, extract_state_dict, normalize_model_variant
from eval_thresholds import build_eval_dataset, collect_probabilities, metric_row
from train import DeviceDataLoader, get_default_data_root, get_default_device


def parse_model_spec(spec):
    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError(
            "Each --model must use: variant|checkpoint|weight"
        )
    variant, checkpoint, weight = parts
    return normalize_model_variant(variant), Path(checkpoint), float(weight)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate weighted model ensemble.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec: variant|checkpoint|weight. Repeat for each model.",
    )
    parser.add_argument("--data-root", type=Path, default=get_default_data_root())
    parser.add_argument("--scenario", type=str, default="scenario3")
    parser.add_argument("--envs", type=str, default=None)
    parser.add_argument("--train-envs", type=str, default=None)
    parser.add_argument("--test-envs", type=str, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--start", type=float, default=0.001)
    parser.add_argument("--end", type=float, default=0.900)
    parser.add_argument("--step", type=float, default=0.001)
    return parser.parse_args()


def main():
    args = parse_args()
    specs = [parse_model_spec(spec) for spec in args.model]
    total_weight = sum(weight for _, _, weight in specs)
    if total_weight <= 0:
        raise ValueError("Total ensemble weight must be positive.")

    dataset = build_eval_dataset(args)
    device = get_default_device()
    loader = DeviceDataLoader(
        DataLoader(dataset, batch_size=args.batch_size, shuffle=False),
        device,
    )

    ensemble_probability = None
    labels = None
    for variant, checkpoint_path, weight in specs:
        model = build_model(variant, pretrained_backbone=True).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=False)
        probabilities, labels = collect_probabilities(model, loader)
        contribution = probabilities * (weight / total_weight)
        ensemble_probability = (
            contribution
            if ensemble_probability is None
            else ensemble_probability + contribution
        )

    thresholds = []
    value = args.start
    while value <= args.end + 1e-12:
        thresholds.append(round(value, 6))
        value += args.step

    rows = [
        metric_row(ensemble_probability, labels, threshold)
        for threshold in thresholds
    ]
    best_acc = max(rows, key=lambda row: (row["acc"], row["f1"], row["rec"]))
    best_f1 = max(rows, key=lambda row: (row["f1"], row["acc"], row["rec"]))

    print("Best by Acc")
    print(
        f"threshold={best_acc['threshold']:.3f}, "
        f"Acc={best_acc['acc'] * 100:.2f}%, "
        f"Prec={best_acc['prec'] * 100:.2f}%, "
        f"Rec={best_acc['rec'] * 100:.2f}%, "
        f"F1={best_acc['f1'] * 100:.2f}%, "
        f"TP={best_acc['tp']}, TN={best_acc['tn']}, "
        f"FP={best_acc['fp']}, FN={best_acc['fn']}"
    )
    print("Best by F1")
    print(
        f"threshold={best_f1['threshold']:.3f}, "
        f"Acc={best_f1['acc'] * 100:.2f}%, "
        f"Prec={best_f1['prec'] * 100:.2f}%, "
        f"Rec={best_f1['rec'] * 100:.2f}%, "
        f"F1={best_f1['f1'] * 100:.2f}%, "
        f"TP={best_f1['tp']}, TN={best_f1['tn']}, "
        f"FP={best_f1['fp']}, FN={best_f1['fn']}"
    )


if __name__ == "__main__":
    main()
