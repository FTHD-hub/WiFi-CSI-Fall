import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T

from custom_models import build_model, extract_state_dict
from train import (
    DeviceDataLoader,
    build_aggregated_split_datasets,
    build_dataset_entries,
    get_default_data_root,
    get_default_device,
    load_compact_dataset,
    load_compact_dataset_with_env_ids,
    normalize_model_variant,
)


def build_eval_dataset(args):
    train_entries, test_entries, selected_entries, scenario_preset = build_dataset_entries(args)

    if selected_entries is not None:
        compact_data, compact_labels, compact_env_ids = load_compact_dataset_with_env_ids(selected_entries)
        eval_envs = scenario_preset.get("eval_envs") if scenario_preset is not None else None
        _, test_dataset, _ = build_aggregated_split_datasets(
            compact_data,
            compact_labels,
            compact_env_ids,
            args.split_seed,
            eval_envs=eval_envs,
        )
        return test_dataset

    train_data_np, _ = load_compact_dataset(train_entries)
    test_data_np, test_labels_np = load_compact_dataset(test_entries)

    train_data = torch.from_numpy(train_data_np)
    test_data = torch.from_numpy(test_data_np)
    transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for idx in range(train_data.shape[0]):
        train_data[idx] = transform(train_data[idx])
    for idx in range(test_data.shape[0]):
        test_data[idx] = transform(test_data[idx])

    train_max = torch.max(train_data)
    train_data = train_data / train_max
    test_data = test_data / train_max
    train_mean = torch.mean(train_data)
    test_data = test_data - train_mean
    test_labels = torch.from_numpy(test_labels_np).type(torch.LongTensor)
    return TensorDataset(test_data, test_labels)


def collect_probabilities(model, data_loader):
    probabilities = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in data_loader:
            outputs = model(images)
            probabilities.append(F.softmax(outputs, dim=1)[:, 1].detach().cpu())
            labels.append(batch_labels.squeeze().detach().cpu())
    return torch.cat(probabilities), torch.cat(labels)


def metric_row(probabilities, labels, threshold):
    preds = (probabilities >= threshold).long()
    tp = int(torch.sum((preds == 1) & (labels == 1)).item())
    tn = int(torch.sum((preds == 0) & (labels == 0)).item())
    fp = int(torch.sum((preds == 1) & (labels == 0)).item())
    fn = int(torch.sum((preds == 0) & (labels == 1)).item())
    total = max(tp + tn + fp + fn, 1)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "threshold": threshold,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint across decision thresholds.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-variant", type=str, required=True)
    parser.add_argument("--data-root", type=Path, default=get_default_data_root())
    parser.add_argument("--scenario", type=str, default="scenario3")
    parser.add_argument("--envs", type=str, default=None)
    parser.add_argument("--train-envs", type=str, default=None)
    parser.add_argument("--test-envs", type=str, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--end", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.model_variant = normalize_model_variant(args.model_variant)
    dataset = build_eval_dataset(args)
    device = get_default_device()
    loader = DeviceDataLoader(DataLoader(dataset, batch_size=args.batch_size, shuffle=False), device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    model = build_model(
        args.model_variant,
        pretrained_backbone=not args.no_pretrained,
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys ignored: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys ignored: {len(unexpected)}")

    probabilities, labels = collect_probabilities(model, loader)
    thresholds = []
    value = args.start
    while value <= args.end + 1e-9:
        thresholds.append(round(value, 4))
        value += args.step

    rows = [metric_row(probabilities, labels, threshold) for threshold in thresholds]
    best_acc = max(rows, key=lambda row: (row["acc"], row["f1"]))
    best_f1 = max(rows, key=lambda row: (row["f1"], row["acc"]))

    print("Best by Acc")
    print(
        f"threshold={best_acc['threshold']:.2f}, "
        f"Acc={best_acc['acc'] * 100:.2f}%, "
        f"Prec={best_acc['prec'] * 100:.2f}%, "
        f"Rec={best_acc['rec'] * 100:.2f}%, "
        f"F1={best_acc['f1'] * 100:.2f}%"
    )
    print("Best by F1")
    print(
        f"threshold={best_f1['threshold']:.2f}, "
        f"Acc={best_f1['acc'] * 100:.2f}%, "
        f"Prec={best_f1['prec'] * 100:.2f}%, "
        f"Rec={best_f1['rec'] * 100:.2f}%, "
        f"F1={best_f1['f1'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
