import argparse
from pathlib import Path

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from train import to_device
from train import get_default_device, DeviceDataLoader
from train import evaluate
from train import get_default_data_root, get_default_results_dir, get_default_weights_dir
from train import print_dataset_paths, resolve_dataset_paths
from train import load_compact_dataset, print_loaded_dataset_stats
from train import print_available_environments, print_environment_selection
from train import print_split_summary
from train import print_available_scenarios, SCENARIO_PRESETS, normalize_scenario_name
from train import build_aggregated_split_datasets
from train import write_json
from custom_models import (
    build_model,
    extract_state_dict,
    infer_model_variant_from_state_dict,
    normalize_model_variant,
    resolve_ablation_variant,
)
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms as T


def normalize_compact_data_with_reference(reference_data, target_data):
    reference_tensor = torch.from_numpy(reference_data).clone()
    target_tensor = torch.from_numpy(target_data).clone()
    transform = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for idx in range(reference_tensor.shape[0]):
        reference_tensor[idx] = transform(reference_tensor[idx])
    for idx in range(target_tensor.shape[0]):
        target_tensor[idx] = transform(target_tensor[idx])

    reference_max = torch.max(reference_tensor)
    if reference_max.item() != 0:
        reference_tensor = reference_tensor / reference_max
        target_tensor = target_tensor / reference_max

    reference_mean = torch.mean(reference_tensor)
    reference_tensor = reference_tensor - reference_mean
    target_tensor = target_tensor - reference_mean
    return reference_tensor, target_tensor


def collect_logits(model, data_loader):
    model.eval()
    logits_parts = []
    label_parts = []
    with torch.inference_mode():
        for images, labels in data_loader:
            logits_parts.append(model(images).detach().cpu())
            label_parts.append(labels.squeeze().detach().cpu().long())
    return torch.cat(logits_parts), torch.cat(label_parts)


def threshold_metrics(logits, labels, threshold):
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    predictions = (probabilities >= threshold).long()
    labels = labels.long()

    tp = int(torch.sum((predictions == 1) & (labels == 1)).item())
    tn = int(torch.sum((predictions == 0) & (labels == 0)).item())
    fp = int(torch.sum((predictions == 1) & (labels == 0)).item())
    fn = int(torch.sum((predictions == 0) & (labels == 1)).item())
    count = int(labels.numel())

    accuracy = (tp + tn) / count if count else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        'threshold': float(threshold),
        'val_loss': float(F.cross_entropy(logits, labels).item()),
        'val_acc': accuracy,
        'val_prec': precision,
        'val_rec': recall,
        'val_f1': f1,
        'val_tp': tp,
        'val_tn': tn,
        'val_fp': fp,
        'val_fn': fn,
        'val_count': count,
    }


def sweep_thresholds(logits, labels, objective):
    candidates = []
    for step in range(20, 81):
        threshold = step / 100.0
        candidates.append(threshold_metrics(logits, labels, threshold))
    return max(candidates, key=lambda item: item[f'val_{objective}']), candidates


def resolve_evaluation_entries(data_root, scenario, envs):
    if scenario and envs:
        raise ValueError('--scenario cannot be combined with --envs.')

    if scenario:
        scenario_key = normalize_scenario_name(scenario)
        preset = SCENARIO_PRESETS[scenario_key]
        if preset['mode'] in {'aggregated', 'aggregated_paper'}:
            return resolve_dataset_paths(data_root, ','.join(preset['envs'])), preset
        return resolve_dataset_paths(data_root, ','.join(preset['test_envs'])), preset

    return resolve_dataset_paths(data_root, envs), None


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the ENetFall baseline model.')
    parser.add_argument(
        '--data-root',
        type=Path,
        default=get_default_data_root(),
        help='Directory that contains the .mat dataset files.',
    )
    parser.add_argument(
        '--scenario',
        type=str,
        help='Preset scenario name, for example scenario1, scenario2, scenario2-paper, scenario2-heldout, scenario3, or scenario4.',
    )
    parser.add_argument(
        '--envs',
        type=str,
        help='Comma-separated environments to evaluate, for example A,D or C_NLoS.',
    )
    parser.add_argument(
        '--weights-path',
        type=Path,
        default=get_default_weights_dir() / 'B0(modified)_trained_with_all_data.pth',
        help='Path to the model checkpoint file.',
    )
    parser.add_argument(
        '--ablation',
        type=str,
        choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
        default=None,
        help='Ablation preset: A baseline, B B0+DWT, C soft threshold, '
             'D motion gate, E training augmentation, F DWT+Transformer, '
             'G Transformer+motion, H Transformer+motion+augmentation.',
    )
    parser.add_argument(
        '--model-variant',
        type=str,
        default=None,
        help='Model family to load. Use --ablation A-H or omit it to use checkpoint metadata.',
    )
    parser.add_argument(
        '--check-data-only',
        action='store_true',
        help='Resolve and print dataset paths, then exit without loading weights.',
    )
    parser.add_argument(
        '--split-summary-only',
        action='store_true',
        help='Print selected-environment sample summaries, then exit.',
    )
    parser.add_argument(
        '--load-data-only',
        action='store_true',
        help='Load the compact dataset, print shape/memory stats, then exit without loading weights.',
    )
    parser.add_argument(
        '--split-seed',
        type=int,
        default=42,
        help='Random seed used for deterministic 80/20 dataset splits in aggregated scenarios.',
    )
    parser.add_argument(
        '--metrics-path',
        type=Path,
        default=get_default_results_dir() / 'test_metrics.json',
        help='Path to save evaluation metrics as JSON.',
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=2,
        help='Number of output classes for the classifier head.',
    )
    parser.add_argument(
        '--fpnet-base-channels',
        type=int,
        default=48,
        help='Default base width for the wavelet-fpnet variant if the checkpoint does not store it.',
    )
    parser.add_argument(
        '--fpnet-pyramid-channels',
        type=int,
        default=160,
        help='Default pyramid width for the wavelet-fpnet variant if the checkpoint does not store it.',
    )
    parser.add_argument(
        '--decision-threshold',
        type=float,
        default=0.5,
        help='Probability threshold for predicting Fall. Default is 0.50.',
    )
    parser.add_argument(
        '--threshold-sweep',
        action='store_true',
        help='Scan thresholds from 0.20 to 0.80 and select the best one.',
    )
    parser.add_argument(
        '--threshold-objective',
        choices=['acc', 'prec', 'rec', 'f1'],
        default='f1',
        help='Metric used to select the threshold during --threshold-sweep.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.num_classes < 2:
        raise ValueError('--num-classes must be at least 2')
    if not 0.0 < args.decision_threshold < 1.0:
        raise ValueError('--decision-threshold must be between 0 and 1')
    device = get_default_device()
    num_classes = args.num_classes
    dataset_entries, scenario_preset = resolve_evaluation_entries(args.data_root, args.scenario, args.envs)
    print_available_environments()
    print_available_scenarios()
    if scenario_preset is not None:
        print(f"Using {scenario_preset['name']}: {scenario_preset['description']}")
    print_environment_selection(dataset_entries, 'Evaluation environments')
    print_dataset_paths(dataset_entries)

    if args.check_data_only:
        print('Data path check completed. Evaluation was not started.')
        raise SystemExit(0)
    if args.split_summary_only:
        print_split_summary(selected_entries=dataset_entries)
        print('Split summary check completed. Evaluation was not started.')
        raise SystemExit(0)

    split_info = None
    if scenario_preset is not None and scenario_preset['mode'] in {'aggregated', 'aggregated_paper'}:
        from train import load_compact_dataset_with_env_ids

        compact_data, compact_labels, compact_env_ids = load_compact_dataset_with_env_ids(dataset_entries)
        if args.load_data_only:
            print_loaded_dataset_stats(compact_data, compact_labels)
            print('Compact data load check completed. Evaluation was not started.')
            raise SystemExit(0)

        _, eval_dataset, split_info = build_aggregated_split_datasets(
            compact_data,
            compact_labels,
            compact_env_ids,
            args.split_seed,
            eval_envs=scenario_preset.get('eval_envs'),
        )
        print(f"Held-out evaluation subset size: {split_info['eval_count']}")
        test_data_loader = DataLoader(dataset=eval_dataset, batch_size=50, shuffle=False, drop_last=False)
    else:
        compact_data, compact_labels = load_compact_dataset(dataset_entries)
        if args.load_data_only:
            print_loaded_dataset_stats(compact_data, compact_labels)
            print('Compact data load check completed. Evaluation was not started.')
            raise SystemExit(0)

        if scenario_preset is not None and scenario_preset['mode'] == 'explicit':
            train_entries = resolve_dataset_paths(
                args.data_root,
                ','.join(scenario_preset['train_envs']),
            )
            train_compact_data, _ = load_compact_dataset(train_entries)
            _, data_reshape = normalize_compact_data_with_reference(
                train_compact_data,
                compact_data,
            )
            print('Evaluation normalization: reused training-split statistics for consistency.')
        else:
            _, data_reshape = normalize_compact_data_with_reference(
                compact_data,
                compact_data,
            )
            print('Evaluation normalization: used evaluation-set statistics.')

        labels_all = torch.from_numpy(compact_labels).type(torch.LongTensor)
        dataset = TensorDataset(data_reshape, labels_all)
        test_data_loader = DataLoader(dataset=dataset, batch_size=50, shuffle=False, drop_last=False)

    test_data_loader = DeviceDataLoader(test_data_loader, device)
    checkpoint = torch.load(args.weights_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    checkpoint_variant = None
    checkpoint_config = {}
    if isinstance(checkpoint, dict):
        checkpoint_variant = checkpoint.get('model_variant')
        checkpoint_config = checkpoint.get('model_config', {})
        if not isinstance(checkpoint_config, dict):
            checkpoint_config = {}

    if args.ablation is not None and args.model_variant is not None:
        raise ValueError('--ablation cannot be combined with --model-variant.')
    requested_variant = (
        resolve_ablation_variant(args.ablation)
        if args.ablation is not None
        else args.model_variant
    )
    inferred_variant = normalize_model_variant(
        infer_model_variant_from_state_dict(state_dict)
    )
    resolved_variant = (
        normalize_model_variant(requested_variant)
        if requested_variant is not None
        else (
            inferred_variant
            if checkpoint_variant is None
            or normalize_model_variant(checkpoint_variant) != inferred_variant
            else normalize_model_variant(checkpoint_variant)
        )
    )

    model_kwargs = {}
    if resolved_variant == 'wavelet-fpnet':
        model_kwargs = {
            'base_channels': int(checkpoint_config.get('base_channels', args.fpnet_base_channels)),
            'pyramid_channels': int(checkpoint_config.get('pyramid_channels', args.fpnet_pyramid_channels)),
            'wavelet_channels': int(checkpoint_config.get('wavelet_channels', 3)),
        }

    testmodel = build_model(
        resolved_variant,
        num_classes=num_classes,
        pretrained_backbone=False,
        **model_kwargs,
    )
    testmodel.load_state_dict(state_dict)
    to_device(testmodel, device)
    testmodel.eval()
    logits, labels = collect_logits(testmodel, test_data_loader)
    if args.threshold_sweep:
        result, threshold_candidates = sweep_thresholds(
            logits,
            labels,
            args.threshold_objective,
        )
        print(
            f"Threshold sweep selected {result['threshold']:.2f} "
            f"by val_{args.threshold_objective}."
        )
        print(
            'Warning: this threshold was selected on the current evaluation split. '
            'For strict paper reporting, tune on validation data and apply it to test data.'
        )
    else:
        result = threshold_metrics(logits, labels, args.decision_threshold)
        threshold_candidates = None
    print(result)
    payload = {
        'scenario': scenario_preset['name'] if scenario_preset else None,
        'scenario_description': scenario_preset['description'] if scenario_preset else None,
        'model_variant': resolved_variant,
        'eval_envs': [entry['env_name'] for entry in dataset_entries],
        'split_seed': args.split_seed,
        'split_info': split_info,
        'weights_path': str(args.weights_path),
        'decision_threshold': result['threshold'],
        'threshold_objective': args.threshold_objective if args.threshold_sweep else None,
        'threshold_candidates': threshold_candidates,
        'metrics': result,
    }
    write_json(args.metrics_path, payload)
    print(f"Metrics written to {args.metrics_path}")
