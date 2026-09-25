import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys

import torch
import math
import numpy as np
import torch.nn as nn
import scipy.io as sio
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms as T

from custom_models import (
    build_checkpoint_payload,
    build_model,
    extract_state_dict,
    normalize_model_variant,
    resolve_ablation_variant,
)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


ENVIRONMENT_SPECS = {
    'A': {
        'display_name': 'Living Room A',
        'filename': 'dataset_living_room.mat',
    },
    'B': {
        'display_name': 'Meeting Room B',
        'filename': 'dataset_meeting_room.mat',
    },
    'C_LoS': {
        'display_name': 'Home Lab C LoS',
        'filename': 'dataset_home_lab(L).mat',
    },
    'C_NLoS': {
        'display_name': 'Home Lab C NLoS',
        'filename': 'dataset_home_lab(R).mat',
    },
    'D': {
        'display_name': 'Lecture Room D',
        'filename': 'dataset_lecture_room.mat',
    },
}

ENVIRONMENT_ALIASES = {
    'a': 'A',
    'room_a': 'A',
    'living_room_a': 'A',
    'livingrooma': 'A',
    'b': 'B',
    'room_b': 'B',
    'meeting_room_b': 'B',
    'meetingroomb': 'B',
    'c_los': 'C_LoS',
    'c-los': 'C_LoS',
    'clos': 'C_LoS',
    'room_c_los': 'C_LoS',
    'home_lab_l': 'C_LoS',
    'home_lab_left': 'C_LoS',
    'left': 'C_LoS',
    'c_nlos': 'C_NLoS',
    'c-nlos': 'C_NLoS',
    'cnlos': 'C_NLoS',
    'room_c_nlos': 'C_NLoS',
    'home_lab_r': 'C_NLoS',
    'home_lab_right': 'C_NLoS',
    'right': 'C_NLoS',
    'd': 'D',
    'room_d': 'D',
    'lecture_room_d': 'D',
    'lectureroomd': 'D',
}

SCENARIO_PRESETS = {
    '1': {
        'name': 'Scenario 1',
        'description': 'Aggregated dataset with random 80/20 split over all environments.',
        'mode': 'aggregated',
        'envs': ['A', 'B', 'C_LoS', 'C_NLoS', 'D'],
    },
    '2': {
        'name': 'Scenario 2',
        'description': 'Train on the aggregated dataset and evaluate on the NLoS subset C_NLoS.',
        'mode': 'explicit',
        'train_envs': ['A', 'B', 'C_LoS', 'C_NLoS', 'D'],
        'test_envs': ['C_NLoS'],
        'allow_overlap': True,
    },
    '2paper': {
        'name': 'Scenario 2 Paper',
        'description': 'Paper-facing room-wise evaluation: report the full C_NLoS subset (85 samples) for direct comparison.',
        'mode': 'explicit',
        'train_envs': ['A', 'B', 'C_LoS', 'C_NLoS', 'D'],
        'test_envs': ['C_NLoS'],
        'allow_overlap': True,
    },
    '2heldout': {
        'name': 'Scenario 2 Held-Out',
        'description': 'Strict aggregated 80/20 split over all environments, then report only the held-out C_NLoS subset.',
        'mode': 'aggregated_paper',
        'envs': ['A', 'B', 'C_LoS', 'C_NLoS', 'D'],
        'eval_envs': ['C_NLoS'],
    },
    '3': {
        'name': 'Scenario 3',
        'description': 'Train on B and C, then test on unseen A and D.',
        'mode': 'explicit',
        'train_envs': ['B', 'C_LoS', 'C_NLoS'],
        'test_envs': ['A', 'D'],
    },
    '4': {
        'name': 'Scenario 4',
        'description': 'Train on B and D, then test on C_LoS and C_NLoS.',
        'mode': 'explicit',
        'train_envs': ['B', 'D'],
        'test_envs': ['C_LoS', 'C_NLoS'],
    },
}

SCENARIO_REFERENCE_METRICS = {
    '1': {
        'label': 'Modified B0 [34]',
        'acc': 0.949,
        'prec': 0.917,
        'rec': 0.968,
    },
    '2': {
        'label': 'Modified B0 [34]',
        'acc': 0.820,
        'prec': 0.727,
        'rec': 1.000,
    },
    '2paper': {
        'label': 'Modified B0 [34]',
        'acc': 0.820,
        'prec': 0.727,
        'rec': 1.000,
    },
    '2heldout': {
        'label': 'Modified B0 [34]',
        'acc': 0.820,
        'prec': 0.727,
        'rec': 1.000,
    },
}


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to the device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


class FocalLoss(nn.Module):
    def __init__(self, class_weights=(1.0, 3.0), gamma=2.0, reduction='mean'):
        super().__init__()
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer(
            'class_weights',
            torch.tensor(class_weights, dtype=torch.float32),
        )

    def forward(self, logits, targets):
        cross_entropy = F.cross_entropy(logits, targets, reduction='none')
        probability = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        sample_weights = self.class_weights[targets]
        loss = sample_weights * (1.0 - probability).pow(self.gamma) * cross_entropy
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class CSIAugmentedDataset(Dataset):
    def __init__(
        self,
        dataset,
        *,
        probability=0.5,
        noise_std=0.01,
        shift_max=20,
        amplitude_scale=0.10,
    ):
        self.dataset = dataset
        self.probability = probability
        self.noise_std = noise_std
        self.shift_max = shift_max
        self.amplitude_scale = amplitude_scale

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample, label = self.dataset[index]
        sample = sample.clone()

        if torch.rand(()) >= self.probability:
            return sample, label

        if self.noise_std > 0:
            sample = sample + torch.randn_like(sample) * self.noise_std

        if self.amplitude_scale > 0:
            scale = torch.empty(sample.shape[0], 1, 1).uniform_(
                1.0 - self.amplitude_scale,
                1.0 + self.amplitude_scale,
            )
            sample = sample * scale

        if self.shift_max > 0:
            shift = int(torch.randint(-self.shift_max, self.shift_max + 1, ()).item())
            sample = torch.roll(sample, shifts=shift, dims=1)

        if torch.rand(()) < 0.5:
            original_length = sample.shape[1]
            reduced_length = max(original_length // 2, 1)
            temporal = sample.permute(0, 2, 1).unsqueeze(0)
            temporal = F.interpolate(
                temporal,
                size=(sample.shape[2], reduced_length),
                mode='bilinear',
                align_corners=False,
            )
            temporal = F.interpolate(
                temporal,
                size=(sample.shape[2], original_length),
                mode='bilinear',
                align_corners=False,
            )
            sample = temporal.squeeze(0).permute(0, 2, 1)

        return sample, label


def get_default_device():
    """"Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensors to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def format_pct(value):
    return f"{value * 100:.2f}%"


def format_pp(current, reference):
    return f"{(current - reference) * 100:+.2f}pp"


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def parse_float_list(raw_value, expected_count=None):
    parts = [part.strip() for part in str(raw_value).split(',') if part.strip()]
    values = [float(part) for part in parts]
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(f'Expected {expected_count} comma-separated values, got {len(values)}')
    return values


def resolve_loss_type(args):
    if args.loss_type is not None:
        return args.loss_type
    if args.model_variant in {'wavelet-fpnet', 'differential-tcn'}:
        return 'focal'
    return 'cross_entropy'


def resolve_scheduler_type(args):
    if args.scheduler is not None:
        return args.scheduler
    if args.model_variant in {'wavelet-fpnet', 'differential-tcn'}:
        return 'cosine'
    return 'step'


def resolve_learning_rate(args):
    if args.learning_rate is not None:
        return args.learning_rate
    if args.model_variant == 'differential-tcn':
        return 1e-3
    if args.model_variant == 'wavelet-fpnet':
        return 5e-4
    return 1e-3


def create_loss_fn(args, device):
    if args.loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss().to(device)
    if args.loss_type == 'focal':
        class_weights = parse_float_list(args.focal_class_weights, expected_count=2)
        return FocalLoss(
            class_weights=tuple(class_weights),
            gamma=args.focal_gamma,
        ).to(device)
    raise ValueError(f'Unsupported loss type: {args.loss_type}')


def create_scheduler(optimizer, args):
    if args.scheduler == 'none':
        return None
    if args.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.min_learning_rate,
        )
    raise ValueError(f'Unsupported scheduler: {args.scheduler}')


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def predict_fall_labels(outputs, decision_threshold=0.5):
    fall_probability = torch.softmax(outputs, dim=1)[:, 1]
    return (fall_probability >= decision_threshold).long()


def precision(outputs, labels):
    preds = predict_fall_labels(outputs)
    num_tp = 0
    num_fp = 0
    for idx in range(len(labels)):
        if preds[idx].item() == labels[idx].item() and preds[idx].item() == 1:
            num_tp = num_tp + 1
        if preds[idx].item() != labels[idx].item() and preds[idx].item() == 1:
            num_fp = num_fp + 1
    denominator = num_tp + num_fp
    if denominator == 0:
        return torch.tensor(0.0)
    return torch.tensor(num_tp / denominator)


def recall(outputs, labels):
    preds = predict_fall_labels(outputs)
    num_tp = 0
    num_fn = 0
    for idx in range(len(labels)):
        if preds[idx].item() == labels[idx].item() and preds[idx].item() == 1:
            num_tp = num_tp + 1
        if preds[idx].item() != labels[idx].item() and preds[idx].item() == 0:
            num_fn = num_fn + 1
    denominator = num_tp + num_fn
    if denominator == 0:
        return torch.tensor(0.0)
    return torch.tensor(num_tp / denominator)


def validation_step(model, batch, criterion=None, decision_threshold=0.5):
    images, labels = batch
    out = model(images)
    labels = labels.squeeze()
    loss = F.cross_entropy(out, labels) if criterion is None else criterion(out, labels)
    preds = predict_fall_labels(out, decision_threshold=decision_threshold)
    tp = int(torch.sum((preds == 1) & (labels == 1)).item())
    tn = int(torch.sum((preds == 0) & (labels == 0)).item())
    fp = int(torch.sum((preds == 1) & (labels == 0)).item())
    fn = int(torch.sum((preds == 0) & (labels == 1)).item())
    batch_size = int(labels.numel())
    correct = int(torch.sum(preds == labels).item())
    return {
        'val_loss_sum': loss.detach().item() * batch_size,
        'val_correct': correct,
        'val_count': batch_size,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def epoch_end(model, epoch, result):
    print(
        "Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, "
        "val_acc: {:.4f}, val_prec: {:.4f}, val_rec: {:.4f}, val_f1: {:.4f}".format(
            epoch,
            result['train_loss'],
            result['train_acc'],
            result['val_loss'],
            result['val_acc'],
            result['val_prec'],
            result['val_rec'],
            result['val_f1'],
        )
    )


def validation_epoch_end(model, outputs):
    total_loss = sum(x['val_loss_sum'] for x in outputs)
    total_correct = sum(x['val_correct'] for x in outputs)
    total_count = sum(x['val_count'] for x in outputs)
    tp = sum(x['tp'] for x in outputs)
    tn = sum(x['tn'] for x in outputs)
    fp = sum(x['fp'] for x in outputs)
    fn = sum(x['fn'] for x in outputs)

    val_loss = total_loss / total_count if total_count else 0.0
    val_acc = total_correct / total_count if total_count else 0.0
    val_prec = tp / (tp + fp) if (tp + fp) else 0.0
    val_rec = tp / (tp + fn) if (tp + fn) else 0.0
    val_f1 = (2 * val_prec * val_rec / (val_prec + val_rec)) if (val_prec + val_rec) else 0.0

    return {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_prec': val_prec,
        'val_rec': val_rec,
        'val_f1': val_f1,
        'val_tp': tp,
        'val_tn': tn,
        'val_fp': fp,
        'val_fn': fn,
        'val_count': total_count,
    }


def evaluate(model, val_loader, criterion=None, decision_threshold=0.5):
    model.eval()
    outputs = [
        validation_step(
            model,
            batch,
            criterion,
            decision_threshold=decision_threshold,
        )
        for batch in val_loader
    ]
    return validation_epoch_end(model, outputs)


def training_step(model, batch, criterion):
    images, labels = batch
    out = model(images)
    labels = labels.squeeze()
    loss = criterion(out, labels)
    return loss


def get_project_root():
    return Path(__file__).resolve().parents[1]


def get_default_data_root():
    return get_project_root() / 'data'


def get_default_weights_dir():
    return get_project_root() / 'weights'


def get_default_results_dir():
    return get_project_root() / 'results'


def get_default_environment_names():
    return list(ENVIRONMENT_SPECS.keys())


def normalize_scenario_name(name):
    normalized = name.strip().lower().replace(' ', '').replace('_', '').replace('-', '')
    if normalized.startswith('scenario'):
        normalized = normalized[len('scenario'):]
    if normalized in SCENARIO_PRESETS:
        return normalized
    valid = ', '.join(f"scenario{key}" for key in SCENARIO_PRESETS)
    raise ValueError(f'Unknown scenario "{name}". Valid values: {valid}')


def normalize_environment_name(name):
    normalized = name.strip()
    if not normalized:
        raise ValueError('Environment name cannot be empty.')

    canonical_key = normalized.upper()
    if canonical_key in ENVIRONMENT_SPECS:
        return canonical_key

    alias_key = (
        normalized.lower()
        .replace(' ', '_')
        .replace('/', '_')
    )
    if alias_key in ENVIRONMENT_ALIASES:
        return ENVIRONMENT_ALIASES[alias_key]

    valid = ', '.join(get_default_environment_names())
    raise ValueError(f'Unknown environment "{name}". Valid values: {valid}')


def parse_environment_list(raw_value, default_names=None):
    if raw_value is None:
        return list(default_names or get_default_environment_names())

    names = []
    seen = set()
    for item in raw_value.split(','):
        canonical = normalize_environment_name(item)
        if canonical not in seen:
            names.append(canonical)
            seen.add(canonical)

    if not names:
        raise ValueError('At least one environment must be selected.')
    return names


def resolve_dataset_paths(data_root, environment_names=None):
    root = Path(data_root).resolve()
    selected_names = parse_environment_list(environment_names, get_default_environment_names())
    entries = []
    missing = []

    for env_name in selected_names:
        spec = ENVIRONMENT_SPECS[env_name]
        path = root / spec['filename']
        if not path.exists():
            missing.append(path)
            continue
        entries.append({
            'env_name': env_name,
            'display_name': spec['display_name'],
            'path': path,
        })

    if missing:
        missing_str = '\n'.join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing dataset files under data root: {root}\n{missing_str}"
        )
    return entries


def print_dataset_paths(dataset_entries, header='Resolved dataset files:'):
    print(header)
    for entry in dataset_entries:
        print(f"  {entry['env_name']} ({entry['display_name']}): {entry['path']}")


def print_environment_selection(dataset_entries, title):
    names = ', '.join(entry['env_name'] for entry in dataset_entries)
    print(f'{title}: {names}')


def print_available_environments():
    print('Available environments:')
    for env_name, spec in ENVIRONMENT_SPECS.items():
        print(f"  {env_name}: {spec['display_name']} -> {spec['filename']}")


def print_available_scenarios():
    print('Available scenarios:')
    for key, preset in SCENARIO_PRESETS.items():
        display_key = key
        if key == '2paper':
            display_key = '2-paper'
        elif key == '2heldout':
            display_key = '2-heldout'
        print(f"  scenario{display_key}: {preset['description']}")


def summarize_labels(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    pairs = {int(label): int(count) for label, count in zip(unique_labels, counts)}
    return pairs


def format_label_distribution(distribution):
    return ', '.join(f'{label}={count}' for label, count in sorted(distribution.items()))


def summarize_environment_entries(dataset_entries):
    summaries = []
    for entry in dataset_entries:
        labels = sio.loadmat(
            entry['path'],
            variable_names=['dataset_labels'],
        )['dataset_labels'].reshape(-1).astype(np.int64, copy=False)
        summaries.append({
            'env_name': entry['env_name'],
            'display_name': entry['display_name'],
            'sample_count': len(labels),
            'label_distribution': summarize_labels(labels),
        })
    return summaries


def print_environment_summaries(dataset_entries, title):
    print(title)
    summaries = summarize_environment_entries(dataset_entries)
    total_samples = 0
    total_distribution = {}
    for summary in summaries:
        total_samples += summary['sample_count']
        for label, count in summary['label_distribution'].items():
            total_distribution[label] = total_distribution.get(label, 0) + count
        print(
            f"  {summary['env_name']} ({summary['display_name']}): "
            f"samples={summary['sample_count']}, "
            f"labels={format_label_distribution(summary['label_distribution'])}"
        )
    print(
        f"  total: samples={total_samples}, "
        f"labels={format_label_distribution(total_distribution)}"
    )


def print_split_summary(train_entries=None, test_entries=None, selected_entries=None):
    if selected_entries is not None:
        print_environment_summaries(selected_entries, 'Selected environments summary:')
        return

    if train_entries is not None:
        print_environment_summaries(train_entries, 'Train environments summary:')
    if test_entries is not None:
        print_environment_summaries(test_entries, 'Test environments summary:')


def validate_split_args(args):
    if args.scenario and (args.envs or args.train_envs or args.test_envs):
        raise ValueError('--scenario cannot be combined with --envs or --train-envs/--test-envs.')
    if bool(args.train_envs) != bool(args.test_envs):
        raise ValueError('--train-envs and --test-envs must be provided together.')
    if args.envs and (args.train_envs or args.test_envs):
        raise ValueError('--envs cannot be combined with --train-envs/--test-envs.')


def build_dataset_entries(args):
    validate_split_args(args)
    if args.scenario:
        scenario_key = normalize_scenario_name(args.scenario)
        preset = SCENARIO_PRESETS[scenario_key]
        if preset['mode'] in {'aggregated', 'aggregated_paper'}:
            selected_entries = resolve_dataset_paths(args.data_root, ','.join(preset['envs']))
            return None, None, selected_entries, preset

        train_entries = resolve_dataset_paths(args.data_root, ','.join(preset['train_envs']))
        test_entries = resolve_dataset_paths(args.data_root, ','.join(preset['test_envs']))
        allow_overlap = preset.get('allow_overlap', False)
        if not allow_overlap:
            train_names = {entry['env_name'] for entry in train_entries}
            test_names = {entry['env_name'] for entry in test_entries}
            overlap = sorted(train_names & test_names)
            if overlap:
                overlap_str = ', '.join(overlap)
                raise ValueError(
                    f'Train and test environments must be disjoint. Overlap: {overlap_str}'
                )
        return train_entries, test_entries, None, preset

    if args.train_envs and args.test_envs:
        train_entries = resolve_dataset_paths(args.data_root, args.train_envs)
        test_entries = resolve_dataset_paths(args.data_root, args.test_envs)
        train_names = {entry['env_name'] for entry in train_entries}
        test_names = {entry['env_name'] for entry in test_entries}
        overlap = sorted(train_names & test_names)
        if overlap:
            overlap_str = ', '.join(overlap)
            raise ValueError(
                f'Train and test environments must be disjoint. Overlap: {overlap_str}'
            )
        return train_entries, test_entries, None, None

    selected_entries = resolve_dataset_paths(args.data_root, args.envs)
    return None, None, selected_entries, None


def inspect_dataset_layout(dataset_entries):
    total_samples = 0
    metadata = []
    expected_shape = None

    for entry in dataset_entries:
        path = entry['path']
        variables = {name: shape for name, shape, _ in sio.whosmat(path)}
        if 'dataset_CSI_t' not in variables or 'dataset_labels' not in variables:
            raise KeyError(f'Missing dataset_CSI_t or dataset_labels in {path}')

        csi_shape = variables['dataset_CSI_t']
        label_shape = variables['dataset_labels']

        if len(csi_shape) != 3:
            raise ValueError(f'Unexpected CSI tensor shape in {path}: {csi_shape}')

        if expected_shape is None:
            expected_shape = csi_shape[1:]
        elif csi_shape[1:] != expected_shape:
            raise ValueError(
                f'Inconsistent CSI tensor shape in {path}: {csi_shape[1:]} != {expected_shape}'
            )

        sample_count = csi_shape[0]
        if label_shape[0] != sample_count:
            raise ValueError(
                f'Label count does not match sample count in {path}: {label_shape} vs {csi_shape}'
            )

        metadata.append({
            'env_name': entry['env_name'],
            'display_name': entry['display_name'],
            'path': path,
            'sample_count': sample_count,
        })
        total_samples += sample_count

    return metadata, total_samples, expected_shape


def load_compact_dataset_with_env_ids(dataset_entries):
    metadata, total_samples, csi_shape = inspect_dataset_layout(dataset_entries)
    time_steps, subcarriers = csi_shape
    if subcarriers % 3 != 0:
        raise ValueError(f'Expected subcarriers to be divisible by 3, got {subcarriers}')

    compact_data = np.empty((total_samples, 3, time_steps, subcarriers // 3), dtype=np.float32)
    compact_labels = np.empty(total_samples, dtype=np.int64)
    compact_env_ids = np.empty(total_samples, dtype=object)

    offset = 0
    for item in metadata:
        data = sio.loadmat(
            item['path'],
            variable_names=['dataset_CSI_t', 'dataset_labels'],
        )
        csi = data['dataset_CSI_t']
        labels = data['dataset_labels'].reshape(-1).astype(np.int64, copy=False)
        next_offset = offset + item['sample_count']

        # Write directly into the final float32 tensor to avoid repeated concatenation
        # and to avoid keeping multiple large CSI copies resident at once.
        compact_data[offset:next_offset, 0, :, :] = csi[:, :, 0:subcarriers:3]
        compact_data[offset:next_offset, 1, :, :] = csi[:, :, 1:subcarriers:3]
        compact_data[offset:next_offset, 2, :, :] = csi[:, :, 2:subcarriers:3]
        compact_labels[offset:next_offset] = labels
        compact_env_ids[offset:next_offset] = item['env_name']
        offset = next_offset

    return compact_data, compact_labels, compact_env_ids


def load_compact_dataset(dataset_entries):
    compact_data, compact_labels, _ = load_compact_dataset_with_env_ids(dataset_entries)
    return compact_data, compact_labels


def print_loaded_dataset_stats(compact_data, compact_labels, title='Loaded dataset summary:'):
    print(title)
    print(f'  data_shape: {compact_data.shape}')
    print(f'  data_dtype: {compact_data.dtype}')
    print(f'  data_bytes: {compact_data.nbytes}')
    print(f'  data_mb: {compact_data.nbytes / (1024 ** 2):.2f}')
    print(f'  labels_shape: {compact_labels.shape}')
    print(f'  labels_dtype: {compact_labels.dtype}')
    print(f'  labels_bytes: {compact_labels.nbytes}')
    print(f'  label_distribution: {format_label_distribution(summarize_labels(compact_labels))}')


def build_split_indices(total_count, split_seed, held_out_ratio=0.2):
    generator = torch.Generator()
    generator.manual_seed(split_seed)
    indices = torch.randperm(total_count, generator=generator).tolist()
    held_out_count = math.ceil(total_count * held_out_ratio)
    train_indices = indices[:-held_out_count]
    held_out_indices = indices[-held_out_count:]
    return train_indices, held_out_indices


def build_aggregated_split_datasets(
    compact_data,
    compact_labels,
    compact_env_ids,
    split_seed,
    eval_envs=None,
):
    num_instances = compact_data.shape[0]
    data_reshape = torch.from_numpy(compact_data)
    transform = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for idx in range(num_instances):
        data_reshape[idx] = transform(data_reshape[idx])

    labels_all = torch.from_numpy(compact_labels).type(torch.LongTensor)
    train_indices, held_out_indices = build_split_indices(num_instances, split_seed)
    train_max = torch.max(data_reshape[train_indices])
    data_reshape = data_reshape / train_max
    train_mean = torch.mean(data_reshape[train_indices])
    data_reshape = data_reshape - train_mean

    dataset = TensorDataset(data_reshape, labels_all)
    eval_indices = held_out_indices
    eval_env_list = None
    if eval_envs:
        eval_env_list = [normalize_environment_name(name) for name in eval_envs]
        eval_env_set = set(eval_env_list)
        eval_indices = [idx for idx in held_out_indices if compact_env_ids[idx] in eval_env_set]
        if not eval_indices:
            selected = ', '.join(eval_env_list)
            raise ValueError(f'No held-out samples were selected for evaluation environments: {selected}')

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    eval_dataset = torch.utils.data.Subset(dataset, eval_indices)
    split_info = {
        'split_seed': split_seed,
        'held_out_ratio': 0.2,
        'train_count': len(train_indices),
        'held_out_count': len(held_out_indices),
        'eval_count': len(eval_indices),
        'eval_envs': eval_env_list,
    }
    return train_dataset, eval_dataset, split_info


def print_training_config(args, device):
    print('Training configuration:')
    print(f'  device: {device}')
    print(f'  ablation: {args.ablation}')
    print(f'  model_variant: {args.model_variant}')
    print(f'  batch_size: {args.batch_size}')
    print(f'  epochs: {args.epochs}')
    print(f'  learning_rate: {args.learning_rate}')
    print(f'  min_learning_rate: {args.min_learning_rate}')
    print(f'  weight_decay: {args.weight_decay}')
    print(f'  loss_type: {args.loss_type}')
    print(f'  scheduler: {args.scheduler}')
    print(f'  decision_threshold: {args.decision_threshold}')
    if args.loss_type == 'focal':
        print(f'  focal_gamma: {args.focal_gamma}')
        print(f'  focal_class_weights: {args.focal_class_weights}')
    print(f'  train_augmentation: {args.train_augmentation}')
    if args.train_augmentation:
        print(f'  augmentation_probability: {args.augmentation_probability}')
        print(f'  augmentation_noise_std: {args.augmentation_noise_std}')
        print(f'  augmentation_shift_max: {args.augmentation_shift_max}')
        print(f'  augmentation_scale: {args.augmentation_scale}')
    print(f'  desired_acc: {args.desired_acc}')
    print(f'  pretrained_backbone: {not args.no_pretrained}')
    print(f'  save_path: {args.save_path}')
    print(f'  init_checkpoint: {args.init_checkpoint}')
    print(f'  metrics_path: {args.metrics_path}')
    print(f'  log_path: {args.log_path}')
    print(f'  split_seed: {args.split_seed}')
    print(f'  train_seed: {args.train_seed}')
    print(f'  smoke_test: {args.smoke_test}')
    print(f'  save_best: {args.save_best}')


def write_json(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def ensure_parent_dir(path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_run_logging(log_path):
    log_path = ensure_parent_dir(log_path)
    log_file = log_path.open('a', encoding='utf-8')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    print(f"\n===== Training run started at {timestamp} =====")
    print(f"Log file: {log_path}")
    return log_file, original_stdout, original_stderr


def teardown_run_logging(log_file, original_stdout, original_stderr):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"===== Training run ended at {timestamp} =====\n")
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


def serialize_history(history):
    serialized = []
    for index, item in enumerate(history):
        row = {'epoch': index}
        for key, value in item.items():
            if isinstance(value, list):
                row[key] = value
            else:
                row[key] = float(value) if isinstance(value, (int, float)) else value
        serialized.append(row)
    return serialized


def build_run_summary(args, scenario_preset, train_entries, test_entries, history, split_info=None):
    def entry_names(entries):
        if entries is None:
            return None
        return [entry['env_name'] for entry in entries]

    best_entry = max(history, key=lambda item: item['val_acc']) if history else None
    last_entry = history[-1] if history else None

    return {
        'scenario': scenario_preset['name'] if scenario_preset else None,
        'scenario_key': normalize_scenario_name(args.scenario) if args.scenario else None,
        'scenario_description': scenario_preset['description'] if scenario_preset else None,
        'model_variant': args.model_variant,
        'train_envs': entry_names(train_entries),
        'test_envs': entry_names(test_entries),
        'config': {
            'model_variant': args.model_variant,
            'ablation': args.ablation,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'min_learning_rate': args.min_learning_rate,
            'weight_decay': args.weight_decay,
            'loss_type': args.loss_type,
            'scheduler': args.scheduler,
            'decision_threshold': args.decision_threshold,
            'focal_gamma': args.focal_gamma,
            'focal_class_weights': args.focal_class_weights,
            'train_augmentation': args.train_augmentation,
            'augmentation_probability': args.augmentation_probability,
            'augmentation_noise_std': args.augmentation_noise_std,
            'augmentation_shift_max': args.augmentation_shift_max,
            'augmentation_scale': args.augmentation_scale,
            'desired_acc': args.desired_acc,
            'split_seed': args.split_seed,
            'train_seed': args.train_seed,
            'init_checkpoint': str(args.init_checkpoint) if args.init_checkpoint else None,
            'pretrained_backbone': not args.no_pretrained,
            'save_best': args.save_best,
        },
        'best': best_entry,
        'last': last_entry,
        'split_info': split_info,
        'history': serialize_history(history),
    }


def print_reference_comparison(run_summary):
    scenario_key = run_summary.get('scenario_key')
    best = run_summary.get('best')
    if not scenario_key or not best:
        return

    reference = SCENARIO_REFERENCE_METRICS.get(scenario_key)
    if reference is None:
        print('No single paper reference row is configured for this scenario.')
        return

    print('\n=== 本次训练结果 vs 论文参考 ===')
    print(f"Reference: {reference['label']}")
    print(f"Accuracy  当前: {format_pct(best.get('val_acc', 0.0))}   论文: {format_pct(reference['acc'])}")
    print(f"Precision 当前: {format_pct(best.get('val_prec', 0.0))}   论文: {format_pct(reference['prec'])}")
    print(f"Recall    当前: {format_pct(best.get('val_rec', 0.0))}   论文: {format_pct(reference['rec'])}")


def print_run_metrics(run_summary):
    best = run_summary.get('best')
    if not best:
        return

    print('\n=== Run Metrics (Best Validation) ===')
    print(f"Acc : {format_pct(best.get('val_acc', 0.0))}")
    print(f"Prec: {format_pct(best.get('val_prec', 0.0))}")
    print(f"Rec : {format_pct(best.get('val_rec', 0.0))}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train the ENetFall baseline model.')
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
        help='Comma-separated environments for the aggregated dataset, for example A,B,C_LoS.',
    )
    parser.add_argument(
        '--train-envs',
        type=str,
        help='Comma-separated environments to use as the explicit training split.',
    )
    parser.add_argument(
        '--test-envs',
        type=str,
        help='Comma-separated environments to use as the explicit test split.',
    )
    parser.add_argument(
        '--check-data-only',
        action='store_true',
        help='Resolve and print dataset paths, then exit without training.',
    )
    parser.add_argument(
        '--split-summary-only',
        action='store_true',
        help='Print train/test or selected-environment sample summaries, then exit.',
    )
    parser.add_argument(
        '--load-data-only',
        action='store_true',
        help='Load the compact dataset, print shape/memory stats, then exit without training.',
    )
    parser.add_argument(
        '--model-variant',
        type=str,
        default=None,
        help='Model family to train, for example baseline, motion-efficientnet-d1, motion-efficientnet-d12, motion-efficientnet, or differential-tcn. Use --ablation A-H for paper presets.',
    )
    parser.add_argument(
        '--fpnet-base-channels',
        type=int,
        default=48,
        help='Base width for the wavelet-fpnet variant.',
    )
    parser.add_argument(
        '--fpnet-pyramid-channels',
        type=int,
        default=160,
        help='Pyramid width for the wavelet-fpnet variant.',
    )
    parser.add_argument(
        '--train-augmentation',
        action='store_true',
        help='Enable CSI augmentation on the training dataset only.',
    )
    parser.add_argument(
        '--augmentation-probability',
        type=float,
        default=0.5,
        help='Probability of applying augmentation to each training sample.',
    )
    parser.add_argument(
        '--augmentation-noise-std',
        type=float,
        default=0.01,
        help='Standard deviation of additive CSI noise after normalization.',
    )
    parser.add_argument(
        '--augmentation-shift-max',
        type=int,
        default=20,
        help='Maximum absolute temporal shift in samples.',
    )
    parser.add_argument(
        '--augmentation-scale',
        type=float,
        default=0.10,
        help='Amplitude scaling range, for example 0.10 means 0.90 to 1.10.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Training batch size. Default is 8 for a safer memory footprint on this machine.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=40,
        help='Number of training epochs. Default is 40 to match the paper more closely.',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate for the optimizer. Defaults to 5e-4 for wavelet-fpnet and 1e-3 for baseline.',
    )
    parser.add_argument(
        '--min-learning-rate',
        type=float,
        default=1e-6,
        help='Minimum learning rate used by cosine annealing.',
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay for the optimizer.',
    )
    parser.add_argument(
        '--loss-type',
        type=str,
        choices=['cross_entropy', 'focal'],
        default=None,
        help='Training loss. Defaults to focal for wavelet-fpnet and cross_entropy for baseline.',
    )
    parser.add_argument(
        '--focal-gamma',
        type=float,
        default=2.0,
        help='Gamma value used by focal loss.',
    )
    parser.add_argument(
        '--focal-class-weights',
        type=str,
        default='1.0,2.0',
        help='Comma-separated class weights for focal loss, for example 1.0,2.0.',
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['step', 'cosine', 'none'],
        default=None,
        help='Learning-rate scheduler. Defaults to cosine for wavelet-fpnet and step for baseline.',
    )
    parser.add_argument(
        '--decision-threshold',
        type=float,
        default=0.5,
        help='Probability threshold for predicting Fall during validation. Default is 0.50 to match code(2).',
    )
    parser.add_argument(
        '--desired-acc',
        type=float,
        default=0.94,
        help='Validation accuracy threshold used by the current early-stop/save logic.',
    )
    parser.add_argument(
        '--split-seed',
        type=int,
        default=42,
        help='Random seed used for deterministic 80/20 dataset splits in aggregated scenarios.',
    )
    parser.add_argument(
        '--train-seed',
        type=int,
        default=None,
        help='Optional random seed for model initialization and data-loader shuffling.',
    )
    parser.add_argument(
        '--save-path',
        type=Path,
        default=None,
        help='Path to save the trained checkpoint.',
    )
    parser.add_argument(
        '--init-checkpoint',
        type=Path,
        default=None,
        help='Optional checkpoint used to initialize model weights before training.',
    )
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Disable ImageNet pretrained weights for the EfficientNet-B0 backbone.',
    )
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Run a 1-epoch sanity check with the configured batch size and then stop.',
    )
    parser.add_argument(
        '--save-best',
        action='store_true',
        help='Save the best checkpoint seen during this run based on validation accuracy.',
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=2,
        help='Number of output classes for the classifier head.',
    )
    parser.add_argument(
        '--metrics-path',
        type=Path,
        default=get_default_results_dir() / 'train_metrics.json',
        help='Path to save run metrics as JSON.',
    )
    parser.add_argument(
        '--log-path',
        type=Path,
        default=get_default_results_dir() / 'train.log',
        help='Path to append the console training log.',
    )
    return parser.parse_args()


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, desired_acc,
                  save_path, criterion, scheduler_type, min_learning_rate=1e-6,
                  smoke_test=False, save_best=False, weight_decay=0, grad_clip=None,
                  decision_threshold=0.5,
                  opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    best_val_acc = float('-inf')
    save_path = ensure_parent_dir(save_path)
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    if scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=min_learning_rate,
        )
    else:
        scheduler = None
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_correct = 0
        train_count = 0
        lrs = []
        for batch in train_loader:
            images, labels = batch
            out = model(images)
            labels = labels.squeeze()
            loss = criterion(out, labels)
            train_losses.append(loss)
            preds = predict_fall_labels(out, decision_threshold=decision_threshold)
            train_correct += int(torch.sum(preds == labels).item())
            train_count += int(labels.numel())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            lrs.append(get_lr(optimizer))

        result = evaluate(
            model,
            val_loader,
            criterion,
            decision_threshold=decision_threshold,
        )
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc'] = train_correct / train_count if train_count else 0.0
        result['lrs'] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
        if save_best and result['val_acc'] > best_val_acc:
            best_val_acc = result['val_acc']
            torch.save(build_checkpoint_payload(model), save_path)
            print(f'Best checkpoint updated: val_acc={best_val_acc:.6f}, path={save_path}')
        if scheduler is not None:
            scheduler.step()
        if smoke_test:
            print('Smoke test completed after 1 epoch. Training stopped intentionally.')
            return history
        if result['val_acc'] >= desired_acc and epoch > 40:
            torch.save(build_checkpoint_payload(model), save_path)
            return history
    return history


if __name__ == '__main__':
    args = parse_args()
    if args.num_classes < 2:
        raise ValueError('--num-classes must be at least 2')
    if not 0.0 < args.decision_threshold < 1.0:
        raise ValueError('--decision-threshold must be between 0 and 1')
    if not 0.0 <= args.augmentation_probability <= 1.0:
        raise ValueError('--augmentation-probability must be between 0 and 1')
    if args.augmentation_noise_std < 0:
        raise ValueError('--augmentation-noise-std must be non-negative')
    if args.augmentation_shift_max < 0:
        raise ValueError('--augmentation-shift-max must be non-negative')
    if not 0.0 <= args.augmentation_scale < 1.0:
        raise ValueError('--augmentation-scale must be between 0 and 1')
    if args.ablation is not None and args.model_variant is not None:
        raise ValueError('--ablation cannot be combined with --model-variant.')
    if args.ablation is not None:
        args.model_variant = resolve_ablation_variant(args.ablation)
    args.model_variant = normalize_model_variant(args.model_variant)
    if args.model_variant.endswith('-aug'):
        args.train_augmentation = True
    if args.train_seed is not None:
        set_global_seed(args.train_seed)
    args.learning_rate = resolve_learning_rate(args)
    args.loss_type = resolve_loss_type(args)
    args.scheduler = resolve_scheduler_type(args)
    if args.save_path is None:
        if args.model_variant == 'baseline':
            args.save_path = get_default_weights_dir() / 'ENetmod_pretrained.pth'
        else:
            safe_variant = args.model_variant.replace('-', '_')
            args.save_path = get_default_weights_dir() / f'code3_{safe_variant}.pth'
    num_classes = args.num_classes
    device = get_default_device()
    if args.smoke_test:
        args.epochs = 1
    log_file, original_stdout, original_stderr = setup_run_logging(args.log_path)
    try:
        print_training_config(args, device)
        train_entries, test_entries, selected_entries, scenario_preset = build_dataset_entries(args)
        print_available_environments()
        print_available_scenarios()
        if scenario_preset is not None:
            print(f"Using {scenario_preset['name']}: {scenario_preset['description']}")

        active_train_entries = None
        active_test_entries = None
        split_info = None

        if selected_entries is not None:
            print_environment_selection(selected_entries, 'Selected environments')
            print_dataset_paths(selected_entries)
            if args.check_data_only:
                print('Data path check completed. Training was not started.')
                raise SystemExit(0)
            if args.split_summary_only:
                print_split_summary(selected_entries=selected_entries)
                print('Split summary check completed. Training was not started.')
                raise SystemExit(0)

            compact_data, compact_labels, compact_env_ids = load_compact_dataset_with_env_ids(selected_entries)
            if args.load_data_only:
                print_loaded_dataset_stats(compact_data, compact_labels)
                print('Compact data load check completed. Training was not started.')
                raise SystemExit(0)

            eval_envs = None
            if scenario_preset is not None:
                eval_envs = scenario_preset.get('eval_envs')
            train_dataset, test_dataset, split_info = build_aggregated_split_datasets(
                compact_data,
                compact_labels,
                compact_env_ids,
                args.split_seed,
                eval_envs=eval_envs,
            )
            active_train_entries = selected_entries
            if eval_envs:
                active_test_entries = resolve_dataset_paths(args.data_root, ','.join(eval_envs))
                print_environment_selection(active_test_entries, 'Evaluation environments from held-out split')
            else:
                active_test_entries = selected_entries
        else:
            print_environment_selection(train_entries, 'Train environments')
            print_dataset_paths(train_entries, header='Resolved train dataset files:')
            print_environment_selection(test_entries, 'Test environments')
            print_dataset_paths(test_entries, header='Resolved test dataset files:')
            if args.check_data_only:
                print('Data path check completed. Training was not started.')
                raise SystemExit(0)
            if args.split_summary_only:
                print_split_summary(train_entries=train_entries, test_entries=test_entries)
                print('Split summary check completed. Training was not started.')
                raise SystemExit(0)

            train_data_np, train_labels_np = load_compact_dataset(train_entries)
            test_data_np, test_labels_np = load_compact_dataset(test_entries)
            if args.load_data_only:
                print_loaded_dataset_stats(train_data_np, train_labels_np, title='Loaded training dataset summary:')
                print_loaded_dataset_stats(test_data_np, test_labels_np, title='Loaded test dataset summary:')
                print('Compact data load check completed. Training was not started.')
                raise SystemExit(0)

            train_data_reshape = torch.from_numpy(train_data_np)
            test_data_reshape = torch.from_numpy(test_data_np)
            transform = T.Compose([
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            for idx in range(train_data_reshape.shape[0]):
                train_data_reshape[idx] = transform(train_data_reshape[idx])
            for idx in range(test_data_reshape.shape[0]):
                test_data_reshape[idx] = transform(test_data_reshape[idx])

            train_max = torch.max(train_data_reshape)
            train_data_reshape = train_data_reshape / train_max
            test_data_reshape = test_data_reshape / train_max
            train_mean = torch.mean(train_data_reshape)
            train_data_reshape = train_data_reshape - train_mean
            test_data_reshape = test_data_reshape - train_mean

            train_labels = torch.from_numpy(train_labels_np).type(torch.LongTensor)
            test_labels = torch.from_numpy(test_labels_np).type(torch.LongTensor)
            train_dataset = TensorDataset(train_data_reshape, train_labels)
            test_dataset = TensorDataset(test_data_reshape, test_labels)
            active_train_entries = train_entries
            active_test_entries = test_entries

        if args.train_augmentation:
            train_dataset = CSIAugmentedDataset(
                train_dataset,
                probability=args.augmentation_probability,
                noise_std=args.augmentation_noise_std,
                shift_max=args.augmentation_shift_max,
                amplitude_scale=args.augmentation_scale,
            )
            print('Training augmentation enabled. Validation data remains unchanged.')

        train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        train_data_loader = DeviceDataLoader(train_data_loader, device)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_data_loader = DeviceDataLoader(test_data_loader, device)

        model_ft = build_model(
            args.model_variant,
            num_classes=num_classes,
            pretrained_backbone=not args.no_pretrained,
            base_channels=args.fpnet_base_channels,
            pyramid_channels=args.fpnet_pyramid_channels,
        )
        model_ft = model_ft.to(device)
        if args.init_checkpoint is not None:
            checkpoint = torch.load(args.init_checkpoint, map_location=device)
            state_dict = extract_state_dict(checkpoint)
            missing_keys, unexpected_keys = model_ft.load_state_dict(state_dict, strict=False)
            print(f'Initialized from checkpoint: {args.init_checkpoint}')
            if missing_keys:
                print(f'  missing_keys: {len(missing_keys)}')
            if unexpected_keys:
                print(f'  unexpected_keys: {len(unexpected_keys)}')
        criterion = create_loss_fn(args, device)
        history = []
        history += fit_one_cycle(
            args.epochs,
            args.learning_rate,
            model_ft,
            train_data_loader,
            test_data_loader,
            args.desired_acc,
            args.save_path,
            criterion,
            args.scheduler,
            args.min_learning_rate,
            smoke_test=args.smoke_test,
            save_best=args.save_best,
            grad_clip=None,
            weight_decay=args.weight_decay,
            decision_threshold=args.decision_threshold,
            opt_func=torch.optim.Adam,
        )
        run_summary = build_run_summary(
            args,
            scenario_preset,
            active_train_entries,
            active_test_entries,
            history,
            split_info=split_info,
        )
        write_json(args.metrics_path, run_summary)
        print(f"Metrics written to {args.metrics_path}")
        print_run_metrics(run_summary)
    finally:
        teardown_run_logging(log_file, original_stdout, original_stderr)
