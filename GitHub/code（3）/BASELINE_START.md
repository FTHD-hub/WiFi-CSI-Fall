# ENetFall Baseline Start Guide

This guide only covers the baseline entry point, data paths, and safe startup checks.

## Dataset root

The current baseline expects these `.mat` files under the dataset root:

- `dataset_home_lab(L).mat`
- `dataset_home_lab(R).mat`
- `dataset_lecture_room.mat`
- `dataset_living_room.mat`
- `dataset_meeting_room.mat`

In your current workspace, the dataset root is:

```text
D:\back-wifi
```

## Recommended working directory

Run commands from:

```text
D:\back-wifi\ENetFall-main
```

## Activate the environment

```powershell
conda activate fd
```

## Check data paths without training

```powershell
python train.py --check-data-only
```

This uses the default dataset root:

```text
D:\back-wifi
```

## Check data paths with an explicit dataset root

```powershell
python train.py --data-root D:\back-wifi --check-data-only
```

## Load compact data and print memory or shape stats

```powershell
python train.py --load-data-only
```

This will:

- resolve the dataset paths
- load the compact `float32` tensor
- print the final data shape
- print the final in-memory size
- exit before training starts

## Show environment split summary without training

```powershell
python train.py --split-summary-only
```

## Available environment names

- `A`: Living Room A
- `B`: Meeting Room B
- `C_LoS`: Home Lab C LoS
- `C_NLoS`: Home Lab C NLoS
- `D`: Lecture Room D

## Example: cross-environment summary

This matches the paper-style idea of training on one set of rooms and testing on another:

```powershell
python train.py --train-envs B,C_LoS,C_NLoS --test-envs A,D --split-summary-only
```

## Example: use a selected subset with the original random 80/20 split

```powershell
python train.py --envs A,B,D --split-summary-only
```

## Paper scenario presets

You can now call the paper-style setups directly:

```powershell
python train.py --scenario scenario1 --split-summary-only
python train.py --scenario scenario2 --split-summary-only
python train.py --scenario scenario3 --split-summary-only
python train.py --scenario scenario4 --split-summary-only
```

Their meanings are:

- `scenario1`: all environments together, then random `80/20`
- `scenario2`: train on the aggregated dataset, test on `C_NLoS`
- `scenario3`: train on `B + C_LoS + C_NLoS`, test on `A + D`
- `scenario4`: train on `B + D`, test on `C_LoS + C_NLoS`

## Smoke test command

When you want to verify that training can start without committing to a long run, use:

```powershell
python train.py --scenario scenario1 --smoke-test --batch-size 8
```

This will:

- use the baseline training loop
- run exactly 1 epoch
- keep the safer `batch_size=8`
- stop intentionally after the sanity check

## Reproduce experiment result files

Train and save metrics to JSON:

```powershell
python train.py --scenario scenario2 --batch-size 8 --epochs 50 --save-best --save-path results\scenario2_best.pth --metrics-path results\scenario2_train.json
```

Evaluate a saved checkpoint and export metrics:

```powershell
python test.py --scenario scenario2 --weights-path results\scenario2_best.pth --metrics-path results\scenario2_test.json
```

The exported JSON files now contain dataset-level metrics such as:

- `val_acc`
- `val_prec`
- `val_rec`
- `val_f1`

## Baseline training command

When you are ready to actually train later, use:

```powershell
python train.py --data-root D:\back-wifi
```

## Baseline evaluation command

To evaluate a trained checkpoint later, use:

```powershell
python test.py --data-root D:\back-wifi --weights-path "B0(modified)_trained_with_all_data.pth"
```

## What changed

- `train.py` now resolves dataset files from a configurable `--data-root`.
- `test.py` now resolves both `--data-root` and `--weights-path`.
- `--check-data-only` lets you validate the startup path setup without launching training or evaluation.
