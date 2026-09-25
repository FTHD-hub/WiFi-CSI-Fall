# Ablation Experiments

The ablation presets are available through `--ablation A` to `--ablation H`.
All presets keep the binary Fall/Nonfall output and the 2-layer MLP head.

| Preset | Model | Difference |
| --- | --- | --- |
| A | baseline | Pretrained EfficientNet-B0 |
| B | b0-dwt | B0 + time-axis db4 DWT, level 3 |
| C | b0-dwt-soft | B0 + db4 DWT + soft threshold |
| D | b0-dwt-motion | B0 + db4 DWT + static/motion gate |
| E | b0-dwt-aug | B0 + db4 DWT + training augmentation |
| F | dwt-transformer-soft | db4 DWT + soft threshold + Transformer |
| G | dwt-transformer-motion | F + static/motion gate |
| H | dwt-transformer-motion-aug | G + training augmentation |

The wavelet is applied along the time axis only. The subcarrier axis is kept
as a feature axis to avoid treating subcarriers as image height/width.

## Smoke Test

```powershell
cd D:\back-wifi\code3
python train.py --ablation B --scenario scenario3 --batch-size 8 --epochs 1 --smoke-test
python train.py --ablation H --scenario scenario3 --batch-size 8 --epochs 1 --smoke-test
```

## Full Runs

Use the same optimizer settings for every preset when comparing modules:

```powershell
cd D:\back-wifi\code3
python train.py --ablation A --scenario scenario3 --batch-size 8 --epochs 50 --learning-rate 5e-4 --loss-type cross_entropy --scheduler cosine --save-best --save-path ..\weights\ablation_A_best.pth --metrics-path ..\results\ablation_A_train.json --log-path ..\results\ablation_A_train.log
```

Replace `A` in the command with `B`, `C`, `D`, `E`, `F`, `G`, or `H`.
Presets E and H automatically enable training-only augmentation.

## Evaluation

```powershell
python test.py --ablation A --scenario scenario3 --weights-path ..\weights\ablation_A_best.pth --metrics-path ..\results\ablation_A_test.json
```

Use the same preset letter as the training run. If `--ablation` is omitted,
`test.py` reads the model variant from checkpoint metadata.

For strict reporting, do not use the target-room test set to select the best
epoch. The current legacy explicit-scenario loop still reports target-room
metrics every epoch; use a source-environment validation split before using
these numbers in a paper.
