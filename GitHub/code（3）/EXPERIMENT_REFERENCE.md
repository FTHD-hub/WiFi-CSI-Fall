# WiFi Fall Detection 实验结果参考

这份文档把论文里的关键实验结果和当前复现实验的输出方式整理到一起，方便后续对照和复现。

## 论文截图参考

![Aggregated Dataset](/D:/back-wifi/ENetFall-main/paper_aggregated_dataset.png)

## 论文实验结果

### Scenario 1: Aggregated Dataset

| Method | Acc | Prec | Rec |
| --- | --- | --- | --- |
| Modified B0 [34] | 94.9% | 91.7% | 96.8% |
| FallDeFi | 80.8% | 75.0% | 81.9% |
| Work in [28] | 88.0% | 82.9% | 90.2% |
| Proposed | 99.5% | 98.8% | 100.0% |

### Scenario 2: NLoS Dataset (C_NLoS)

| Method | Acc | Prec | Rec |
| --- | --- | --- | --- |
| Modified B0 [34] | 82.0% | 72.7% | 100.0% |
| FallDeFi | 68.2% | 66.4% | 70.9% |
| Work in [28] | 75.4% | 72.7% | 76.2% |
| Proposed | 97.6% | 97.3% | 100.0% |

### Scenario 3: Unseen Room A and D

#### Lecture Room D

| Method | Acc | Prec | Rec | F1 |
| --- | --- | --- | --- | --- |
| Modified B0 [34] | 72.0% | 69.0% | 77.0% | 73.0% |
| FallDeFi | 68.0% | 66.0% | 72.0% | 69.0% |
| Work in [28] | 73.0% | 72.0% | 77.0% | 74.0% |
| Proposed | 92.0% | 94.0% | 90.0% | 92.0% |

#### Living Room A

| Method | Acc | Prec | Rec | F1 |
| --- | --- | --- | --- | --- |
| Modified B0 [34] | 69.0% | 55.0% | 90.0% | 68.0% |
| FallDeFi | 60.0% | 49.0% | 79.0% | 61.0% |
| Work in [28] | 66.0% | 53.0% | 94.0% | 68.0% |
| Proposed | 88.0% | 82.0% | 87.0% | 84.0% |

### Scenario 4: Sparse Training Data

#### Left Living Room C / LoS

| Method | Acc | Prec | Rec | F1 |
| --- | --- | --- | --- | --- |
| Modified B0 [34] | 90.0% | 92.0% | 88.0% | 90.0% |
| FallDeFi | 73.0% | 69.0% | 80.0% | 74.0% |
| Work in [28] | 76.0% | 74.0% | 83.0% | 78.0% |
| Proposed | 99.0% | 98.0% | 100.0% | 99.0% |

#### Right Living Room C / NLoS

| Method | Acc | Prec | Rec | F1 |
| --- | --- | --- | --- | --- |
| Modified B0 [34] | 78.0% | 66.0% | 95.0% | 78.0% |
| FallDeFi | 68.0% | 62.0% | 76.0% | 68.0% |
| Work in [28] | 73.0% | 68.0% | 85.0% | 76.0% |
| Proposed | 93.0% | 92.0% | 92.0% | 92.0% |

## 当前代码如何导出结果

训练并保存 JSON：

```powershell
python train.py --scenario scenario2 --batch-size 8 --epochs 50 --save-best --save-path results\scenario2_best.pth --metrics-path results\scenario2_train.json --log-path results\scenario2_train.log
```

测试并保存 JSON：

```powershell
python test.py --scenario scenario2 --weights-path results\scenario2_best.pth --metrics-path results\scenario2_test.json
```

把 JSON 自动整理成论文风格的 Markdown 报告：

```powershell
python ENetFall-main\report_results.py --scenario scenario2 --train-json results\scenario2_train.json --test-json results\scenario2_test.json --output results\scenario2_report.md --image-path /D:/back-wifi/ENetFall-main/paper_aggregated_dataset.png
```

## 说明

- 当前 `train.py` 导出的 `train_metrics.json` 里包含整数据集级别的验证指标。
- 当前 `test.py` 导出的 `test_metrics.json` 里可以进一步生成分类报告。
- 上面的 Scenario 3 / 4 数值来自论文图中的标注，适合做复现对照参考。
