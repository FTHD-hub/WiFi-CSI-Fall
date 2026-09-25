# ENetFall Baseline 环境说明
## 创建环境

在 `D:\back-wifi` 目录下运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\ENetFall-main\setup_env.ps1
```

## 激活环境

```powershell
conda activate fd
```

## 已安装依赖

- Python `3.10`
- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
- `scipy==1.15.3`
- `numpy==2.2.6`

## 环境验证命令
```powershell
conda run -n fd python -c "import torch, torchvision, scipy, numpy as np; print(torch.__version__); print(torch.cuda.is_available())"
```
