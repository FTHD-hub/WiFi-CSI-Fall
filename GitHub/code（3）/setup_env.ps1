param(
    [string]$EnvName = "fd"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating conda environment '$EnvName' with Python 3.10..."
conda create -y -n $EnvName python=3.10

Write-Host "Installing PyTorch with CUDA 12.1 support..."
conda run -n $EnvName pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing SciPy..."
conda run -n $EnvName pip install scipy

Write-Host "Verifying the environment..."
conda run -n $EnvName python -c "import torch, torchvision, scipy, numpy as np; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('scipy', scipy.__version__); print('numpy', np.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
