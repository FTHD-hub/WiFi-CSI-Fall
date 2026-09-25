param(
    [int]$Epochs = 40,
    [int]$BatchSize = 8,
    [string]$RunId = $(Get-Date -Format "yyyyMMdd_HHmmss")
)

$ErrorActionPreference = "Stop"

$Conda = "C:\Users\lenovo\anaconda3\Scripts\conda.exe"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ResultsDir = Join-Path $ProjectRoot "results"
$WeightsDir = Join-Path $ProjectRoot "weights"
$MasterLog = Join-Path $ResultsDir "scenario3_ablation_$RunId.master.log"

$experiments = @(
    @{
        Name = "baseline"
        Args = @("--model-variant", "baseline")
    },
    @{
        Name = "differential_tcn"
        Args = @("--model-variant", "differential-tcn")
    },
    @{
        Name = "motion_enet"
        Args = @("--model-variant", "motion-efficientnet")
    },
    @{
        Name = "motion_enet_no_pretrain"
        Args = @("--model-variant", "motion-efficientnet", "--no-pretrained")
    }
)

Set-Location $PSScriptRoot
"Scenario3 ablation started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Tee-Object -FilePath $MasterLog
"RunId: $RunId" | Tee-Object -FilePath $MasterLog -Append

foreach ($experiment in $experiments) {
    $name = $experiment.Name
    $logPath = Join-Path $ResultsDir "scenario3_ablation_${name}_${RunId}.log"
    $metricsPath = Join-Path $ResultsDir "scenario3_ablation_${name}_${RunId}_metrics.json"
    $savePath = Join-Path $WeightsDir "scenario3_ablation_${name}_${RunId}_best.pth"

    "`n===== $name started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" |
        Tee-Object -FilePath $MasterLog -Append

    $trainArgs = @(
        "run",
        "--no-capture-output",
        "-n",
        "fd",
        "python",
        "train.py",
        "--scenario",
        "scenario3",
        "--epochs",
        "$Epochs",
        "--batch-size",
        "$BatchSize",
        "--save-best",
        "--metrics-path",
        $metricsPath,
        "--log-path",
        $logPath,
        "--save-path",
        $savePath
    ) + $experiment.Args

    & $Conda @trainArgs 2>&1 | Tee-Object -FilePath $MasterLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment $name failed with exit code $LASTEXITCODE"
    }

    "===== $name ended at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" |
        Tee-Object -FilePath $MasterLog -Append
}

"`nScenario3 ablation finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" |
    Tee-Object -FilePath $MasterLog -Append
