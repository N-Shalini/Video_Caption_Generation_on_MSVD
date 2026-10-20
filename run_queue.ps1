param([string]$Jobs = "temporal:43,spatiotemporal_attention:42")

Set-Location C:\dev\video
$env:OMP_NUM_THREADS = "3"; $env:MKL_NUM_THREADS = "3"; $env:PYTHONUNBUFFERED = "1"
$py = (Get-Command python).Source
New-Item -ItemType Directory -Force results\logs | Out-Null

function Run-Step($arguments, $logBase) {
  $p = Start-Process $py -ArgumentList $arguments -NoNewWindow -PassThru `
       -RedirectStandardOutput "$logBase.log" -RedirectStandardError "$logBase.err"
  $h = $p.Handle
  $p.PriorityClass = "BelowNormal"
  $p.ProcessorAffinity = 0xFC
  $p.WaitForExit()
}

foreach ($item in $Jobs.Split(",")) {
  $parts = $item.Split(":")
  $enc   = $parts[0]
  $seed  = $parts[1]
  $attn  = if ($parts.Length -gt 2 -and $parts[2]) { $parts[2] } else { $null }
  $feat  = if ($parts.Length -gt 3 -and $parts[3]) { $parts[3] } else { $null }

  if ($feat -and $feat -ne "vgg16_spatial") {
    if ($attn -and $attn -ne "additive") {
      $run = "${enc}_${attn}_${feat}_v2_s${seed}"
    } else {
      $run = "${enc}_${feat}_v2_s${seed}"
    }
  } elseif ($attn -and $attn -ne "additive") {
    $run = "${enc}_${attn}_v2_s${seed}"
  } else {
    $run = "${enc}_v2_s${seed}"
  }
  $best    = "checkpoints\${run}_best.pt"
  $last    = "checkpoints\${run}_last.pt"
  $evalOut = "results\${run}_best_test_evaluation.json"

  $done = $false
  $logs = Get-ChildItem "results\logs\${run}*.log" -ErrorAction SilentlyContinue
  if ($logs) { $done = [bool]($logs | Select-String -Pattern "TRAINING COMPLETED" -Quiet) }

  if (-not $done) {
    $extra = ""; $logBase = "results\logs\$run"
    if (Test-Path $last) { $extra = " --resume"; $logBase = "results\logs\${run}_resume" }
    $attnFlag = if ($attn) { " --attention-type $attn" } else { "" }
    $featFlag = if ($feat) { " --feature-type $feat" } else { "" }
    Write-Host "$(Get-Date) training $run$extra"
    Run-Step "train_spatiotemporal.py --encoder-type $enc --seed $seed --run-name $run$attnFlag$featFlag$extra" $logBase
  }

  if ((Test-Path $best) -and -not (Test-Path $evalOut)) {
    Write-Host "$(Get-Date) evaluating $run"
    Run-Step "evaluate_test_set.py --checkpoint $best --decoding-config results\fixed_beam5_config.json" "results\logs\${run}_eval"
    Remove-Item results\test_set_evaluation.json, results\test_set_predictions.json -ErrorAction SilentlyContinue
  }
}
Write-Host "$(Get-Date) queue finished"