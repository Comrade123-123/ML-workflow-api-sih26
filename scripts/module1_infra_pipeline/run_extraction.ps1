# Drives extract_one_report.py over every raw PDF using genuine Windows
# process control (Start-Process + Wait-Process -Timeout + Stop-Process).
# A prior attempt used MSYS git-bash's `timeout` command, but it does not
# reliably kill a native python.exe child on this Windows setup -- the
# process kept running well past the configured limit. PowerShell's own
# process cmdlets do work (confirmed: Stop-Process -Force has successfully
# killed stuck python.exe processes earlier in this session).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")

$rawDir = "data/module1_infra_source/raw_reports"
$extractedDir = "data/module1_infra_source/extracted"
# 34MB+/700+ page re-downloaded reports have been observed taking ~4-5 min
# to extract legitimately (confirmed: FR_2020_05 succeeded after ~4-5 min in
# an earlier run). A separate, genuine hang (FR_2020_06: near-zero CPU for
# 17+ min) is a different failure mode. 600s comfortably covers legitimately
# slow large PDFs while still bounding a true hang.
$timeoutSecs = 600
$pollIntervalSecs = 5

New-Item -ItemType Directory -Force -Path $extractedDir | Out-Null
Get-ChildItem "$extractedDir/*_summary_partial.json" -ErrorAction SilentlyContinue | Remove-Item -Force

$pdfs = Get-ChildItem "$rawDir/*.pdf" | Sort-Object Name
foreach ($pdf in $pdfs) {
    Write-Host "Processing $($pdf.Name) ..."
    $proc = Start-Process -FilePath "python" -ArgumentList @("scripts/module1_infra_pipeline/extract_one_report.py", $pdf.FullName) -PassThru -NoNewWindow
    # Manual poll loop instead of $proc.WaitForExit(ms) -- WaitForExit's
    # timeout overload was observed not to fire reliably in this environment
    # (a process kept running well past its configured timeout with no
    # "FAILED: timed out" message ever printed).
    $elapsed = 0
    while (-not $proc.HasExited -and $elapsed -lt $timeoutSecs) {
        Start-Sleep -Seconds $pollIntervalSecs
        $elapsed += $pollIntervalSecs
    }
    if (-not $proc.HasExited) {
        Write-Host "  FAILED: timed out after ${timeoutSecs}s (pypdf hang) -- killing process"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($pdf.Name)
        $hash = (Get-FileHash -Algorithm SHA256 -Path $pdf.FullName).Hash.ToLower()
        $errObj = @{
            report_id = $stem
            source_file = $pdf.Name
            sha256 = $hash
            size_bytes = $pdf.Length
            error = "TimeoutError: extraction exceeded ${timeoutSecs}s"
        }
        $errObj | ConvertTo-Json | Set-Content -Encoding utf8 "$extractedDir/${stem}_summary_partial.json"
    }
}

python -c "
import json
from pathlib import Path
extracted = Path('$extractedDir')
summaries = []
for f in sorted(extracted.glob('*_summary_partial.json')):
    summaries.append(json.loads(f.read_text(encoding='utf-8')))
    f.unlink()
(extracted / 'extraction_summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
print(f'Wrote extraction_summary.json for {len(summaries)} reports.')
"
