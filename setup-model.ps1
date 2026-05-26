# Run this ONCE on your dev machine before the first `.\build.bat`.
#
# It copies the Whisper 'small' model from your existing HuggingFace
# cache into ./models/hf-cache/hub/ so the PyInstaller spec can bundle
# it next to the .exe. After this runs, every build / friend's launch
# is fully offline - zero downloads.
#
# If you've never downloaded 'small' before, this script will fall
# through and tell you to run `python run.py` once first to fetch it.

$ErrorActionPreference = "Stop"

$dst = ".\models\hf-cache\hub"
New-Item -ItemType Directory -Force -Path $dst | Out-Null

$searchRoots = @(
    "$env:USERPROFILE\.cache\huggingface\hub",
    "$env:USERPROFILE\.cluely_killer\hf-cache\hub"
)

$found = $false
foreach ($root in $searchRoots) {
    if (-not (Test-Path $root)) { continue }
    Get-ChildItem -Path $root -Directory -Filter "models--*--faster-whisper-small" -ErrorAction SilentlyContinue | ForEach-Object {
        $target = Join-Path $dst $_.Name
        if (Test-Path $target) {
            Write-Host "Already bundled: $($_.Name)" -ForegroundColor Green
        } else {
            Write-Host "Copying: $($_.FullName) -> $target" -ForegroundColor Cyan
            Copy-Item -Path $_.FullName -Destination $target -Recurse -Force
        }
        $script:found = $true
    }
}

if (-not $found) {
    Write-Host ""
    Write-Host "ERROR: Whisper 'small' model not found in any HF cache." -ForegroundColor Red
    Write-Host "Run 'python run.py' once on this machine first - it will download" -ForegroundColor Red
    Write-Host "the model (~466 MB) to your HF cache. Then re-run this script." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Done. Bundled model contents:" -ForegroundColor Green
Get-ChildItem $dst -Recurse | Where-Object { $_.Length -gt 1MB } | Format-Table FullName, @{N='SizeMB';E={[math]::Round($_.Length/1MB,0)}} -AutoSize
Write-Host "Now run .\build.bat to build the .exe with the bundled model."
