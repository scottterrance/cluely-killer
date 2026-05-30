# Stage a faster-whisper model into ./models/whisper-<model>/ as a FLAT
# directory (model.bin + config.json + tokenizer.json + vocabulary.txt),
# which is exactly what the app and build.bat expect. Run this ONCE on
# your dev machine before `.\build.bat`.
#
# Usage:
#   .\setup-model.ps1                       # downloads large-v3-turbo (default)
#   .\setup-model.ps1 -Model small          # stage a different size
#   .\setup-model.ps1 -FromCache            # don't download; copy from HF cache
#
# Default mode DOWNLOADS the model directly into the target folder using
# faster-whisper's own downloader (no HuggingFace cache structure to
# flatten, no symlinks to resolve). This is the simplest, most reliable
# path. Use -FromCache only if you already pulled the model some other
# way and just want to flatten an existing HF cache snapshot.

param(
    [string]$Model = "large-v3-turbo",
    [switch]$FromCache
)

$ErrorActionPreference = "Stop"
$dst = ".\models\whisper-$Model"

# Wipe any previous (possibly partial) staging.
if (Test-Path $dst) {
    Write-Host "Removing existing $dst (re-staging from scratch)..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $dst
}
New-Item -ItemType Directory -Force -Path $dst | Out-Null

if (-not $FromCache) {
    # ---- Primary path: download straight into the flat target dir ----
    Write-Host "Downloading '$Model' into $dst via faster-whisper..." -ForegroundColor Cyan
    Write-Host "(one-time; needs internet; ~1.6 GB for large-v3-turbo)" -ForegroundColor DarkGray

    $py = @"
from faster_whisper.utils import download_model
# download_model accepts either a size alias ('large-v3-turbo', 'small')
# or an explicit HF repo id. output_dir gets the flat CT2 files.
try:
    p = download_model('$Model', output_dir=r'$dst')
except Exception as e:
    # Fallback to an explicit, known-good CT2 turbo repo if the size
    # alias isn't recognized by this faster-whisper version.
    if '$Model' == 'large-v3-turbo':
        print('[setup] size alias failed, trying deepdml CT2 repo...', flush=True)
        p = download_model('deepdml/faster-whisper-large-v3-turbo-ct2', output_dir=r'$dst')
    else:
        raise
print('[setup] downloaded to:', p, flush=True)
"@
    python -c $py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: download failed. Is 'pip install faster-whisper' done and internet up?" -ForegroundColor Red
        exit 1
    }
}
else {
    # ---- Fallback path: flatten an existing HF cache snapshot ----
    $searchRoots = @(
        "$env:USERPROFILE\.cache\huggingface\hub",
        "$env:USERPROFILE\.cluely_killer\hf-cache\hub"
    )
    $srcSnapshot = $null
    foreach ($root in $searchRoots) {
        if (-not (Test-Path $root)) { continue }
        $cacheFolders = Get-ChildItem -Path $root -Directory -Filter "models--*--*$Model*" -ErrorAction SilentlyContinue
        foreach ($cacheFolder in $cacheFolders) {
            $snapshots = Join-Path $cacheFolder.FullName "snapshots"
            if (-not (Test-Path $snapshots)) { continue }
            foreach ($snap in (Get-ChildItem -Path $snapshots -Directory -ErrorAction SilentlyContinue)) {
                if (Test-Path (Join-Path $snap.FullName "model.bin")) {
                    $srcSnapshot = $snap.FullName
                    Write-Host "Found snapshot: $srcSnapshot" -ForegroundColor Green
                    break
                }
            }
            if ($srcSnapshot) { break }
        }
        if ($srcSnapshot) { break }
    }
    if (-not $srcSnapshot) {
        Write-Host "ERROR: no HF cache snapshot for '$Model' found. Run without -FromCache to download." -ForegroundColor Red
        exit 1
    }
    Get-ChildItem -Path $srcSnapshot -File | ForEach-Object {
        $target = Join-Path $dst $_.Name
        $resolved = $_.FullName
        try {
            $item = Get-Item -LiteralPath $_.FullName
            if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                $linkTarget = (Get-Item -LiteralPath $_.FullName).Target
                if ($linkTarget) {
                    $candidate = Join-Path (Split-Path $_.FullName) $linkTarget
                    if (Test-Path $candidate) { $resolved = (Resolve-Path $candidate).Path }
                }
            }
        } catch { }
        Copy-Item -Path $resolved -Destination $target -Force
        Write-Host "Copied: $($_.Name) ($([math]::Round((Get-Item $target).Length / 1MB, 1)) MB)" -ForegroundColor Cyan
    }
}

# ---- Sanity check ----
$modelBin = Join-Path $dst "model.bin"
if (-not (Test-Path $modelBin)) {
    Write-Host "ERROR: model.bin missing in $dst." -ForegroundColor Red
    exit 1
}
$sizeMB = [math]::Round((Get-Item $modelBin).Length / 1MB, 0)
if ($sizeMB -lt 100) {
    Write-Host "ERROR: model.bin is only $sizeMB MB - looks incomplete." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Done. $dst contents:" -ForegroundColor Green
Get-ChildItem $dst | Format-Table Name, @{N='SizeMB';E={[math]::Round($_.Length/1MB,1)}} -AutoSize
Write-Host "model.bin = $sizeMB MB. Now run .\build.bat to bundle it with the .exe." -ForegroundColor Green
