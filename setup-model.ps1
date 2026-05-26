# Run this ONCE on your dev machine before the first `.\build.bat`.
#
# It finds the Whisper 'small' model in any standard HuggingFace cache
# location, then copies its files (FLAT, no HF cache structure) into
# ./models/whisper-small/ where the app and PyInstaller spec expect them.
#
# Why flat instead of preserving HF's blobs/refs/snapshots tree?
# Because PyInstaller mangles the HF cache structure when bundling, and
# faster-whisper's "load from a directory path" mode bypasses HF
# entirely. We want a directory containing model.bin + config.json +
# tokenizer.json + a few other small files, period.

$ErrorActionPreference = "Stop"

$dst = ".\models\whisper-small"

# Wipe any previous (possibly partial) staging.
if (Test-Path $dst) {
    Write-Host "Removing existing $dst (re-staging from scratch)..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $dst
}
New-Item -ItemType Directory -Force -Path $dst | Out-Null

# Look for the HF-cache-style folder for Systran/faster-whisper-small in
# a few well-known locations. The folder name has a per-org prefix
# (Systran, mobiuslabsgmbh, ...) so we glob 'models--*--faster-whisper-small'.
$searchRoots = @(
    "$env:USERPROFILE\.cache\huggingface\hub",
    "$env:USERPROFILE\.cluely_killer\hf-cache\hub"
)

$srcSnapshot = $null
foreach ($root in $searchRoots) {
    if (-not (Test-Path $root)) { continue }
    $cacheFolders = Get-ChildItem -Path $root -Directory -Filter "models--*--faster-whisper-small" -ErrorAction SilentlyContinue
    foreach ($cacheFolder in $cacheFolders) {
        # Each cacheFolder has refs/main + blobs/* + snapshots/<sha>/.
        # The actual usable files live in snapshots/<sha>/.
        $snapshots = Join-Path $cacheFolder.FullName "snapshots"
        if (-not (Test-Path $snapshots)) { continue }
        $snapshotFolders = Get-ChildItem -Path $snapshots -Directory -ErrorAction SilentlyContinue
        foreach ($snap in $snapshotFolders) {
            # Pick the first snapshot that actually has a model.bin.
            $modelBin = Join-Path $snap.FullName "model.bin"
            if (Test-Path $modelBin) {
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
    Write-Host ""
    Write-Host "ERROR: Whisper 'small' model snapshot not found." -ForegroundColor Red
    Write-Host "Run this once first to download it:" -ForegroundColor Red
    Write-Host '  $env:HF_HUB_OFFLINE = "0"' -ForegroundColor Yellow
    Write-Host '  python -c "from faster_whisper import WhisperModel; WhisperModel(\"small\", device=\"cpu\", compute_type=\"int8\")"' -ForegroundColor Yellow
    Write-Host '  Remove-Item Env:HF_HUB_OFFLINE' -ForegroundColor Yellow
    Write-Host "Then re-run this script." -ForegroundColor Red
    exit 1
}

# Copy every file from the snapshot directory (resolving symlinks where
# present) into our flat destination.
Get-ChildItem -Path $srcSnapshot -File | ForEach-Object {
    $target = Join-Path $dst $_.Name
    # If the source is a symlink/junction, resolve to the underlying blob
    # and copy the actual content (not a broken link).
    $resolved = $_.FullName
    try {
        $item = Get-Item -LiteralPath $_.FullName
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            # Follow the link target manually.
            $linkTarget = (Get-Item -LiteralPath $_.FullName).Target
            if ($linkTarget) {
                # Target may be relative to the snapshot dir.
                $candidate = Join-Path (Split-Path $_.FullName) $linkTarget
                if (Test-Path $candidate) {
                    $resolved = (Resolve-Path $candidate).Path
                }
            }
        }
    } catch { }
    Copy-Item -Path $resolved -Destination $target -Force
    Write-Host "Copied: $($_.Name) ($([math]::Round((Get-Item $target).Length / 1MB, 1)) MB)" -ForegroundColor Cyan
}

# Sanity check
$modelBin = Join-Path $dst "model.bin"
if (-not (Test-Path $modelBin)) {
    Write-Host ""
    Write-Host "ERROR: Failed to stage model.bin into $dst." -ForegroundColor Red
    exit 1
}
$sizeMB = [math]::Round((Get-Item $modelBin).Length / 1MB, 0)
if ($sizeMB -lt 100) {
    Write-Host ""
    Write-Host "ERROR: model.bin is only $sizeMB MB, expected ~466 MB." -ForegroundColor Red
    Write-Host "Source snapshot may have been incomplete (broken symlinks?)." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Done. ./models/whisper-small/ contents:" -ForegroundColor Green
Get-ChildItem $dst | Format-Table Name, @{N='SizeMB';E={[math]::Round($_.Length/1MB,1)}} -AutoSize
Write-Host "Now run .\build.bat to build the .exe with the bundled model."
