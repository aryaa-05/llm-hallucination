# sync.ps1 - Push local changes to BOTH GitHub and the HuggingFace Space.
#
# Usage:
#   .\sync.ps1 "commit message"        # commit + push to GitHub + upload to HF
#   .\sync.ps1                         # push + upload only (no new commit)
#   .\sync.ps1 -DeployHfOnly           # upload to HF only (skip GitHub)
#
# Requirements:
#   - Git installed, authenticated for GitHub (git push)
#   - HF CLI (venv\Scripts\hf.exe) authenticated (hf auth login)

param(
    [string]$Message = "",
    [switch]$DeployHfOnly
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$hfCli = Join-Path $root "venv\Scripts\hf.exe"
$space = "aryaa-05/llm-hallucination"

Write-Host "=== Sync: $root ===" -ForegroundColor Cyan

# 0) Fresh venv hf CLI?
if (-not (Test-Path $hfCli)) {
    Write-Warning "HF CLI not found at $hfCli"
    $hfCli = "hf"
}

# 1) Optionally commit
if ($Message) {
    git -C $root add -A
    if (-not $Message) { $Message = "Update" }
    git -C $root commit -m $Message
}

# 2) Push to GitHub (skip if not configured / DeployHfOnly)
if (-not $DeployHfOnly) {
    $ghRemote = git -C $root remote
    if ($ghRemote -match "origin") {
        Write-Host "Pushing to GitHub (origin/main)..." -ForegroundColor Cyan
        git -C $root push origin main
    } else {
        Write-Warning "No 'origin' remote configured; skipping GitHub push. Add one with: git remote add origin <url>"
    }
} else {
    Write-Host "Skipping GitHub push (DeployHfOnly)." -ForegroundColor Yellow
}

# 3) Upload to HF Space (file-level upload, triggers rebuild)
Write-Host "Uploading to HF Space $space ..." -ForegroundColor Cyan
& $hfCli upload $space $root "." --repo-type space `
    --exclude ".git/*" `
    --exclude "venv/*" `
    --exclude "__pycache__/*" `
    --exclude "*.pyc" `
    --exclude "*.log" `
    --exclude ".env" `
    --exclude ".env.*" `
    --commit-message "sync: $Message from local"

if ($LASTEXITCODE -ne 0) {
    Write-Error "HF upload failed"
}
Write-Host "Done. HF Space will rebuild." -ForegroundColor Green
