# sync.ps1 - Commit local changes and push to BOTH GitHub (origin) and the
#            HuggingFace Space (hf). Both stay in sync automatically.
#
# Usage:
#   .\sync.ps1 "commit message"    # stage+commit, push origin, push hf
#   .\sync.ps1 -PushOnly           # no commit, just push both remotes
#   .\sync.ps1 -Message "msg"      # same as positional arg
#
# Requirements:
#   - Git installed + authenticated for GitHub (origin)
#   - Authenticated for HF (hf remote). Run once:
#         huggingface_hub's `hf auth login` OR configure git credential
#         helper for huggingface.co (the `hf auth login --add-to-git-credential`
#         step when you first logged in handles this).

param(
    [string]$Message = "",
    [switch]$PushOnly
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

Write-Host "=== Sync repo at $root ===" -ForegroundColor Cyan

# Ensure on main
git -C $root checkout main

if (-not $PushOnly) {
    if (-not $Message) { $Message = "Update" }
    Write-Host "Staging and committing: $Message" -ForegroundColor Cyan
    git -C $root add -A
    git -C $root commit -m $Message
}

Write-Host "Pushing to GitHub (origin/main)..." -ForegroundColor Cyan
git -C $root push origin main
if ($LASTEXITCODE -ne 0) { Write-Error "GitHub push failed" }

Write-Host "Pushing to HuggingFace Space (hf/main)..." -ForegroundColor Cyan
git -C $root push hf main
if ($LASTEXITCODE -ne 0) { Write-Error "HF push failed" }

Write-Host "Done. Both GitHub and HF are in sync at:" -ForegroundColor Green
git -C $root rev-parse --short HEAD
