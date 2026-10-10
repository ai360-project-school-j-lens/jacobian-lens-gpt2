#!/usr/bin/env bash
# Set up a rented GPU instance (e.g. vast.ai) for J-lens experiments.
#
# Clones this repository, installs the locked environment with uv, registers a
# Jupyter kernel, and downloads a model and its pre-fitted lens into HF_HOME.
# Safe to rerun: existing checkouts, environments and downloads are reused.
#
# Usage on the instance:
#   curl -LsSf https://raw.githubusercontent.com/ai360-project-school-j-lens/jacobian-lens-gpt2/main/scripts/vast_setup.sh | bash
#   # or, from a checkout:
#   BRANCH=qwen3.5-9B bash scripts/vast_setup.sh
#
# Requirements: NVIDIA driver supporting CUDA >= 13.0 (uv.lock pins torch built
# for CUDA 13.0), ~60 GB of disk for a 9B model, git and curl.
#
# Configuration (environment variables, defaults in brackets):
#   WORKDIR      where the repo and caches live           [/workspace]
#   REPO_URL     repository to clone                      [this fork, HTTPS]
#   BRANCH       branch to check out                      [main]
#   MODEL_ID     Hugging Face model to download           [Qwen/Qwen3.5-9B]
#   LENS_REPO    Hugging Face repo holding the lens       [bcywinski/jacobian-lens-qwen3.5-9b]
#   LENS_FILE    lens file inside LENS_REPO               [lens_n1000.pt]
#   FAST_KERNELS install flash-linear-attention (1/0)     [0]
#   HF_TOKEN     optional; raises Hugging Face rate limits
#
# Cached experiment results (runs/, git-ignored) are not part of the repo. To
# reuse them, copy them from your machine after this script finishes:
#   scp -r runs/<run-name> vast:"$WORKDIR"/jacobian-lens-gpt2/runs/

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/ai360-project-school-j-lens/jacobian-lens-gpt2.git}"
BRANCH="${BRANCH:-main}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
LENS_REPO="${LENS_REPO:-bcywinski/jacobian-lens-qwen3.5-9b}"
LENS_FILE="${LENS_FILE:-lens_n1000.pt}"
FAST_KERNELS="${FAST_KERNELS:-0}"
REPO_DIR="$WORKDIR/jacobian-lens-gpt2"
export HF_HOME="${HF_HOME:-$WORKDIR/hf}"

step() { printf '\n==> %s\n' "$*"; }

step "GPU and driver"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
df -h "$WORKDIR" | tail -1

step "uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
fi
uv --version

step "Repository ($BRANCH)"
mkdir -p "$WORKDIR"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
git log --oneline -1

step "Python environment"
uv sync --extra dev
if [ "$FAST_KERNELS" = "1" ]; then
    # Fast Gated DeltaNet kernels for Qwen3.5; optional for inference.
    # Not in pyproject.toml, so a later `uv sync` removes it again.
    uv pip install flash-linear-attention
fi
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name())"

step "Jupyter kernel 'jlens (uv)'"
.venv/bin/python -m ipykernel install --user --name jlens \
    --display-name "jlens (uv)" --env HF_HOME "$HF_HOME"

step "HF_HOME=$HF_HOME"
grep -q "export HF_HOME=" "$HOME/.bashrc" 2>/dev/null \
    || echo "export HF_HOME=$HF_HOME" >> "$HOME/.bashrc"

step "Download $MODEL_ID"
.venv/bin/hf download "$MODEL_ID"

step "Download $LENS_REPO/$LENS_FILE"
.venv/bin/hf download "$LENS_REPO" "$LENS_FILE"

step "Done"
du -sh "$HF_HOME" "$REPO_DIR/.venv"
cat <<EOF

Next:
  cd $REPO_DIR && source .venv/bin/activate
  jupyter lab --no-browser --port 8888      # then: ssh -L 8888:localhost:8888 ...
or open $REPO_DIR in VS Code Remote-SSH and pick the kernel "jlens (uv)".
EOF
