#!/usr/bin/env bash
set -euo pipefail

# Reproducible Linux setup for both project tasks:
#   1) sensitivity classification
#   2) ColPali retrieval / visual RAG
#
# This script never downloads a model checkpoint. Put the local checkpoint at
# COLPALI_MODEL_PATH before running GPU smoke tests.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${VLM_PYTHON:-python3}"
DATA_ROOT="${DOCVQA_DATA_ROOT:-$PROJECT_ROOT/data}"
MODEL_ROOT="${COLPALI_MODEL_PATH:-$PROJECT_ROOT/checkpoint}"
CACHE_ROOT="${VLM_CACHE_ROOT:-$PROJECT_ROOT/.cache/vlm}"
TORCH_INDEX_URL="${VLM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
NO_INSTALL=0

for argument in "$@"; do
    case "$argument" in
        --no-install) NO_INSTALL=1 ;;
        *)
            echo "Unknown argument: $argument"
            echo "Usage: bash setup.sh [--no-install]"
            exit 2
            ;;
    esac
done

export DOCVQA_DATA_ROOT="$DATA_ROOT"
export COLPALI_MODEL_PATH="$MODEL_ROOT"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export TMPDIR="${TMPDIR:-$CACHE_ROOT/tmp}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

echo "========================================================================"
echo "Project setup"
echo "  project:      $PROJECT_ROOT"
echo "  python:       $PYTHON_BIN"
echo "  data:         $DATA_ROOT"
echo "  base model:   $MODEL_ROOT"
echo "  cache:        $CACHE_ROOT"
echo "========================================================================"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $PYTHON_BIN"
    exit 1
fi
"$PYTHON_BIN" -c \
    'import sys; assert sys.version_info >= (3, 10), sys.version; print("Python", sys.version)'

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi is unavailable; GPU checks will run in validate_project.py"
fi

if [ "$NO_INSTALL" -eq 0 ]; then
    if "$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
    import torchvision
except Exception:
    raise SystemExit(1)
valid = torch.__version__.startswith("2.7.1") and torchvision.__version__.startswith("0.22.1")
raise SystemExit(0 if valid else 1)
PY
    then
        echo "Verified PyTorch/torchvision already installed; skipping reinstall."
    else
        echo "Installing verified PyTorch 2.7.1 / torchvision 0.22.1 from $TORCH_INDEX_URL"
        "$PYTHON_BIN" -m pip install \
            torch==2.7.1 torchvision==0.22.1 \
            --index-url "$TORCH_INDEX_URL"
    fi
    "$PYTHON_BIN" -m pip install -r requirements.txt
else
    echo "Dependency installation skipped (--no-install)."
fi

echo ""
echo "Data audit"
data_ok=1
for relative in \
    docvqa_extracted \
    docvqa_images \
    ocr \
    desensitized/docvqa_extracted \
    desensitized/docvqa_images \
    desensitized/ocr
do
    if [ ! -d "$DATA_ROOT/$relative" ]; then
        echo "  MISSING $DATA_ROOT/$relative"
        data_ok=0
    fi
done
if [ "$data_ok" -eq 1 ]; then
    full_count=$(find "$DATA_ROOT/docvqa_images" -maxdepth 1 -type f -name '*.png' | wc -l)
    positive_count=$(find "$DATA_ROOT/desensitized/docvqa_images" -maxdepth 1 -type f -name '*.png' | wc -l)
    echo "  full pages:      $full_count"
    echo "  positive pages:  $positive_count"
else
    echo "WARNING: incomplete data; upload the full data tree before building manifests."
fi

echo ""
echo "Checkpoint audit (no automatic download)"
if [ -f "$MODEL_ROOT/config.json" ] \
    && find "$MODEL_ROOT" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q .
then
    echo "  checkpoint present: $MODEL_ROOT"
else
    echo "WARNING: local checkpoint is incomplete: $MODEL_ROOT"
    echo "         Upload config/tokenizer/processor files, the index, and all shards."
fi

echo ""
echo "Running structural acceptance checks..."
if [ ! -f "$DATA_ROOT/manifests/sensitivity/summary.json" ]; then
    echo "Building missing sensitivity manifest..."
    "$PYTHON_BIN" scripts/build_sensitivity_manifest.py \
        --data-root "$DATA_ROOT"
fi
if [ ! -f "$DATA_ROOT/manifests/retrieval/summary.json" ]; then
    echo "Building missing retrieval manifest..."
    "$PYTHON_BIN" scripts/build_retrieval_manifest.py \
        --data-root "$DATA_ROOT"
fi
"$PYTHON_BIN" scripts/validate_project.py \
    --data-root "$DATA_ROOT" \
    --model "$MODEL_ROOT" \
    --skip-gpu-smoke \
    --output "$PROJECT_ROOT/outputs/setup_validation.json"

cat <<EOF

========================================================================
Setup finished. No checkpoint was downloaded.

Export these variables in each new shell (or put them in .env):
  export DOCVQA_DATA_ROOT="$DATA_ROOT"
  export COLPALI_MODEL_PATH="$MODEL_ROOT"
  export VLM_CACHE_ROOT="$CACHE_ROOT"

Build both leak-free manifests:
  $PYTHON_BIN scripts/build_sensitivity_manifest.py --data-root "$DATA_ROOT"
  $PYTHON_BIN scripts/build_retrieval_manifest.py --data-root "$DATA_ROOT"

Run complete project acceptance, including both real GPU forwards:
  $PYTHON_BIN scripts/validate_project.py --data-root "$DATA_ROOT" --model "$MODEL_ROOT"

Task A — sensitivity classification:
  $PYTHON_BIN scripts/smoke_sensitivity.py --data-root "$DATA_ROOT" --model "$MODEL_ROOT"
  $PYTHON_BIN scripts/train_sensitivity.py --config configs/sensitivity_head_5090.yaml --data-root "$DATA_ROOT" --model "$MODEL_ROOT"

Profile the actual 5090 before formal training:
  $PYTHON_BIN scripts/profile_training_memory.py --task sensitivity-unfreeze4 --data-root "$DATA_ROOT" --model "$MODEL_ROOT" --work-dir "$PROJECT_ROOT/outputs/profiles"
  $PYTHON_BIN scripts/profile_training_memory.py --task global --data-root "$DATA_ROOT" --model "$MODEL_ROOT" --work-dir "$PROJECT_ROOT/outputs/profiles"
  $PYTHON_BIN scripts/profile_training_memory.py --task late --data-root "$DATA_ROOT" --model "$MODEL_ROOT" --work-dir "$PROJECT_ROOT/outputs/profiles"

Task B — global then native multi-vector retrieval:
  $PYTHON_BIN scripts/train_retrieval_global.py --data-root "$DATA_ROOT" --model "$MODEL_ROOT"
  $PYTHON_BIN scripts/train_retrieval_late.py --data-root "$DATA_ROOT" --model "$MODEL_ROOT"
========================================================================
EOF
