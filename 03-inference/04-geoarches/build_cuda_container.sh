#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SIF_NAME="${1:-pytorch_training_geoarches_nv.sif}"
DEF_NAME="geoarches_cuda.def"

echo "=== Building NVIDIA Apptainer image ==="
echo "DEF: $DEF_NAME"
echo "SIF: $SIF_NAME"

# Important: set a larger temp dir if /tmp is small.
# Example:
#   export APPTAINER_TMPDIR=/scratch/$USER/apptmp

apptainer build "$SIF_NAME" "$DEF_NAME"

echo "=== Done: $SIF_NAME ==="
