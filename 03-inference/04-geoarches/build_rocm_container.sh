#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SIF_NAME="${1:-pytorch_training_geoarches.sif}"
DEF_NAME="geoarches_rocm.def"

echo "=== Building Apptainer image ==="
echo "DEF: $DEF_NAME"
echo "SIF: $SIF_NAME"

aexists=0
if [ -f "$SIF_NAME" ]; then
  echo "SIF already exists: $SIF_NAME (will rebuild anyway)"
fi

# Build requires Apptainer installed.
# If your cluster restricts builds, you may need to build on a login node.
apptainer build "$SIF_NAME" "$DEF_NAME"

echo "=== Done: $SIF_NAME ==="
