#!/bin/bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: bash hw4_2.sh <data_root> <output_dir>" >&2
    exit 1
fi

DATA_ROOT="$1"   # e.g. ./hw4_2_data
OUT_DIR="$2"     # e.g. ./p2_outputs

# Resolve paths without changing the working directory to comply with rules.
SCRIPT_DIR="$(python3 - <<'PY'
import os
print(os.path.dirname(os.path.abspath(__file__)))
PY
)"

python3 "${SCRIPT_DIR}/InstantSplat/render.py" \
    -s "$DATA_ROOT" \
    -m "${SCRIPT_DIR}/p2_ckpt" \
    -r 1 \
    --n_views 3 \
    --iterations 3000 \
    --eval

RENDER_SRC_DIR="${SCRIPT_DIR}/p2_ckpt/test/ours_3000/renders"

RENDER_SRC_DIR="$RENDER_SRC_DIR" OUT_DIR="$OUT_DIR" python3 - <<'PY'
import glob
import os
import shutil

src_dir = os.path.abspath(os.environ["RENDER_SRC_DIR"])
dst_dir = os.path.abspath(os.environ["OUT_DIR"])

os.makedirs(dst_dir, exist_ok=True)

pngs = glob.glob(os.path.join(src_dir, "*.png"))
if not pngs:
    print(f"Warning: No .png files found in {src_dir}. Check rendering output.")
else:
    for filename in pngs:
        shutil.copy(filename, dst_dir)
        print(f"Copied {filename} -> {dst_dir}")
PY

echo "Done. Test renders saved to: ${OUT_DIR}"
