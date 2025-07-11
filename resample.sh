#!/usr/bin/env bash
# -----------------------------------------------------------------
# Resample all registered NIfTI files (r_*.nii.gz) to 32³, 64³, or 128³ grid
# -----------------------------------------------------------------

set -euo pipefail

read -rp "Enter the directory containing r_*.nii.gz files: " dir
[[ -d "$dir" ]] || { echo "❌ Directory not found."; exit 1; }

# Ask user for desired resolution
read -rp "Choose output resolution (32, 64, 128): " res
[[ "$res" =~ ^(32|64|128)$ ]] || { echo "❌ Invalid resolution."; exit 2; }

mni_ref="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
[[ -r "$mni_ref" ]] || { echo "❌ MNI152 template not found in \$FSLDIR"; exit 3; }

echo "🔍 Looking for registered files in: $dir"
mapfile -t files < <(find "$dir" -type f -iname "r_*.nii.gz")
[[ ${#files[@]} -eq 0 ]] && { echo "⚠️  No registered r_*.nii.gz files found."; exit 0; }

# Create the resampling grid for the requested resolution
gridfile=$(mktemp --suffix=.nii.gz)
python3 - "$mni_ref" "$gridfile" "$res" <<'PY'
import sys, numpy as np, nibabel as nb
ref, out, res = sys.argv[1], sys.argv[2], int(sys.argv[3])
img = nb.load(ref)
shape = np.array(img.shape)
zooms = np.abs(img.header.get_zooms())
fov = shape * zooms
voxel = fov / res
aff = img.affine.copy()
aff[:3, :3] = np.diag(np.sign(np.diag(aff)[:3]) * voxel)
nb.Nifti1Image(np.zeros((res,res,res), dtype=np.uint8), aff).to_filename(out)
PY

# Loop through files and resample
for input in "${files[@]}"; do
  base=$(basename "$input")
  dirpath=$(dirname "$input")
  base_noext="${base%.nii.gz}"
  out_r="${dirpath}/r${res}_${base_noext#r_}.nii.gz"

  echo "📐 Resampling $base → $(basename "$out_r")"

  flirt -in "$input" \
        -ref "$gridfile" \
        -applyxfm \
        -init <(echo -e "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1") \
        -out "$out_r" \
        -interp trilinear
done

rm -f "$gridfile"
echo "✅ All files resampled to ${res}³ and saved with r${res}_ prefix."

