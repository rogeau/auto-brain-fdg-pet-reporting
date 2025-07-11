#!/usr/bin/env bash
# -----------------------------------------------------------------
# Resample all registered NIfTI files (r_*.nii.gz) to 32³, 64³, or 128³ using ANTs ResampleImage
# -----------------------------------------------------------------

set -euo pipefail

read -rp "Enter the directory containing r_*.nii.gz files: " dir
[[ -d "$dir" ]] || { echo "❌ Directory not found."; exit 1; }

read -rp "Choose output resolution (32, 64, 128): " res
[[ "$res" =~ ^(32|64|128)$ ]] || { echo "❌ Invalid resolution."; exit 2; }

echo "🔍 Looking for registered files in: $dir"
mapfile -t files < <(find "$dir" -type f -iname "r_*.nii.gz")
[[ ${#files[@]} -eq 0 ]] && { echo "⚠️  No registered r_*.nii.gz files found."; exit 0; }

# Loop and resample using ResampleImage
for input in "${files[@]}"; do
  base=$(basename "$input")
  dirpath=$(dirname "$input")
  base_noext="${base%.nii.gz}"
  out_r="${dirpath}/r${res}_${base_noext#r_}.nii.gz"

  echo "📐 Resampling $input to ${res}³"

  ResampleImage 3 "$input" "$out_r" ${res}x${res}x${res} 1 0
done

echo "✅ All files resampled to ${res}³ and saved with r${res}_ prefix."

