#!/usr/bin/env bash
# -----------------------------------------------------------------
# Batch rigid registration of all NIfTI files in a user-specified
# folder and its subfolders using FSL FLIRT.
# -----------------------------------------------------------------

set -euo pipefail

read -rp "Enter the directory containing .nii.gz files: " dir
[[ -d "$dir" ]] || { echo "❌ Directory not found."; exit 1; }

mni_ref="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
[[ -r "$mni_ref" ]] || { echo "❌ MNI152 template not found in \$FSLDIR."; exit 2; }

mapfile -t files < <(find "$dir" -type f -iname "*.nii.gz" ! -iname "r_*.nii.gz")
[[ ${#files[@]} -eq 0 ]] && { echo "⚠️  No NIfTI files found in subfolders."; exit 0; }

for input in "${files[@]}"; do
  base=$(basename "$input")
  dirpath=$(dirname "$input")
  base_noext="${base%%.nii*}"

  out_r="$dirpath/r_${base_noext}.nii.gz"
  
  if [[ -f "$out_r" ]]; then
    echo "⏩ Skipping (already registered): $out_r"
    continue
  fi

  echo "🔄 Registering: $input"

  # flirt -in "$input" \
  #       -ref "$mni_ref" \
  #       -dof 6 \
  #       -cost mutualinfo \
  #       -out "$out_r" \
  # 	-searchrx -20 20 -searchry -20 20 -searchrz -20 20 \
  # 	-interp trilinear \

  reg_aladin -ref "$mni_ref" \
            -flo "$input" \
            -res "$out_r" \
            -rigOnly
      
done

echo "✅ All files rigidly registered to MNI."
