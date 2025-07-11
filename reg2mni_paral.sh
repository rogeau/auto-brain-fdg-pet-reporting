#!/usr/bin/env bash
# -----------------------------------------------------------------
# Batch rigid registration of all NIfTI files in a user-specified
# folder and its subfolders using NiftyReg (reg_aladin) + GNU parallel
# -----------------------------------------------------------------

set -euo pipefail

read -rp "Enter the directory containing .nii.gz files: " dir
[[ -d "$dir" ]] || { echo "❌ Directory not found."; exit 1; }

mni_ref="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
[[ -r "$mni_ref" ]] || { echo "❌ MNI152 template not found in \$FSLDIR."; exit 2; }

export mni_ref  # Export so GNU parallel can access it

find "$dir" -type f -iname "*.nii.gz" ! -iname "r_*.nii.gz" | parallel --jobs 5 --bar --env mni_ref '
  input={}
  base=$(basename "$input")
  dirpath=$(dirname "$input")
  base_noext="${base%%.nii*}"
  out_r="$dirpath/r_${base_noext}.nii.gz"

  if [[ -f "$out_r" ]]; then
    echo "⏩ Skipping: $out_r"
  else
    echo "🔄 Registering: $input"
    reg_aladin -ref "$mni_ref" -flo "$input" -res "$out_r" -rigOnly -omp 4
  fi
'

echo "✅ All files registered."


