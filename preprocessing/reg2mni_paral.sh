#!/usr/bin/env bash
# -----------------------------------------------------------------
# Batch rigid registration of all NIfTI files in a user-specified
# folder and its subfolders using NiftyReg (reg_aladin) + GNU parallel
# -----------------------------------------------------------------

# set -euo pipefail

# read -rp "Enter the directory containing .nii.gz files: " dir
# [[ -d "../$dir" ]] || { echo "❌ Directory not found."; exit 1; }

# mni_ref="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
# [[ -r "$mni_ref" ]] || { echo "❌ MNI152 template not found in \$FSLDIR."; exit 2; }

# export mni_ref

# find "../$dir" -type f -iname "*.nii.gz" ! -iname "r*.nii.gz" | parallel --jobs 5 --bar --env mni_ref '
#   input={}
#   base=$(basename "$input")
#   dirpath=$(dirname "$input")
#   base_noext="${base%%.nii*}"
#   out_r="$dirpath/r_${base_noext}.nii.gz"

#   if [[ -f "$out_r" ]]; then
#     echo "⏩ Skipping: $out_r"
#   else
#     echo "🔄 Registering: $input"
#     reg_aladin -ref "$mni_ref" -flo "$input" -res "$out_r" -rigOnly -omp 4
#   fi
# '

# echo "✅ All files registered."


#!/bin/bash
set -euo pipefail

read -rp "Enter the directory containing .nii.gz files: " dir
[[ -d "../$dir" ]] || { echo "❌ Directory not found."; exit 1; }

mni_ref="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
[[ -r "$mni_ref" ]] || { echo "❌ MNI152 template not found in \$FSLDIR."; exit 2; }

export mni_ref

# Function to register if the file is missing or empty
register_if_failed() {
  input="$1"
  base=$(basename "$input")
  dirpath=$(dirname "$input")
  base_noext="${base%%.nii*}"
  out_r="$dirpath/r_${base_noext}.nii.gz"

  # Skip if the original file is empty
  if [[ ! -s "$input" ]]; then
    echo "❌ Input empty or missing: $input"
    return
  fi

  # Check if output is missing or empty
  if [[ ! -s "$out_r" ]]; then
    echo "🔄 Registering: $input"
    reg_aladin -ref "$mni_ref" -flo "$input" -res "$out_r" -rigOnly -omp 4
  else
    echo "✅ Already registered: $out_r"
  fi
}

export -f register_if_failed

find "../$dir" -type f -iname "*.nii.gz" ! -iname "r*.nii.gz" | \
  parallel --jobs 5 --bar register_if_failed {}

echo "✅ Registration complete."
