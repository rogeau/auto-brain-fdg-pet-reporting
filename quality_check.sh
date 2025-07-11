#!/bin/bash

read -rp "Enter folder path to search for .nii.gz files: " input_dir

if [ ! -d "$input_dir" ]; then
    echo "Error: Directory '$input_dir' does not exist."
    exit 1
fi

# Find all .nii.gz files recursively, sorted
mapfile -t nii_files < <(find "$input_dir" -type f -name "*.nii.gz" | sort)

if [ ${#nii_files[@]} -eq 0 ]; then
    echo "No .nii.gz files found in $input_dir"
    exit 0
fi

echo "Found ${#nii_files[@]} .nii.gz files."

for nii_file in "${nii_files[@]}"; do
    echo "Opening $nii_file in mricrogl..."

    # Launch mricrogl and wait for it to close before continuing
    mricrogl "$nii_file"

    # After mricrogl window closes, continue to next file
done

echo "All files processed."
