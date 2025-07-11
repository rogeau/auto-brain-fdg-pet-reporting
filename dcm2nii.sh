#!/bin/bash

echo "Available folders in dcm/:"
select folder_path in dcm/*/; do
    if [ -n "$folder_path" ]; then
        echo "You selected: $folder_path"
        break
    else
        echo "Invalid selection. Try again."
    fi
done

# Remove trailing slash
folder_path="${folder_path%/}"

# Extract only the folder name (basename)
folder_name=$(basename "$folder_path")

echo "Selected folder name: $folder_name"

mkdir -p "nifti/$folder_name"

dcm2niix -z y -b n -f "%i_%t_%d" -o "nifti/$folder_name" "dcm/$folder_name"
