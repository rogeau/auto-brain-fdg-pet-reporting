#!/bin/bash

read -rp "Enter the directory containing .nii.gz files: " nifti_dir
if [[ ! -d "../$nifti_dir" ]]; then
  echo "Error: '$nifti_dir' is not a valid directory."
  exit 1
fi

echo "Checking for uncompressed .nii files..."

while IFS= read -r -d '' nii_file; do
  echo "Compressing '$nii_file' -> '${nii_file}.gz'"
  gzip -f "$nii_file"
done < <(find "../$nifti_dir" -type f -name "*.nii" ! -name "*.nii.gz" -print0)

declare -A rest_map

# Step 1: Rename files
while IFS= read -r -d '' file; do
  dir=$(dirname "$file")
  filename=$(basename "$file")
  base="${filename%.nii.gz}"

  if [[ "$base" =~ ^([^_]+)_([0-9]{14})_(.*)$ ]]; then
    id_part="${BASH_REMATCH[1]}"
    datetime="${BASH_REMATCH[2]}"
    rest="${BASH_REMATCH[3]}"

    # Extract first digit sequence (for short ID)
    digits_only_id=$(echo "$id_part" | grep -oE '[0-9]+' | head -n1)
    short_id="${digits_only_id:0:10}"
    short_date="${datetime:0:8}"
    newname="${short_id}_${short_date}_${rest}.nii.gz"

    newpath="${dir}/${newname}"

    if [[ "$file" != "$newpath" ]]; then
      echo "Renaming '$file' -> '$newpath'"
      mv -- "$file" "$newpath"
      file="$newpath"
    fi

    rest_map["$rest"]+="$file"$'\n'

  else
    echo "Skipping '$filename': does not match expected pattern."
  fi

done < <(find "../$nifti_dir" -type f -name "*.nii.gz" -print0)


# Step 2: Ask the user what to keep/delete
echo
echo "Review each 'rest' value below. Type 'y' to keep files, 'n' to delete."

for rest in "${!rest_map[@]}"; do
  echo
  echo -e "\033[1mFile type: $rest\033[0m"
  echo "Files:"
  echo "${rest_map[$rest]}"

  while true; do
    read -rp "Keep these files? (y/n): " choice
    case "$choice" in
      [Yy]*) 
        echo "Keeping files with rest: $rest"
        break
        ;;
      [Nn]*) 
        echo "Deleting files with rest: $rest"
        while IFS= read -r f; do
          if [[ -n "$f" ]]; then
            echo "Deleting '$f'"
            rm -- "$f"
          fi
        done <<< "${rest_map[$rest]}"
        break
        ;;
      *) echo "Please answer y (yes) or n (no)." ;;
    esac
  done
done
