#!/bin/bash

# INPUT_ROOT=$1

# find "$INPUT_ROOT" -type d | while read -r SUBDIR; do
#     if compgen -G "$SUBDIR/*.dcm" > /dev/null || compgen -G "$SUBDIR/*.v" > /dev/null; then
#         echo "Converting: $SUBDIR"
#         dcm2niix -z y -b n -f 4dnifti "$SUBDIR"
#         find "$SUBDIR" -type f \( -name "*.dcm" -o -name "*.v" \) -delete
#         mcflirt -in "$SUBDIR"/4dnifti.nii.gz -meanvol -spline_final -out "$SUBDIR"/realigned_4d.nii.gz
#         fslmaths "$SUBDIR"/realigned_4d.nii.gz -Tmean "$SUBDIR"/mean_volume.nii.gz
#         rm "$SUBDIR"/4dnifti.nii.gz "$SUBDIR"/realigned_4d.nii.gz_mean_reg.nii.gz "$SUBDIR"/realigned_4d.nii.gz
#     fi
# done

#!/bin/bash

INPUT_ROOT=$1
LOG_FILE="failed_log.txt"
> "$LOG_FILE"  # Empty the log file at the beginning

find "$INPUT_ROOT" -type d | while read -r SUBDIR; do
    if compgen -G "$SUBDIR"/*.dcm > /dev/null || compgen -G "$SUBDIR"/*.v > /dev/null; then
        echo "Processing: $SUBDIR"

        dcm2niix -z y -b n -f 4dnifti "$SUBDIR" || {
            echo "[dcm2niix FAILED] $SUBDIR" >> "$LOG_FILE"
            continue
        }

        find "$SUBDIR" -type f \( -name "*.dcm" -o -name "*.v" \) -delete || {
            echo "[DELETE FAILED] $SUBDIR" >> "$LOG_FILE"
            continue
        }

        mcflirt -in "$SUBDIR"/4dnifti.nii.gz -meanvol -spline_final -out "$SUBDIR"/realigned_4d.nii.gz || {
            echo "[MCFLIRT FAILED] $SUBDIR" >> "$LOG_FILE"
            continue
        }

        fslmaths "$SUBDIR"/realigned_4d.nii.gz -Tmean "$SUBDIR"/mean_volume.nii.gz || {
            echo "[FSLMATHS FAILED] $SUBDIR" >> "$LOG_FILE"
            continue
        }

        rm "$SUBDIR"/4dnifti.nii.gz "$SUBDIR"/realigned_4d.nii.gz_mean_reg.nii.gz "$SUBDIR"/realigned_4d.nii.gz || {
            echo "[CLEANUP FAILED] $SUBDIR" >> "$LOG_FILE"
            continue
        }

        echo "✓ Completed: $SUBDIR"
    fi
done

