#!/bin/bash

# Path to the JSON file containing the list of required filenames
JSON_FILE="./annotations/thumos14.json"
# Path to the ZIP archive
ZIP_FILE="./videos/TH14_validation_set_mp4.zip"
# Google Cloud Storage bucket path
GCS_BUCKET="gs://daiy/validation"
# Temporary local extraction directory
TEMP_DIR="./temp_extracted"

# Ensure gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo "Error: gsutil not found. Install the Google Cloud SDK."
    exit 1
fi

# Ensure the temporary directory exists
mkdir -p "$TEMP_DIR"

# Read filenames from the JSON file (extracting keys from the 'database' object)
jq -r '.database | keys[]' "$JSON_FILE" | awk '{print "validation/" $1 ".mp4"}' > file_list.txt

# Loop through the filenames and extract them to a temp folder
while read -r FILE; do
    if unzip -l "$ZIP_FILE" | grep -q "$FILE"; then
        echo "Extracting $FILE..."
        unzip -o "$ZIP_FILE" "$FILE" -d "$TEMP_DIR"
        
        # Upload to GCS
        gsutil cp "$TEMP_DIR/$FILE" "$GCS_BUCKET/"
        echo "Uploaded $FILE to $GCS_BUCKET"

        # Delete local file to save space
        rm "$TEMP_DIR/$FILE"
    else
        echo "Skipping $FILE (not found in archive)"
    fi
done < file_list.txt

# Cleanup
rmdir "$TEMP_DIR"

echo "Extraction and upload complete."
