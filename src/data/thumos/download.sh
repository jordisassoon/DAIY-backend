#!/bin/bash

# Define base URLs
BASE_URL_TEST="https://www.crcv.ucf.edu/THUMOS14/test_set/TH14_test_set_mp4/"

# Path to the JSON file
JSON_FILE="./annotations/thumos14.json"

# Directory to save downloaded videos
SAVE_DIR="./videos"

# Create the directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Extract filenames from the JSON file
video_files=$(jq -r '.database | keys[]' "$JSON_FILE")

# Loop through each video file
for video in $video_files; do
    FILE_NAME="${video}.mp4"
    FILE_PATH="${SAVE_DIR}/${FILE_NAME}"

    # Check if the file already exists
    if [ -f "$FILE_PATH" ]; then
        echo "Skipping $FILE_NAME (already exists)."
    else
        echo "Downloading $FILE_NAME..."
        wget --no-check-certificate -O "$FILE_PATH" "${BASE_URL_TEST}${FILE_NAME}"
    fi
done