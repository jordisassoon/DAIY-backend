#!/bin/bash

TXT_FILE="annotations/Temporal_Anomaly_Annotation.txt"  # Change this to the actual txt filename
VIDEO_DIR="videos"        # Change this to the actual video directory

# Create a set of valid video files from the TXT file
declare -A valid_videos

while read -r line; do
    VIDEO_FILE=$(echo "$line" | awk '{print $1}')
    ACTION_CLASS=$(echo "$line" | awk '{print $2}')
    
    if [[ "$ACTION_CLASS" == "Normal" ]]; then
        # Delete "Normal" class videos
        FILE_PATH="$VIDEO_DIR/$VIDEO_FILE"
        if [[ -f "$FILE_PATH" ]]; then
            echo "Deleting $FILE_PATH (Normal class)"
            rm "$FILE_PATH"
        fi
    else
        valid_videos["$VIDEO_FILE"]=1
    fi
done < "$TXT_FILE"

# Delete videos that are not in the TXT file
for file in "$VIDEO_DIR"/*.mp4; do
    filename=$(basename "$file")
    if [[ ! ${valid_videos[$filename]} ]]; then
        echo "Deleting $file (Not in TXT file)"
        rm "$file"
    fi
done

echo "Cleanup complete."
