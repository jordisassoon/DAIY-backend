#!/bin/bash

directory="./videos"  # Change this to your target directory
total_duration=0
count=0

for file in "$directory"/*.mp4; do
    if [[ -f "$file" ]]; then
        duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
        total_duration=$(echo "$total_duration + $duration" | bc)
        count=$((count + 1))
    fi
done

if [[ $count -eq 0 ]]; then
    echo "No MP4 files found in the directory."
    exit 1
fi

avg_duration=$(echo "$total_duration / $count" | bc)
minutes=$((avg_duration / 60))
seconds=$((avg_duration % 60))

echo "Average video length: $minutes minutes, $seconds seconds"