import json
import argparse
import random
import cv2
import os

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else None
    cap.release()
    return fps, duration

def process_txt_to_json(input_file, output_file, video_dir):
    database = {}
    label_mapping = {}
    label_counter = 0
    
    with open(input_file, "r") as file:
        lines = [line.strip() for line in file if line.strip()]
    
    filtered_lines = [line for line in lines if " Normal " not in line]  # Skip "Normal" class
    random.shuffle(filtered_lines)  # Shuffle for split
    
    split_index = int(len(filtered_lines) * 0.8)  # 4:1 ratio (80% Validation, 20% Test)
    val_videos = set(filtered_lines[:split_index])
    test_videos = set(filtered_lines[split_index:])
    
    for line in filtered_lines:
        parts = line.split()
        if len(parts) < 4:
            continue  # Skip invalid lines
        
        video_name = parts[0]
        label = parts[1]
        subset = "Validation" if line in val_videos else "Test"
        
        video_path = os.path.join(video_dir, video_name)
        fps, duration = get_video_properties(video_path)
        if fps is None or duration is None:
            print(f"Skipping {video_name}: Video file not found or invalid.")
            continue
        
        if label not in label_mapping:
            label_mapping[label] = label_counter
            label_counter += 1
        
        label_id = label_mapping[label]
        segments = []
        
        # Process segments (frames to seconds conversion)
        for i in range(2, len(parts), 2):
            if parts[i] == "-1" or parts[i+1] == "-1":
                continue
            t_start = int(parts[i]) / fps
            t_end = int(parts[i+1]) / fps
            segments.append({
                "label": label,
                "label_id": label_id,
                "segment": [t_start, t_end]
            })
        
        if video_name not in database:
            database[video_name] = {
                "subset": subset,
                "duration": duration,
                "fps": fps,
                "annotations": []
            }
        
        database[video_name]["annotations"].extend(segments)
    
    output_json = {"version": "UCF", "database": database}
    
    with open(output_file, "w") as json_file:
        json.dump(output_json, json_file, indent=4)
    
    print(f"JSON saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input .txt file")
    parser.add_argument("output_file", help="Path to the output .json file")
    parser.add_argument("video_dir", help="Path to the directory containing video files")
    
    args = parser.parse_args()
    process_txt_to_json(args.input_file, args.output_file, args.video_dir)