import os
import torch
import numpy as np

def convert_pt_to_npy(directory):
    for file in os.listdir(directory):
        if file.endswith(".pt"):
            pt_path = os.path.join(directory, file)
            npy_path = os.path.join(directory, file.replace(".pt", ".npy"))
            
            try:
                tensor = torch.load(pt_path, map_location=torch.device('cpu'))
                np.save(npy_path, tensor.numpy())
                print(f"Converted {file} to {npy_path}")
            except Exception as e:
                print(f"Error converting {file}: {e}")

# Example usage:
directory = "./internvideo2"  # Change this to your target directory
convert_pt_to_npy(directory)
