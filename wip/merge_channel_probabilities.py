from email.mime import base
from pathlib import Path
from sklearn import preprocessing
import h5py
import numpy as np
from tqdm import tqdm, trange

base_path = Path(r"C:\Data\Code\MicroscopyPipeline\3pos\pos35")
assert base_path.exists()

# paths = [path for path in base_path.rglob(r"*enhanced.h5") if "C1" not in str(path)]

red_channel_file = base_path / 'C2_enhanced_Probabilities.h5'
green_channel_file = base_path / 'C3_enhanced_Probabilities.h5'


dataset_name = "exported_data"

red_stack = h5py.File(red_channel_file, "r").get(dataset_name)
green_stack = h5py.File(green_channel_file, "r").get(dataset_name)


new_file_path = base_path / f"merged_probabilities.h5"
with h5py.File(new_file_path, "w") as f:
    dataset = f.create_dataset(
        dataset_name, red_stack.shape, dtype=red_stack.dtype, chunks=True
    )

    for frame in trange(red_stack.shape[0]):
        red_frame = np.where(red_stack[frame, ...] > 0.8, 1, 0)
        green_frame = np.where(green_stack[frame, ...] > 0.8, 1, 0)
        merged_frame = red_frame + green_frame

        dataset[frame] = np.clip(merged_frame, 0, 1)



    
