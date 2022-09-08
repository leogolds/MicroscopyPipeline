from email.mime import base
from pathlib import Path
from sklearn import preprocessing
import h5py
import numpy as np
from tqdm import tqdm

# base_path = Path(r"D:\Data\MicroscopyPipeline\ser1-1-20")
base_path = Path(r"D:\Data\MicroscopyPipeline\ser1")
assert base_path.exists()

# paths = [path for path in base_path.rglob(r"*enhanced.h5") if "C1" not in str(path)]
paths = [base_path / "red_contrast.h5", base_path / "green_contrast.h5"]

# dataset_name = "exported_data"
dataset_name = "data"

clip_to_frame = 20

for path in tqdm(paths):
    stack = h5py.File(path, "r")
    stack = stack.get(dataset_name)
    frames, _, _, _, channels = stack.shape
    # frames, _, _ = stack.shape

    clipped_stack = stack[:clip_to_frame, ...]

    new_file_path = path.parent / f"{path.stem}_short.h5"
    with h5py.File(new_file_path, "w") as f:
        f.create_dataset(
            dataset_name, data=clipped_stack, dtype=clipped_stack.dtype, chunks=True
        )
