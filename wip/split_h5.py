from email.mime import base
from pathlib import Path
from sklearn import preprocessing
import h5py
import numpy as np
from tqdm import tqdm

base_path = Path(r"C:\Data\Code\MicroscopyPipeline\3pos\pos35")
assert base_path.exists()

paths = [path for path in base_path.rglob(r"*enhanced.h5") if "C1" not in str(path)]

dataset_name = "data"

clip_to_frame = 80

for path in tqdm(paths):
    stack = h5py.File(path, "r").get(dataset_name)
    frames, _, _, _, channels = stack.shape
    
    sections = list(range(0, frames, 30))
    for section in zip(sections, sections[1:]):
        clipped_stack = stack[section[0]:section[1], ...]

        new_file_path = path.parent / f"{path.stem}_half_{section}.h5"
        with h5py.File(new_file_path, "w") as f:
            f.create_dataset(
                dataset_name, data=clipped_stack, dtype=clipped_stack.dtype, chunks=True
            )

    clipped_stack = stack[sections[-1]:, ...]
    new_file_path = path.parent / f"{path.stem}_half_{sections[-1]}_to_end.h5"
    with h5py.File(new_file_path, "w") as f:
        f.create_dataset(
            dataset_name, data=clipped_stack, dtype=clipped_stack.dtype, chunks=True
        )