from email.mime import base
from lib2to3.pytree import convert
from pathlib import Path
from sklearn import preprocessing
import h5py
import numpy as np
from tqdm import tqdm
from aicsimageio.transforms import reshape_data


base_path = Path(r"C:\Data\Code\MicroscopyPipeline\3pos\pos35")
assert base_path.exists()

paths = [
    path
    for path in base_path.rglob(r"*enhanced_Probabilities.h5")
    if "C1" not in str(path)
]

dataset_name = "exported_data"

for path in tqdm(paths):
    stack = h5py.File(path, "r")
    stack = stack.get(dataset_name)
    frames, _, _, _, channels = stack.shape

    converted_stack = (stack[..., 0] * 255).astype(np.uint8)
    converted_stack = reshape_data(converted_stack, "TZYX", "TZYXC")

    new_file_path = path.parent / f"{path.stem}_uint8.h5"
    with h5py.File(new_file_path, "w") as f:
        f.create_dataset(
            dataset_name, data=converted_stack, dtype=converted_stack.dtype, chunks=True
        )
