from email.mime import base
from pathlib import Path
from sklearn import preprocessing
import h5py
import numpy as np
from tqdm import tqdm, trange
import tifffile

# base_path = Path(r"D:\Data\MicroscopyPipeline\ser1-1-20")
base_path = Path(r"D:\Data\MicroscopyPipeline\ser1")
assert base_path.exists()

# paths = [path for path in base_path.rglob(r"*enhanced.h5") if "C1" not in str(path)]
paths = [
    base_path / "red_contrast_short_cropped.tif",
    base_path / "green_contrast_short_cropped.tif",
]

# dataset_name = "exported_data"
dataset_name = "data"

clip_to_frame = 20

for path in tqdm(paths):
    split_path = base_path / "split" / f"{path.stem}_split"
    split_path.mkdir(parents=True, exist_ok=True)
    # stack = h5py.File(path, "r")
    # stack = stack.get(dataset_name)
    stack = tifffile.imread(path)
    # frames, _, _, _, channels = stack.shape
    frames, _, _ = stack.shape

    clipped_stack = stack[:clip_to_frame, ...]

    for frame in trange(clipped_stack.shape[0]):
        tifffile.imwrite(
            split_path / f"{path.stem}_{frame:04d}.tif",
            clipped_stack[0, :],
            metadata={"axes": "ZYXC"},
        )
