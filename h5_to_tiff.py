from email.mime import base
from pathlib import Path
from sklearn import preprocessing
import h5py
import numpy as np
from tqdm import tqdm
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.transforms import reshape_data

base_path = Path(r"C:\Data\Code\MicroscopyPipeline\3pos")
assert base_path.exists()

paths = [
    path
    for path in base_path.rglob(r"*enhanced_short_Probabilities.h5")
    if "C1" not in str(path)
]

dataset_name = "exported_data"

clip_to_frame = 15

for path in tqdm(paths):
    stack = h5py.File(path, "r")
    stack = stack.get(dataset_name)
    frames, _, _, _, channels = stack.shape

    new_file_path = path.parent / f"{path.stem}.tiff"

    reshaped = reshape_data(stack, "TZYXC", "TCZYX")
    OmeTiffWriter.save(
        reshaped,
        new_file_path,
        dim_order="TCZYX",
    )
