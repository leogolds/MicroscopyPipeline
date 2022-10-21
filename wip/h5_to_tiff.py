from email.mime import base
from pathlib import Path
from sklearn import preprocessing
import h5py
import numpy as np
from tqdm import tqdm
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.transforms import reshape_data
import tifffile

base_path = Path(r"D:\Data\MicroscopyPipeline\ser1_medium")
# base_path = Path(r"D:\Data\MicroscopyPipeline\ser1-1-20")
assert base_path.exists()

# paths = [base_path / "segmented.h5"]
paths = [base_path / "red_segmented.h5", base_path / "green_segmented.h5"]

dataset_name = "data"

clip_to_frame = 15

for path in tqdm(paths):
    stack = h5py.File(path, "r")
    stack = stack.get(dataset_name)
    # frames, _, _, _, channels = stack.shape
    frames, _, _ = stack.shape

    new_file_path = path.parent / f"{path.stem}.tiff"

    # reshaped = reshape_data(stack, "TZYXC", "TCZYX")
    reshaped = reshape_data(stack, "TYX", "TCZYX")
    with tifffile.TiffWriter(new_file_path, bigtiff=True) as tif:
        tif.write(reshaped, shape=reshaped.shape)
    # OmeTiffWriter.save(
    #     reshaped,
    #     new_file_path,
    #     dimension_order="TCZYX",
    # )
