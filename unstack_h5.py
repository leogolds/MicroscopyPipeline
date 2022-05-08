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
# dataset_name = "data"

for path in tqdm(paths):
    stack = h5py.File(path, "r")
    stack = stack.get(dataset_name)
    # TZYXC
    frames, _, _, _, channels = stack.shape

    new_path = path.parent / "unstacked"
    channel_paths = [new_path / f"C{i}" for i in range(channels)]

    new_path.mkdir(parents=True, exist_ok=True)
    [path.mkdir(parents=True, exist_ok=True) for path in channel_paths]

    for frame in tqdm(range(frames)):
        for channel in range(channels):
            clipped_stack = stack[frame, 0, :, :, channel]

            new_file_path = (
                channel_paths[channel]
                / f"{channel_paths[channel].parent.parent.name}_{channel_paths[channel].name}_{frame:05d}.tiff"
            )
            # with h5py.File(new_file_path, "w") as f:
            #     f.create_dataset(
            #         dataset_name,
            #         data=clipped_stack,
            #         dtype=clipped_stack.dtype,
            #         chunks=True,
            #     )
            # reshaped = reshape_data(clipped_stack, "TZYXC", "TCZYX")
            reshaped = clipped_stack
            OmeTiffWriter.save(
                reshaped,
                new_file_path,
                dim_order="YX",
            )
