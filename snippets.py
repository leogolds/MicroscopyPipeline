import pandas as pd

import utils
from pathlib import Path
import tifffile
import h5py

# path = Path(r"D:\Data\ser1-1-20.tif")
# path = Path(r"D:\Data\ser2-1-20.tif")
# path = Path(r"D:\Data\ser1.tif")
# path = Path(r"D:\Data\MicroscopyPipeline\ser1-1-20\merged_probabilities.h5")

# a = utils.read_stack(path)

# utils.split_channels_and_convert_to_h5_aicsimageio(path)
# utils.split_channels_and_convert_to_h5_tifffile(path)
# a = tifffile.imread(path, aszarr=True)
# print(a.shape)
# utils.segment_h5_stack(path)


# path = Path(r"D:\Data\MicroscopyPipeline\ser1-1-20\segmented_table.h5")
# df = pd.read_hdf(path, key="table")
#
# vor = utils.compute_voronoi(df)
#
# print(df.head())
def naive_merge(left_path: Path, right_path: Path):
    left_stack = utils.read_stack(left_path)
    right_stack = utils.read_stack(right_path)

    new_file_path = left_path.parent / "naive_merge.h5"

    with h5py.File(new_file_path, "w") as f:
        f.create_dataset(
            "data",
            data=(left_stack[...] + right_stack[...]) / 2,
            dtype="uint8",
            chunks=True,
        )


# left_path = Path(r"D:\Data\MicroscopyPipeline\ser1-1-20\red_contrast.h5")
# right_path = Path(r"D:\Data\MicroscopyPipeline\ser1-1-20\green_contrast.h5")
# left_path = Path(r"D:\Data\MicroscopyPipeline\ser1\red_contrast.h5")
# right_path = Path(r"D:\Data\MicroscopyPipeline\ser1\green_contrast.h5")
# naive_merge(left_path, right_path)

path = Path(r"D:\Data\MicroscopyPipeline\ser1\naive\naive_merge.h5")
utils.segment_h5_stack(path)
