from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.transforms import reshape_data
import numpy as np
import skimage
from sklearn import preprocessing
import h5py
from tqdm import tqdm

path = Path(r"C:\Data\Code\MicroscopyPipeline\3pos\pos35")
assert path.exists()

tif_paths = list(path.rglob("*.tif"))

for path in tqdm(tif_paths):
    stack = AICSImage(path)
    frames, channels, _, _, _ = stack.shape

    image_stack = stack.get_image_dask_data("TYX", Z=0, C=0).compute()

    reshaped = reshape_data(image_stack, "TYX", "TZYXC")

    new_file_path = path.parent / f"{path.stem}.h5"
    with h5py.File(new_file_path, "w") as f:
        f.create_dataset(
            "data", data=skimage.img_as_ubyte(reshaped), dtype="uint8", chunks=True
        )
    # OmeTiffWriter.save(
    #     reshaped,
    #     "contrast_enhanced.tiff",
    #     dim_order="TCZYX",
    # )
