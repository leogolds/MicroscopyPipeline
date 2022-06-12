from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.transforms import reshape_data
import numpy as np
import skimage
from sklearn import preprocessing
import h5py
from tqdm import tqdm

path = Path(r"C:\Data\Code\MicroscopyPipeline\3pos")
assert path.exists()

tif_paths = list(path.rglob("*.tif"))


def adaptive_equalize(img_stack, c):
    img = img_stack[:, :, :, c]

    # Adaptive Equalization
    img_adapteq = [
        (
            skimage.exposure.equalize_adapthist(img[:, :, t], clip_limit=0.03) * 255
        ).astype(np.uint8)
        for t in range(img.shape[2])
    ]

    return np.stack(img_adapteq, 2)


for path in tqdm(tif_paths):
    stack = AICSImage(path)
    frames, channels, _, _, _ = stack.shape

    image_stack = stack.get_image_dask_data("XYTC", Z=0).compute()

    # stack = [adaptive_equalize(image_stack, c) for c in range(channels)]
    # contrast_enhanced_img = np.stack(stack, 3)
    contrast_enhanced_img = adaptive_equalize(image_stack, 0)
    # reshaped = reshape_data(contrast_enhanced_img, "XYTC", "TZXYC")
    reshaped = reshape_data(contrast_enhanced_img, "XYT", "TZYXC")

    new_file_path = path.parent / f"{path.stem}_enhanced.h5"
    with h5py.File(new_file_path, "w") as f:
        f.create_dataset("data", data=reshaped, dtype="uint8", chunks=True)
    # OmeTiffWriter.save(
    #     reshaped,
    #     "contrast_enhanced.tiff",
    #     dim_order="TCZYX",
    # )
