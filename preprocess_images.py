from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.transforms import reshape_data
import numpy as np
import skimage
from sklearn import preprocessing


path = Path(r"C:\Users\Leo\Downloads\raw_data position 001_021.tif")
assert path.exists()

stack = AICSImage(path)
frames, channels, _, _, _ = stack.shape

phase_img = stack.get_image_dask_data("XYTC", Z=0).compute()


# @pn.depends(i=frame_wdgt.param.value_throttled, c=channel_wdgt)
def adaptive_equalize(c):
    img = phase_img[:, :, :, c]

    # Adaptive Equalization
    img_adapteq = [
        (
            skimage.exposure.equalize_adapthist(img[:, :, t], clip_limit=0.03) * 255
        ).astype(np.uint8)
        for t in range(img.shape[2])
    ]

    return np.stack(img_adapteq, 2)


stack = [adaptive_equalize(c) for c in range(channels)]
contrast_enhanced_img = np.stack(stack, 3)
reshaped = reshape_data(contrast_enhanced_img, "XYTC", "TCZYX")

OmeTiffWriter.save(
    reshaped,
    "contrast_enhanced.tiff",
    dim_order="TCZYX",
)
