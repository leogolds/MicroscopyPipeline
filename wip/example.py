from pathlib import Path
from aicsimageio import AICSImage
import holoviews as hv
import numpy as np
import skimage
import panel as pn

hv.extension("bokeh")


path = Path(r"C:\Users\Leo\Downloads\raw_data position 001_021.tif")
assert path.exists()

stack = AICSImage(path)
phase_img = stack.get_image_dask_data("TYX", Z=0, C=0).compute()


def preprocess_image(i, c=0):
    # img = phase_img[i, :, :].copy()
    img = stack.get_image_dask_data("YX", T=i, Z=0, C=c).compute()
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = skimage.exposure.rescale_intensity(img, in_range=(p2, p98))

    # Equalization
    img_eq = skimage.exposure.equalize_hist(img)

    # Adaptive Equalization
    img_adapteq = skimage.exposure.equalize_adapthist(img, clip_limit=0.03)

    return [img, img_rescale, img_eq, img_adapteq]


processed = [
    hv.plotting.Image(i).opts(colorbar=True, cmap="gray") for i in preprocess_image(0)
]
row = pn.Row()
row.extend(processed)

processed = [
    hv.plotting.Image(i).opts(colorbar=True, cmap="gray")
    for i in preprocess_image(0, c=1)
]
row2 = pn.Row()
row2.extend(processed)

processed = [
    hv.plotting.Image(i).opts(colorbar=True, cmap="gray")
    for i in preprocess_image(0, c=2)
]
row3 = pn.Row()
row3.extend(processed)

pn.Column(row, row2, row3).show()
