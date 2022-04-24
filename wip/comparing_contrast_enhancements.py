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
frames, channels, _, _, _ = stack.shape

frame_wdgt = pn.widgets.IntSlider(
    name="Frame", start=0, end=frames - 1, step=1, value=0
)
channel_wdgt = pn.widgets.IntSlider(
    name="Channel", start=0, end=channels - 1, step=1, value=0
)
phase_img = stack.get_image_dask_data("TYXC", Z=0).compute()


@pn.depends(i=frame_wdgt.param.value_throttled, c=channel_wdgt)
def rescale_intensity(i, c=0):
    img = phase_img[i, :, :, c]
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = skimage.exposure.rescale_intensity(img, in_range=(p2, p98))

    # # Equalization
    # img_eq = skimage.exposure.equalize_hist(img)

    # # Adaptive Equalization
    # img_adapteq = skimage.exposure.equalize_adapthist(img, clip_limit=0.03)

    # return [img, img_rescale, img_eq, img_adapteq]
    return hv.plotting.Image(img_rescale).opts(colorbar=False, cmap="gray")


@pn.depends(i=frame_wdgt.param.value_throttled, c=channel_wdgt)
def equalize(i, c=0):
    img = phase_img[i, :, :, c]

    # Equalization
    img_eq = skimage.exposure.equalize_hist(img)

    return hv.plotting.Image(img_eq).opts(colorbar=False, cmap="gray")


@pn.depends(i=frame_wdgt.param.value_throttled, c=channel_wdgt)
def adaptive_equalize(i, c=0):
    img = phase_img[i, :, :, c]

    # Adaptive Equalization
    img_adapteq = skimage.exposure.equalize_adapthist(img, clip_limit=0.03)

    return hv.plotting.Image(img_adapteq).opts(colorbar=False, cmap="gray")


@pn.depends(i=frame_wdgt.param.value_throttled, c=channel_wdgt)
def original(i, c=0):
    img = phase_img[i, :, :, c]

    return pn.Column(
        hv.plotting.Image(img).opts(colorbar=False, cmap="gray"),
        hv.Histogram(np.histogram(img, 20)),
    )


# row = pn.Row(preprocess_image)

pn.Column(
    frame_wdgt,
    channel_wdgt,
    pn.Row(
        # pn.Column("Original", hv.DynamicMap(original)),
        pn.Column("Original", original),
        # pn.Column("Rescaled Intensity", hv.DynamicMap(rescale_intensity)),
        # pn.Column("Rescaled Intensity", rescale_intensity),
        # # pn.Column("Equalize", hv.DynamicMap(equalize)),
        # pn.Column("Equalize", equalize),
        # # pn.Column("Adaptive Equalize", hv.DynamicMap(adaptive_equalize)),
        # pn.Column("Adaptive Equalize", adaptive_equalize),
    ),
).show()
