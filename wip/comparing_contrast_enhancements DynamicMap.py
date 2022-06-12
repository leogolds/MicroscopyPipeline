from pathlib import Path
from aicsimageio import AICSImage
import holoviews as hv
import numpy as np
import skimage
from sklearn import preprocessing
import panel as pn
from holoviews.operation.datashader import regrid, rasterize


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


# @pn.depends(i=frame_wdgt.param.value_throttled, c=channel_wdgt)
def rescale_intensity(i, c=0):
    img = phase_img[i, :, :, c]
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = skimage.exposure.rescale_intensity(img, in_range=(p2, p98))
    shape = img.shape
    img_rescale = preprocessing.minmax_scale(img_rescale.ravel()).reshape(shape)

    # # Equalization
    # img_eq = skimage.exposure.equalize_hist(img)

    # # Adaptive Equalization
    # img_adapteq = skimage.exposure.equalize_adapthist(img, clip_limit=0.03)

    # return [img, img_rescale, img_eq, img_adapteq]
    return hv.plotting.Image(img_rescale).opts(colorbar=False, cmap="gray")


# @pn.depends(i=frame_wdgt.param.value_throttled, c=channel_wdgt)
def equalize(i, c=0):
    img = phase_img[i, :, :, c]

    # Equalization
    img_eq = skimage.exposure.equalize_hist(img)

    return hv.plotting.Image(img_eq).opts(colorbar=False, cmap="gray")


# @pn.depends(i=frame_wdgt.param.value_throttled, c=channel_wdgt)
def adaptive_equalize(i, c=0):
    img = phase_img[i, :, :, c]

    # Adaptive Equalization
    img_adapteq = skimage.exposure.equalize_adapthist(img, clip_limit=0.03)

    return hv.plotting.Image(img_adapteq).opts(colorbar=False, cmap="gray")


# @pn.depends(i=frame_wdgt.param.value_throttled, c=channel_wdgt)
def original(i, c=0):
    img = phase_img[i, :, :, c]
    shape = img.shape
    img = preprocessing.minmax_scale(img.ravel()).reshape(shape)

    # return pn.Column(
    #     hv.plotting.Image(img).opts(colorbar=False, cmap="gray"),
    #     hv.Histogram(np.histogram(img, 20)),
    # )
    return hv.plotting.Image(img).opts(colorbar=False, cmap="gray")


# row = pn.Row(preprocess_image)

# dmap = hv.DynamicMap(original, kdims=["i", "c"])
# bounded_dmap = dmap.redim.values(i=list(range(frames)), c=list(range(channels)))


bounded_dmap_original = hv.DynamicMap(
    pn.bind(
        original,
        i=frame_wdgt.param.value_throttled,
        c=channel_wdgt.param.value_throttled,
    )
)
bounded_dmap_rescale_intensity = hv.DynamicMap(
    pn.bind(
        rescale_intensity,
        i=frame_wdgt.param.value_throttled,
        c=channel_wdgt.param.value_throttled,
    )
)
bounded_dmap_equalize = hv.DynamicMap(
    pn.bind(
        equalize,
        i=frame_wdgt.param.value_throttled,
        c=channel_wdgt.param.value_throttled,
    )
)
bounded_dmap_adaptive_equalize = hv.DynamicMap(
    pn.bind(
        adaptive_equalize,
        i=frame_wdgt.param.value_throttled,
        c=channel_wdgt.param.value_throttled,
    )
)


def build_viz(dmap):
    regridded = regrid(dmap)
    # regridded = rasterize(dmap)
    histogram = regridded.hist(adjoin=False, normed=True)

    return (regridded + histogram).cols(1)


pn.Column(
    pn.Row(
        pn.WidgetBox(frame_wdgt, channel_wdgt),
        # pn.Column("Original", hv.DynamicMap(original)),
        pn.Column("Original", build_viz(bounded_dmap_original)),
        # # pn.Column("Rescaled Intensity", hv.DynamicMap(rescale_intensity)),
        pn.Column("Rescaled Intensity", build_viz(bounded_dmap_rescale_intensity)),
        # # # pn.Column("Equalize", hv.DynamicMap(equalize)),
        pn.Column("Equalize", build_viz(bounded_dmap_equalize)),
        # # # pn.Column("Adaptive Equalize", hv.DynamicMap(adaptive_equalize)),
        pn.Column("Adaptive Equalize", build_viz(bounded_dmap_adaptive_equalize)),
    ),
).show()
