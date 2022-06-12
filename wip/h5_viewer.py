from pathlib import Path
import holoviews as hv
from sklearn import preprocessing
import panel as pn
from holoviews.operation.datashader import regrid
import h5py
import numpy as np

# pn.extension(sizing_mode="stretch_both")

hv.extension("bokeh")


# path = Path(r"C2-contrast_enhanced_Probabilities.h5")
# path = Path(r"3pos/pos35/C2_enhanced_Probabilities.h5")
# path = Path(r"3pos/pos35/c3_segmented.h5")
path = Path(r"3pos/pos35/merged_segmented.h5")
# path = Path(r"3pos/pos35/merged_binary_map.h5")
# path = Path(r"3pos/pos35/c2_segmented.h5")
# path = Path(r"C:\Data\Code\MicroscopyPipeline\3pos\pos35\C2_enhanced_short-data_Object Predictions.h5")
assert path.exists()

stack = h5py.File(path, "r")
print(stack.keys())
stack = stack.get("exported_data")
# frames, _, _, _, channels = stack.shape
frames, _, _ = stack.shape

frame_wdgt = pn.widgets.IntSlider(
    name="Frame", start=0, end=frames - 1, step=1, value=0
)
# channel_wdgt = pn.widgets.IntSlider(
#     name="Channel", start=0, end=channels - 1, step=1, value=0
# )


def original(frame, channel=0):
    # img = stack[frame, 0, :, :, channel]
    img = stack[frame, ...]
    shape = img.shape
    # img = preprocessing.minmax_scale(img.ravel()).reshape(shape)

    # return pn.Column(
    #     hv.plotting.Image(img).opts(colorbar=False, cmap="gray"),
    #     hv.Histogram(np.histogram(img, 20)),
    # )
    # return hv.plotting.Image(img).opts(colorbar=False, cmap="gray")
    return hv.plotting.Image(img).opts(colorbar=False, cmap="gray")


# row = pn.Row(preprocess_image)

# dmap = hv.DynamicMap(original, kdims=["i", "c"])
# bounded_dmap = dmap.redim.values(i=list(range(frames)), c=list(range(channels)))

bounded_dmap_original = hv.DynamicMap(
    pn.bind(
        original,
        frame=frame_wdgt.param.value_throttled,
        # channel=channel_wdgt.param.value_throttled,
    )
).opts(responsive=True)


def build_viz(dmap):
    regridded = regrid(dmap)
    # regridded = rasterize(dmap)
    # histogram = regridded.hist(adjoin=False, normed=True)

    return pn.panel(regridded, sizing_mode="stretch_both")
    # return (regridded + histogram).cols(1)


pn.Row(
    pn.WidgetBox(
        frame_wdgt,
        #  channel_wdgt
    ),
    pn.Column(
        "Original",
        build_viz(bounded_dmap_original),
        sizing_mode="stretch_both",
    ),
    sizing_mode="stretch_both",
).show()
