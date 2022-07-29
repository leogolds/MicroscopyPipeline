from pathlib import Path
import holoviews as hv
from sklearn import preprocessing
import panel as pn
from holoviews.operation.datashader import regrid
import h5py
import numpy as np
import param

# pn.extension(sizing_mode="stretch_both")

hv.extension("bokeh")

path = Path(r"D:\Data\MicroscopyPipeline\ser1")
assert path.exists()


class H5Viewer(param.Parameterized):
    def __init__(self, **params):
        super().__init__(**params)

        self.h5_files = list(path.glob("*.h5"))
        self.file_selector = pn.widgets.Select(name="Stack", options=self.h5_files)

        self.stack = h5py.File(self.file_selector.value)
        self.stack = self.stack.get("exported_data", self.stack.get("data"))

        if len(self.stack.shape) == 5:
            frames, _, _, _, _ = self.stack.shape
            self.stack = self.stack[:, 0, ..., 0]
        else:
            frames, _, _ = self.stack.shape
        self.frame = frame_wdgt = pn.widgets.IntSlider(
            name="Frame", start=0, end=frames - 1, step=1, value=0
        )

        self.template = pn.template.FastGridTemplate(title="H5 Viewer")
        self.template.sidebar.extend([self.frame, self.file_selector])
        # self.template.main[:, :] = pn.Column(
        #     self.create_dmap, sizing_mode="stretch_width"
        # )
        self.template.main[:6, :6] = self.create_dmap

    @param.depends("file_selector.value")
    def load_image(self):
        print("Loading image")
        self.stack = h5py.File(self.file_selector.value)
        self.stack = self.stack.get("exported_data", self.stack.get("data"))
        self.frame.value = self.frame.value

    @param.depends("frame.value_throttled")
    def create_image(self):
        print("Refreshing image")
        img = hv.plotting.Image(self.stack[self.frame.value_throttled, ...]).opts(
            colorbar=False, cmap="gray"
        )
        return img

    def create_dmap(self):
        dmap = hv.DynamicMap(self.create_image).opts(
            responsive=True, aspect="equal", min_height=600
        )

        return pn.panel(regrid(dmap))

    def view(self):
        self.template.show()


viewer = H5Viewer()
viewer.view()
