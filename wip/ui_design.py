import panel as pn
import param
from pathlib import Path
import h5py
import holoviews as hv
from holoviews.operation.datashader import regrid
from typing import Callable
from skimage.exposure import equalize_adapthist

pn.extension(sizing_mode="stretch_width")
# pn.extension(sizing_mode="scale_both")


# class Preprocess(param.Parameterized):
#     clip_limit = param.Number(default=0.03, bounds=(0, 1), step=0.01)

#     def view(self):
#         return f"# Bla: {self.clip_limit:.4f}"
# pp = Preprocess()
# pn.Row(pp.param, pp.view).show()

path = Path(r"3pos/pos35/")
assert path.exists()


class H5Viewer(param.Parameterized):
    stacks = param.ObjectSelector()
    frame = param.Integer()
    clip_limit = param.Number(default=0.03, bounds=(0, 1), step=0.01)

    def __init__(self, **params):
        super().__init__(**params)

        # self.h5_files = list(path.glob("*.h5"))
        # self.file_selector = pn.widgets.Select(name="Stack", options=self.h5_files)
        self.param.stacks.objects = list(path.glob("*.h5"))

        self.stack = h5py.File(self.param.stacks.objects[0])
        self.stack = self.stack.get("exported_data", self.stack.get("data"))
        if len(self.stack.shape) == 5:
            frames, _, _, _, _ = self.stack.shape
            self.stack = self.stack[:, 0, ..., 0]
        else:
            frames, _, _ = self.stack.shape
        self.param.frame.bounds = (0, frames - 1)

        # self.template = pn.template.FastListTemplate(title="H5 Viewer")
        self.template = pn.template.VanillaTemplate(title="H5 Viewer")
        self.template.sidebar.extend(
            [
                pn.Param(self.param, widgets={"frame": pn.widgets.IntSlider}),
            ]
        )

        # self.template.main = [pn.Column(self.create_dmap, sizing_mode="stretch_width")]
        self.template.main.extend(
            [
                pn.Row(
                    pn.Column("# Original", create_dmap(self.original_image)),
                    pn.Column(
                        "# Contrast Enhanced", create_dmap(self.contrast_enhanced_image)
                    ),
                )
            ]
        )

    @param.depends("stacks", watch=True)
    def load_file(self):
        # print("frame loaded")
        self.stack = h5py.File(self.stacks)
        self.stack = self.stack.get("exported_data", self.stack.get("data"))

        if len(self.stack.shape) == 5:
            frames, _, _, _, _ = self.stack.shape
            self.stack = self.stack[:, 0, ..., 0]
        else:
            frames, _, _ = self.stack.shape
        self.frame = frames - 1 if frames - 1 < self.frame else self.frame
        self.param.frame.bounds = (0, frames - 1)
        # print(self.param.frame.bounds)

    #     self.template.main[:6, :6] = self.create_dmap

    # @param.depends("file_selector.value")
    # def load_image(self):
    #     print("Loading image")
    #     self.stack = h5py.File(self.file_selector.value)
    #     self.stack = self.stack.get("exported_data", self.stack.get("data"))
    #     self.frame.value = self.frame.value

    @param.depends("frame", watch=True)
    def original_image(self):
        # print("Refreshing image")
        img = hv.plotting.Image(self.stack[self.frame, ...]).opts(
            # img = hv.plotting.Image(self.stack[self.frame, 0, ..., 0]).opts(
            colorbar=False,
            cmap="gray",
        )
        return img

    @param.depends("frame", "clip_limit", watch=True)
    def contrast_enhanced_image(self):
        # print("Refreshing image")
        equalized = equalize_adapthist(
            self.stack[self.frame, ...], clip_limit=self.clip_limit
        )
        img = hv.plotting.Image(equalized * 255).opts(
            # img = hv.plotting.Image(self.stack[self.frame, 0, ..., 0]).opts(
            colorbar=False,
            cmap="gray",
        )
        return img

        # return pn.panel(regrid(dmap))

    def view(self):
        self.template.show()


def create_dmap(function: Callable):
    dmap = hv.DynamicMap(function).opts(
        responsive=True,
        aspect="equal",
    )

    regridded = regrid(dmap)
    histogram = regridded.hist(adjoin=False, normed=True).opts(
        responsive=True, width=125
    )

    # return (regridded + histogram).cols(1)
    return regridded << histogram

    # return regrid(dmap) + dmap.hist(adjoin=False, normed=True)


# def selected_hist(img, x_range, y_range):
#     # Apply current ranges
#     obj = img.select(x=x_range, y=y_range) if x_range and y_range else img

#     # Compute histogram
#     return hv.operation.histogram(obj)

# # Define a RangeXY stream linked to the image
# rangexy = hv.streams.RangeXY(source=img)

# # Adjoin the dynamic histogram computed based on the current ranges
# img << hv.DynamicMap(selected_hist, streams=[rangexy])

viewer = H5Viewer()
viewer.view()
