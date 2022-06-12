import panel as pn
import param
from numpy import uint8
from pathlib import Path
import h5py
import holoviews as hv
from holoviews.operation.datashader import regrid
from typing import Callable
from skimage.exposure import equalize_adapthist
from utils import create_dmap_from_image
from ui import ContrastEnhancement

pn.extension(
    sizing_mode="stretch_width",
    notifications=True,
)

path = Path(r"3pos/pos35/")
assert path.exists()


if __name__ == "__main__":
    viewer = ContrastEnhancement()
    row = pn.Row(
        viewer.get_controls,
        viewer.get_main_window,
    )
    row.show()
