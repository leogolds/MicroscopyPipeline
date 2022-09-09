from datetime import datetime
from time import sleep
import panel as pn
import param
import numpy as np
from numpy import uint8
from pathlib import Path
import h5py
import holoviews as hv
from holoviews.operation.datashader import regrid
from typing import Callable
from skimage.exposure import equalize_adapthist
from utils import create_dmap_from_image, docker_client, segment_frame
import pandas as pd
from wip.ui import SegmentStack

pn.extension(
    sizing_mode="stretch_width",
    notifications=True,
)

path = Path(r"3pos/pos35/")
assert path.exists()


if __name__ == "__main__":
    viewer = SegmentStack()
    row = pn.Row(
        viewer.get_controls,
        viewer.get_main_window,
    )
    row.show()
