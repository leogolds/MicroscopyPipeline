from pathlib import Path
import holoviews as hv
from sklearn import preprocessing
import panel as pn
from holoviews.operation.datashader import regrid
import h5py
import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import hvplot.pandas

hv.extension("bokeh")

base_path = Path(r"C:\Data\Code\MicroscopyPipeline\3pos\pos35")
c2_path = Path(r"3pos\pos35\C2_enhanced_Probabilities_uint8.h5")
c3_path = Path(r"3pos\pos35\C3_enhanced_Probabilities_uint8.h5")
merged_path = Path(r"3pos\pos35\merged_binary_map.h5")

c2 = h5py.File(c2_path).get("exported_data")[:, 0, ..., 0]
c3 = h5py.File(c3_path).get("exported_data")[:, 0, ..., 0]
merged = h5py.File(merged_path).get("exported_data")
print([a.shape for a in (c2, c3, merged)])

# left = hv.Image(c2[0, ...]).opts(cmap="gray") * hv.Image(c3[0, ...]).opts(cmap="kg")
left = hv.RGB(np.dstack([c2[0, ...], c3[0, ...], np.zeros((2048, 2048))]))
merged = hv.Image(merged[0, ...]).opts(cmap="gray")
pn.Row(left, merged).show()

# bootstrap = pn.template.MaterialTemplate(title="Side-by-side")
# bootstrap.sidebar.extend([frame_wdgt, statistic_widget, quantiles_widget])

# bootstrap.main.extend([bounded_dmap_vor_tiling, bounded_dmap_kymograph])

# bootstrap.show()


# pn.Column(
#     pn.WidgetBox(frame_wdgt, statistic_widget, quantiles_widget),
#     pn.Column(bounded_dmap_vor_tiling, bounded_dmap_kymograph,
#     ),
# ).show()
