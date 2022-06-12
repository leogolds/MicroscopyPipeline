import numpy as np

from pathlib import Path
import h5py
from cellpose import models
from PIL import Image
import holoviews as hv

hv.extension("bokeh")
from scipy import ndimage
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
import hvplot.pandas
from tqdm import trange


magnification_towards_camera = 1
pixel_size_in_microns = 0.65 * magnification_towards_camera
calibration_squared_microns_to_squared_pixel = pixel_size_in_microns**2
typical_cell_area_microns = 400


def compute_voronoi(df):
    # print(df.head())
    vor = Voronoi(df[["center_x", "center_y"]])
    # = pd.DataFrame()

    df["vertice_ids"] = [vor.regions[i] for i in vor.point_region]
    df["valid_region"] = [True if min(l) != -1 else False for l in df.vertice_ids]
    df["vertices"] = [
        np.array([vor.vertices[vertice_id] for vertice_id in vertice_ids])
        for vertice_ids in df.vertice_ids
    ]

    return df


def compute_voronoi_stats(df):
    df["area"] = [
        Polygon(vert).area * calibration_squared_microns_to_squared_pixel
        for vert in df.vertices
    ]
    df["perimeter"] = [
        Polygon(vert).length * pixel_size_in_microns for vert in df.vertices
    ]

    horizontal_bins = range(0, 2080, 20)
    df["bins"] = pd.cut(
        df.center_x, bins=horizontal_bins, labels=range(len(horizontal_bins) - 1)
    )

    return df


def calculate_kymograph(df):
    return df.agg({"area": ["mean", "std"], "perimeter": ["mean", "std"]})


base_path = Path(r"C:\Data\Code\MicroscopyPipeline\3pos\pos35")
path = base_path / "C2_enhanced_Probabilities.h5"
# print(paths)
assert path.exists()


stack = h5py.File(path).get("exported_data")
frames, _, _, _, _ = stack.shape

df = pd.DataFrame(columns=["frame", "center_y", "center_x"])
# for frame in trange(frames):
# img = stack[frame, 0, :, :, 0]
img = stack[:30, 0, :, :, 0]

model_type = "nuclei"
diameter = None
channels = [0, 0]
net_avg = False
resample = False

model = models.Cellpose(
    gpu=False, model_type=model_type
)  # model_type can be 'cyto' or 'nuclei'
masks, flows, styles, diams = model.eval(
    img,
    diameter=diameter,
    channels=channels,
    net_avg=net_avg,
    resample=resample,
    z_axis=0,
)

center_of_mass = ndimage.center_of_mass(masks, labels=masks, index=np.unique(masks)[1:])

sub_df = pd.DataFrame(
    center_of_mass,
    columns=["center_y", "center_x"],
)
sub_df["timestep"] = frame

df = pd.concat([df, sub_df], ignore_index=True)

df.to_csv("bla.csv")

# valid_regions_df = df.groupby("timestep").apply(compute_voronoi).query("valid_region")
# vor_stats_df = valid_regions_df.groupby("timestep").apply(compute_voronoi_stats)

# kymograph_df = (
#     vor_stats_df.query("area.quantile(.05) < area < area.quantile(.95)")
#     .groupby(["timestep", "bins"])
#     .apply(calculate_kymograph)
# )
# kymograph_df.index.rename(names=["timestep", "bins", "statistic"], inplace=True)

# pivot = (
#     kymograph_df.reset_index()
#     .query('statistic == "mean"')
#     .pivot(index="timestep", columns="bins", values="area")
# )
# pivot.sort_index(axis="columns").hvplot.heatmap(flip_yaxis=True).opts(cmap="bky")


# frame = 0

# phase = h5py.File(base_path / "C1_enhanced.h5").get("data")[frame, 0, :, :, 0]
# flourescence = h5py.File(base_path / "C2_enhanced.h5").get("data")[frame, 0, :, :, 0]

# cells_df = df.query(f"timestep == {frame}").copy()
# cells_df.center_x = cells_df.center_x / 2048 - 0.5
# cells_df.center_y = -1 * (cells_df.center_y / 2048 - 0.5)

# center_points = hv.Image(flourescence).opts(cmap="gray") * cells_df.hvplot.scatter(
#     x="center_x", y="center_y"
# )

# v_df = df.query(f"timestep == {frame}").copy()

# v_df.center_x = v_df.center_x / 2048 - 0.5
# v_df.center_y = -1 * (v_df.center_y / 2048 - 0.5)

# v_df = v_df.groupby("timestep").apply(compute_voronoi).query("valid_region")
# v_df = v_df.groupby("timestep").apply(compute_voronoi_stats)

# # v_df.query('timestep == 0 and area.quantile(.05) < area < area.quantile(.95)')

# voronoi_tiling = hv.Image(phase).opts(cmap="gray") * hv.Polygons(
#     [
#         {("x", "y"): vert, "level": Polygon(vert).area}
#         for vert in v_df.query(
#             "area.quantile(.01) < area < area.quantile(.95)"
#         ).vertices
#     ],
#     vdims="level",
# ).opts(alpha=0.4, cmap="bky")

# center_points + voronoi_tiling
