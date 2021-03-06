from typing import Callable
import numpy as np

from pathlib import Path
import h5py
from cellpose import models
from PIL import Image
import holoviews as hv
from docker.client import DockerClient

hv.extension("bokeh")
from scipy import ndimage
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
import hvplot.pandas
from tqdm import trange
from holoviews.operation.datashader import regrid


magnification_towards_camera = 1
# pixel_size_in_microns = 0.65 * magnification_towards_camera
pixel_size_in_microns = 1 * magnification_towards_camera
calibration_squared_microns_to_squared_pixel = pixel_size_in_microns**2
typical_cell_area_microns = 400


def _compute_voronoi(df):
    vor = Voronoi(df[["center_x", "center_y"]])

    # Unpack vertice coordinates from Voronoi object for every nuclei center in df
    df["vertice_ids"] = [vor.regions[i] for i in vor.point_region]
    df["valid_region"] = [True if min(l) != -1 else False for l in df.vertice_ids]
    df["vertices"] = [
        np.array([vor.vertices[vertice_id] for vertice_id in vertice_ids])
        for vertice_ids in df.vertice_ids
    ]

    return df


def _compute_voronoi_stats(df):
    # Given the vertice coordinates of the Voronoi region encompassing each nuclei, calculate area & perimeter
    df["area"] = [
        Polygon(vert).area * calibration_squared_microns_to_squared_pixel
        for vert in df.vertices
    ]
    df["perimeter"] = [
        Polygon(vert).length * pixel_size_in_microns for vert in df.vertices
    ]

    # Assign each nuclei center a bin
    # Each column of ~20 pixels falls in one of 100 bins
    horizontal_bins = np.linspace(start=0, stop=2048, num=100)
    # horizontal_bins = range(0, 2080, 20)
    df["bins"] = pd.cut(
        df.center_x, bins=horizontal_bins, labels=range(len(horizontal_bins) - 1)
    )

    return df


def compute_voronoi(df):
    valid_regions_df = (
        df.groupby("timestep").apply(_compute_voronoi).query("valid_region")
    )
    vor_stats_df = valid_regions_df.groupby("timestep").apply(_compute_voronoi_stats)

    return vor_stats_df


def calculate_kymograph(df):
    return df.agg({"area": ["mean", "std"], "perimeter": ["mean", "std"]})


def read_stack(path) -> np.ndarray:
    f = h5py.File(path)
    return f.get("data", f.get("exported_data"))


def segment_h5_stack(path):
    stack = read_stack(path)

    frames, Y, X = stack.shape

    # df = pd.DataFrame(columns=["center_y", "center_x"])

    new_file_path = path.parent / f"{path.stem}_segmented.h5"
    new_table_path = path.parent / f"{path.stem}_table.h5"
    dataset_name = "data"

    with h5py.File(new_file_path, "w") as f:
        dataset = f.create_dataset(
            dataset_name, shape=(frames, Y, X), dtype=np.uint16, chunks=True
        )

        for frame in trange(frames):
            # img = stack[frame, 0, :, :, 0]
            masks, center_of_mass = segment_frame(stack[frame, ...])

            dataset[frame, :, :] = masks

            sub_df = pd.DataFrame(
                center_of_mass,
                columns=["center_y", "center_x"],
            )
            sub_df["timestep"] = frame

            # df = pd.concat([df, sub_df], ignore_index=True)
            sub_df.to_hdf(
                new_table_path, key="table", format="table", append=True, mode="a"
            )

    return new_file_path


def segment_frame(img, gpu: bool = False, diameter: int = 25):
    model_type = "nuclei"
    # diameter = None
    channels = [0, 0]
    net_avg = False
    resample = False

    model = models.Cellpose(
        gpu=gpu, model_type=model_type
    )  # model_type can be 'cyto' or 'nuclei'
    masks, flows, styles, diams = model.eval(
        img.astype(np.float16, copy=False),
        diameter=diameter,
        channels=channels,
        net_avg=net_avg,
        resample=resample,
        z_axis=0,
    )

    center_of_mass = ndimage.center_of_mass(
        masks, labels=masks, index=np.unique(masks)[1:]
    )

    return masks, center_of_mass


def create_dmap_from_image(function: Callable):
    """Generate DynmaicMap from a callable returning an hv.Image object. Adjoins a histogram to the image.

    Args:
        function (Callable): Callabale that returns an hv.Image object

    Returns:
        hv.DynamicMap: regridded hv.DynamicMap with and adjoined histogram
    """
    dmap = hv.DynamicMap(function).opts(
        responsive=True,
        aspect="equal",
    )

    regridded = regrid(dmap)
    histogram = regridded.hist(adjoin=False, normed=True).opts(
        responsive=True, width=125
    )

    return regridded << histogram


docker_client = DockerClient()
