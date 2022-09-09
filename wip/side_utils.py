from typing import Callable
import psutil
import aicsimageio.transforms
import numpy as np

from pathlib import Path
import h5py
import tifffile
from cellpose import models
from PIL import Image
import holoviews as hv
from docker.client import DockerClient
from docker.types import Mount

hv.extension("bokeh")
from scipy import ndimage
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
import hvplot.pandas
from tqdm import trange
from holoviews.operation.datashader import regrid
from aicsimageio import AICSImage
import skimage
from typing import Dict


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


def split_channels_and_convert_to_h5_aicsimageio(path: Path) -> Dict[str, Path]:
    """
    This function is specifically targeted for the Lior Atia Lab and the format of XXXX camera
    Args:
        path: path to XXXXX tif stack

    Returns:

    """
    output_path = path.parent / "MicroscopyPipeline" / path.stem
    output_path.mkdir(parents=True, exist_ok=True)

    stack = AICSImage(path)
    _, frames, channels, _, _, _ = stack.shape

    channel_map = {0: "phase", 1: "red", 2: "green"}
    output = {}

    for channel, channel_name in channel_map.items():
        image_stack = stack.get_image_dask_data("TYX", Z=0, C=channel, S=0).compute()

        reshaped = aicsimageio.transforms.reshape_data(image_stack, "TYX", "TZYXC")

        new_file_path = output_path / f"{channel_name}.h5"

        with h5py.File(new_file_path, "w") as f:
            f.create_dataset(
                "data",
                data=skimage.img_as_ubyte(skimage.exposure.rescale_intensity(reshaped)),
                dtype="uint8",
                chunks=True,
            )

        output[channel_name] = new_file_path

    return output


def split_channels_and_convert_to_h5_tifffile(path: Path) -> Dict[str, Path]:
    """
    This function is specifically targeted for the Lior Atia Lab and the format of XXXX camera
    Args:
        path: path to XXXXX tif stack

    Returns:

    """
    output_path = path.parent / "MicroscopyPipeline" / path.stem
    output_path.mkdir(parents=True, exist_ok=True)

    stack = tifffile.imread(path)
    frames, channels, _, _ = stack.shape

    channel_map = {0: "phase", 1: "red", 2: "green"}
    output = {}

    for channel, channel_name in channel_map.items():
        image_stack = stack[:, channel, ...]

        reshaped = aicsimageio.transforms.reshape_data(image_stack, "TYX", "TZYXC")

        new_file_path = output_path / f"{channel_name}.h5"

        with h5py.File(new_file_path, "w") as f:
            f.create_dataset(
                "data",
                data=skimage.img_as_ubyte(skimage.exposure.rescale_intensity(reshaped)),
                dtype="uint8",
                chunks=True,
            )

        output[channel_name] = new_file_path

    return output
