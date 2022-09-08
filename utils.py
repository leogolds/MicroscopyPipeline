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


def read_stack(path: Path) -> np.ndarray:
    if path.suffix == "h5":
        f = h5py.File(path)
        return f.get("data", f.get("exported_data"))
    else:
        return tifffile.imread(path)


def segment_h5_stack(path):
    stack = read_stack(path)

    frames, Y, X = stack.shape
    # frames, _, Y, X, _ = stack.shape

    # df = pd.DataFrame(columns=["center_y", "center_x"])

    new_file_path = path.parent / f"segmented.h5"
    new_table_path = path.parent / f"segmented_table.h5"
    dataset_name = "data"

    with h5py.File(new_file_path, "w") as f:
        dataset = f.create_dataset(
            dataset_name, shape=(frames, Y, X), dtype=np.uint16, chunks=True
        )

        for frame in trange(frames):
            # img = stack[frame, 0, :, :, 0]
            masks, center_of_mass = segment_frame(stack[frame, ...], gpu=True)

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

    return new_file_path, new_table_path


def segment_frame(img, gpu: bool = False, diameter: int = 25):
    # model_type = "nuclei"
    pretrained_model = rf"D:\Data\cellpose_models\nuclei_red"
    # pretrained_model = rf"D:\Data\cellpose_models\nuclei_green"
    # diameter = None
    channels = [0, 0]
    net_avg = False
    resample = False

    # model = models.Cellpose(
    model = models.CellposeModel(
        # gpu=gpu, model_type=model_type
        gpu=gpu,
        pretrained_model=pretrained_model,
        nchan=2,
    )  # model_type can be 'cyto' or 'nuclei'
    # masks, flows, styles, diams = model.eval(
    masks, flows, styles = model.eval(
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


def run_trackmate(settings_path: Path, data_path: Path):
    volumes = {
        f"{settings_path.parent.absolute()}": {"bind": "/settings/", "mode": "ro"},
        f"{data_path.parent.absolute()}": {"bind": "/data/", "mode": "rw"},
    }
    settings_mount = Mount(
        target="/settings",
        source=str(settings_path.parent.absolute()),
        type="bind",
        read_only=True,
    )
    data_mount = Mount(
        target="/data",
        source=str(data_path.parent.absolute()),
        type="bind",
        read_only=False,
    )

    container = docker_client.containers.run(
        image="trackmate",
        detach=True,
        # volumes=volumes,
        mounts=[settings_mount, data_mount],
        environment={
            "SETTINGS_XML": settings_path.name,
            "TIFF_STACK": data_path.name,
            "MEMORY": f"{int(psutil.virtual_memory().total // 1024**3 * 0.5)}G",
        },
    )

    for line in container.logs(stream=True):
        print(line.decode("utf-8"))


docker_client = DockerClient()
