from typing import Iterable

import psutil
import numpy as np
from pathlib import Path
import h5py
import tifffile
from cellpose import models
from docker.client import DockerClient
from docker.types import Mount
from scipy import ndimage
from tqdm import trange
import holoviews as hv

hv.extension("bokeh")


def read_stack(path: Path) -> np.ndarray:
    if path.suffix == "h5":
        f = h5py.File(path)
        return f.get("data", f.get("exported_data"))
    else:
        return tifffile.imread(path)


def view_stacks(images: Iterable[np.ndarray], frame: int):
    opts = {
        "aspect": images[0].shape[2] / images[0].shape[1],
        "invert_yaxis": True,
        "responsive": True,
    }
    bounds = (0, 0, images[0].shape[2], images[0].shape[1])

    # construct holoviews objects
    hv_images = [
        hv.Image(np.flipud(img[frame, ...]), bounds=bounds).opts(cmap="gray", **opts)
        for img in images
    ]

    layout = hv.Layout(hv_images).cols(1).opts(shared_axes=True)

    return layout


def view_segmented_stacks(images: Iterable[np.ndarray], frame: int):
    layout = view_stacks(images, frame)
    image_opts = hv.opts.Image(
        colorbar=False,
        cmap="glasbey",
        clipping_colors={"min": "black"},
    )
    layout = layout.redim.range(z=(1, np.inf)).opts(image_opts)
    return layout


def view_segmentation_overlay(
    images: Iterable[np.ndarray], segmentation_maps: Iterable[np.ndarray], frame: int
):
    base_layout = view_stacks(images, frame)
    segmentation_opts = hv.opts.Image(alpha=0.3)
    segmentation_overlay = view_segmented_stacks(segmentation_maps, frame).opts(
        segmentation_opts
    )

    return hv.Layout(
        [
            base_img * segmentation
            for base_img, segmentation in zip(base_layout, segmentation_overlay)
        ]
    ).cols(1)


def segment_stack(path: Path, model: Path, export_tiff=True):
    print(f"segmenting stack at {path} with model at {model}")
    stack = read_stack(path)

    frames, Y, X = stack.shape
    # frames, _, Y, X, _ = stack.shape

    new_file_path = path.parent / f"{path.stem}_segmented.h5"
    dataset_name = "data"

    with h5py.File(new_file_path, "w") as f:
        dataset = f.create_dataset(
            dataset_name, shape=(frames, Y, X), dtype=np.uint16, chunks=True
        )

        for frame in trange(frames):
            masks, center_of_mass = segment_frame(stack[frame, ...], model, gpu=True)
            dataset[frame, :, :] = masks

        if export_tiff:
            new_tiff_path = path.parent / f"{path.stem}_segmented.tiff"
            print(f"exporting to tiff at {new_tiff_path}")
            with tifffile.TiffWriter(new_tiff_path, bigtiff=True) as tif:
                tif.write(f.get(dataset_name), shape=(frames, Y, X))

    print("segmentation complete")


def segment_frame(img, model: Path, gpu: bool = False, diameter: int = 25):
    channels = [0, 0]
    net_avg = False
    resample = False

    model = models.CellposeModel(
        gpu=gpu,
        pretrained_model=str(model.absolute()),
        nchan=2,
    )
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


def run_trackmate(settings_path: Path, data_path: Path):
    print(
        f"Running TrackMate on segmented stack at {data_path} using settings at {settings_path}"
    )
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

    print(f"Tracking on {data_path} complete")


def run_analysis(
    stack_path: Path, cellpose_model_path: Path, trackmate_config_path: Path
):
    assert stack_path.exists(), f"Could not find image stack at {stack_path.absolute()}"
    assert (
        cellpose_model_path.exists()
    ), f"Could not find CellPose model at {cellpose_model_path.absolute()}"
    segment_stack(stack_path, cellpose_model_path)
    segmented_stack_path = stack_path.parent / f"{stack_path.stem}_segmented.tiff"

    assert (
        trackmate_config_path.exists()
    ), f"Could not find TrackMate xml at {trackmate_config_path.absolute()}"
    run_trackmate(trackmate_config_path, segmented_stack_path)
    trackmate_results_path = (
        segmented_stack_path.parent / f"{segmented_stack_path.stem}.xml"
    )

    print(f"Segmentation map saved to:\t{segmented_stack_path.absolute()}")
    print(f"Tracking result saved to:\t{trackmate_results_path.absolute()}")


docker_client = DockerClient()
