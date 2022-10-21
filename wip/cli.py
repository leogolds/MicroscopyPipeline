import typer
from pathlib import Path
from config import Analysis
import yaml
import utils
import h5py
from numpy import uint8
from skimage.exposure import equalize_adapthist
import numpy as np
from datetime import datetime

app = typer.Typer(add_completion=False)


@app.command()
def main(
    # image_stack: Path = typer.Argument(
    #     ...,
    #     exists=True,
    #     file_okay=True,
    #     dir_okay=False,
    #     readable=True,
    #     resolve_path=True,
    #     help="Path to image stack to process",
    # ),
    config: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Optionally specify path to config.yaml. If not specified, will look for config.yaml in image_stack directory",
    ),
):
    # if not config:
    #     config = image_stack.parent / "config.yaml"

    analysis_config = Analysis.parse_obj(yaml.safe_load(config.read_text()))

    current_left_stack = Path(analysis_config.left_channel)
    current_right_stack = Path(analysis_config.right_channel)

    start_time = datetime.now()
    # for step_name, step_config in analysis_config.steps.items():
    # print(step_name, step_config)
    typer.echo(f"Starting preprocessing {current_left_stack}")
    current_left_stack = preprocess(
        current_left_stack, analysis_config.steps.get("Preprocessing").clip_limit_left
    )
    left_preprocess_time = datetime.now()
    typer.echo(f"Finished preprocessing {current_left_stack}\n")

    typer.echo(f"Starting preprocessing {current_right_stack}")
    current_right_stack = preprocess(
        current_right_stack, analysis_config.steps.get("Preprocessing").clip_limit_right
    )
    right_preprocess_time = datetime.now()
    typer.echo(f"Finished preprocessing {current_right_stack}\n")

    typer.echo(f"Starting probability map {current_left_stack}")
    current_left_stack = probability_maps(
        current_left_stack, analysis_config.steps.get("ProbabilityMaps").model_path_left
    )
    left_probability_map_time = datetime.now()
    typer.echo(f"Finished probability map {current_left_stack}\n")
    typer.echo(f"Starting probability map {current_right_stack}")
    current_right_stack = probability_maps(
        current_right_stack,
        analysis_config.steps.get("ProbabilityMaps").model_path_right,
    )
    right_probability_map_time = datetime.now()
    typer.echo(f"Finished probability map {current_right_stack}\n")

    typer.echo(f"Starting merge channels {current_left_stack}, {current_right_stack}")
    merged_probabilties = merge_probabilities_maximum_likelihood(
        current_left_stack,
        current_right_stack,
        analysis_config.steps.get("MergeChannels").probability_threshold_left,
        analysis_config.steps.get("MergeChannels").probability_threshold_right,
    )
    merge_time = datetime.now()
    typer.echo(f"Finished merge channels {merged_probabilties}\n")

    typer.echo(f"Starting segmentation {merged_probabilties}")
    segmented_stack, segmentation_table = utils.segment_h5_stack(merged_probabilties)
    segmentation_time = datetime.now()
    typer.echo(f"Finished segmentation {segmented_stack}, {segmentation_table}\n")

    typer.echo(f"\n\nAnalysis took: {segmentation_time-start_time}")
    typer.echo(f"Preprocessing: {right_preprocess_time-start_time}")
    typer.echo(f"Merging: {merge_time-right_preprocess_time}")
    typer.echo(f"Segmentation: {segmentation_time-merge_time}")


def preprocess(stack: Path, clip_limit: float) -> Path:
    new_file_path = stack.parent / f"{stack.stem}_contrast.h5"

    stack = h5py.File(stack).get("data")
    frames, _, Y, X, _ = stack.shape

    with h5py.File(new_file_path, "w") as f:
        data = f.create_dataset(
            "data",
            shape=(frames, 1, Y, X, 1),
            dtype="uint8",
            chunks=True,
        )
        with typer.progressbar(range(stack.shape[0])) as progress:
            for frame in progress:
                data[frame, 0, ..., 0] = (
                    equalize_adapthist(stack[frame, 0, ..., 0], clip_limit=clip_limit)
                    * 255
                ).astype(uint8)

    return new_file_path


def probability_maps(stack_path: Path, model_path: Path) -> Path:
    stack_path = Path(stack_path)
    model_path = Path(model_path)

    command = [
        f"--project=/model/{model_path.name}",
        f"/data/{stack_path.name}",
    ]

    volumes = {
        f"{model_path.parent.absolute()}": {"bind": "/model/", "mode": "ro"},
        f"{stack_path.parent.absolute()}": {"bind": "/data/", "mode": "rw"},
    }

    container = utils.docker_client.containers.run(
        image="ilastik-container",
        detach=True,
        command=" ".join(command),
        volumes=volumes,
        environment={"LAZYFLOW_THREADS": 32},
    )

    for line in container.logs(stream=True):
        typer.echo(line.decode("utf-8"))

    return stack_path.parent / f"{stack_path.stem}_Probabilities.h5"


def merge_probabilities_maximum_likelihood(
    left_stack_path: Path,
    right_stack_path: Path,
    probability_threshold_left: float,
    probability_threshold_right: float,
) -> Path:
    new_file_path = left_stack_path.parent / f"merged_probabilities.h5"

    left_stack = utils.read_stack(left_stack_path)
    right_stack = utils.read_stack(right_stack_path)

    frames, _, Y, X, _ = left_stack.shape

    with h5py.File(new_file_path, "w") as f:
        data = f.create_dataset(
            "data",
            # shape=left_stack.shape[:-1],
            shape=(frames, Y, X),
            dtype=left_stack.dtype,
            chunks=True,
        )

        with typer.progressbar(range(left_stack.shape[0])) as progress:
            for frame in progress:
                left_frame = left_stack[frame, 0, ..., 0]
                masked_left_frame = np.where(
                    left_frame > probability_threshold_left, left_frame, 0
                )
                right_frame = right_stack[frame, 0, ..., 0]
                masked_right_frame = np.where(
                    right_frame > probability_threshold_right, right_frame, 0
                )
                # merged_frame = np.sqrt(
                #     np.power(masked_left_frame, 2) + np.power(masked_right_frame, 2)
                # )
                merged_frame = np.maximum(masked_left_frame, masked_right_frame)

                data[frame, ...] = merged_frame.astype(left_stack.dtype)

    return new_file_path


if __name__ == "__main__":
    app()
