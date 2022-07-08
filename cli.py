from importlib.resources import read_text
from xxlimited import new
import typer
from pathlib import Path
from config import Analysis
import yaml
import utils
import h5py
from numpy import uint8

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

    current_left_stack = analysis_config.left_channel
    current_right_stack = analysis_config.right_channel

    # for step_name, step_config in analysis_config.steps.items():
    # print(step_name, step_config)
    current_left_stack = preprocess(
        current_left_stack, analysis_config.steps.Preprocessing.clip_limit_left
    )


def preprocess(stack: Path, clip_limit: float) -> Path:
    new_file_path = stack.parent / f"{stack.stem}_contrast.h5"

    frames, Y, X = stack.shape

    with h5py.File(new_file_path, "w") as f:
        data = f.create_dataset(
            "data",
            shape=(frames, 1, Y, X, 1),
            dtype="uint8",
            chunks=True,
        )
        for frame in range(stack.shape[0]):
            data[frame, 0, ..., 0] = (
                utils.equalize_adapthist(stack[frame, ...], clip_limit=clip_limit) * 255
            ).astype(uint8)

    return new_file_path


if __name__ == "__main__":
    app()
