from pydantic import BaseModel, confloat, conint, FilePath, validator
from typing import Literal, Union, Dict
from cellpose.models import MODEL_NAMES
from typing import Sequence, List
from enum import Enum
from pathlib import Path, PosixPath, PurePosixPath, PureWindowsPath, WindowsPath
from importlib import import_module
import sys
import yaml

cellpose_models = set(MODEL_NAMES)


class Preprocessing(BaseModel):
    clip_limit_left: confloat(ge=0, le=1) = 0.03
    clip_limit_right: confloat(ge=0, le=1) = 0.03


class ProbabilityMaps(BaseModel):
    model_path_left: FilePath
    model_path_right: FilePath

    @validator("model_path_left", "model_path_right")
    def pathlib_to_string(cls, v):
        return str(v)

    # class Config:
    #     json_encoders = {FilePath: str}


class MergeChannels(BaseModel):
    probability_threshold_left: confloat(ge=0, le=1) = 0.9
    probability_threshold_right: confloat(ge=0, le=1) = 0.9


class SegmentStack(BaseModel):
    diameter_um: conint(ge=10, le=100) = 25
    use_gpu: bool = False
    model: Union[Literal[tuple(cellpose_models)], FilePath] = "cyto"

    @validator("model")
    def validate_model(cls, value):
        if value in cellpose_models:
            return value

        path = Path(value)
        if path.is_file():
            return path

        raise ValueError


class Analysis(BaseModel):
    left_channel: FilePath
    right_channel: FilePath
    interactive: bool = True
    # steps: List[Union[Preprocessing, ProbabilityMaps, MergeChannels, SegmentStack]]
    steps: Dict[str, BaseModel]

    @validator("left_channel", "right_channel")
    def pathlib_to_string(cls, v):
        return str(Path(v).resolve())

    @validator("steps", pre=True)
    def validate_steps(cls, value):
        if isinstance(value, dict):
            return {
                name: getattr(sys.modules[__name__], name)(**values)
                for name, values in value.items()
            }
        elif isinstance(value, list):
            if not all([isinstance(item, BaseModel) for item in value]):
                raise ValueError
            return {step.__class__.__name__: step for step in value}

        raise ValueError


# def load_yaml(path: Path):
#     d = yaml.safe_load(path.read_text())
#     a = Analysis(interactive=)
