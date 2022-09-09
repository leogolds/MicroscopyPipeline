from pathlib import Path
from utils import segment_stack

base_path = Path(r"data/fucci_60_frames")
red_stack = base_path / "red.tif"
green_stack = base_path / "green.tif"

base_model_path = Path(r"models/cellpose")
red_model = base_model_path / "nuclei_red_v2"
green_model = base_model_path / "nuclei_green_v2"

segment_stack(red_stack, str(red_model))
segment_stack(green_stack, str(green_model))
