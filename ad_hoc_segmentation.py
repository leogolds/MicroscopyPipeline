import utils
from pathlib import Path

base_path = Path(r"D:\Data\MicroscopyPipeline\ser1")
path = base_path / "red_contrast_short_cropped.tif"
segmented_stack, segmentation_table = utils.segment_h5_stack(path)
print("asdf")
