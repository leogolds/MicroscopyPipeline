from utils import segment_stack, run_trackmate
from trackmate_utils import CartesianSimilarity, TrackmateXML
from pathlib import Path

# base_path = Path(r"D:\Data\full_pipeline_tests\left_60_frames")
base_path = Path(r"D:\Data\full_pipeline_tests\right_60_frames")

red_stack = base_path / "red.tif"
green_stack = base_path / "green.tif"

base_model_path = Path(r"models/cellpose")
red_model = base_model_path / "nuclei_red_v2"
green_model = base_model_path / "nuclei_green_v2"

segment_stack(red_stack, red_model)
segment_stack(green_stack, green_model)

settings_xml = Path(r"models/trackmate/basic_settings.xml")
red_data_stack = base_path / "red_segmented.tiff"
green_data_stack = base_path / "green_segmented.tiff"

assert settings_xml.exists(), f"Settings not found at path: {settings_xml}"
assert (
    red_data_stack.exists()
), f"Red segmented data not found at path: {red_data_stack}"
assert (
    green_data_stack.exists()
), f"Green segmented data not found at path: {green_data_stack}"

run_trackmate(settings_xml, red_data_stack)
run_trackmate(settings_xml, green_data_stack)

tm_red = TrackmateXML(base_path / "red_segmented.tiff.xml")
tm_green = TrackmateXML(base_path / "green_segmented.tiff.xml")

metric = CartesianSimilarity(tm_red, tm_green)
metric_df = metric.calculate_metric_for_all_tracks()
metric_df.to_hdf(base_path / "metric.h5", key="metric")

print("Analysis done...")
