from pathlib import Path

import pandas as pd

import utils
import trackmate_utils


base_data_path = Path(r"data/fucci_60_frames")
red_stack_path = base_data_path / "red.tif"
green_stack_path = base_data_path / "green.tif"

base_model_path = Path(r"models")
red_segmentation_model = base_model_path / "cellpose/nuclei_red_v2"
green_segmentation_model = base_model_path / "cellpose/nuclei_green_v2"

red_segmentation_map = utils.read_stack(base_data_path / "red_segmented.tiff")
green_segmentation_map = utils.read_stack(base_data_path / "green_segmented.tiff")

tm_red = trackmate_utils.TrackmateXML(base_data_path / "red_segmented.tiff.xml")
tm_green = trackmate_utils.TrackmateXML(base_data_path / "green_segmented.tiff.xml")

red_stack = utils.read_stack(base_data_path / "red.tif")
green_stack = utils.read_stack(base_data_path / "green.tif")

# metric = trackmate_utils.CartesianSimilarity(tm_red, tm_green)
# metric_df = metric.calculate_metric_for_all_tracks()
# metric_df.to_hdf(base_data_path / "metric.h5", key="metric")
metric_df = pd.read_hdf(base_data_path / "metric.h5", key="metric")
metric = trackmate_utils.CartesianSimilarityFromFile(tm_red, tm_green, metric_df)


viewer = trackmate_utils.TrackViewer(
    red_stack=red_stack,
    green_stack=green_stack,
    tm_red=tm_red,
    tm_green=tm_green,
    red_segmentation_map=red_segmentation_map,
    green_segmentation_map=green_segmentation_map,
    metric=metric,
)
viewer.view().show()
