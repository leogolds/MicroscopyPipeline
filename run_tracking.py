from utils import run_trackmate
from pathlib import Path

settings_xml = Path(r"models/trackmate/basic_settings.xml")
red_data_stack = Path(r"data/fucci_60_frames/red_segmented.tiff")
green_data_stack = Path(r"data/fucci_60_frames/green_segmented.tiff")


assert settings_xml.exists(), f"Settings not found at path: {settings_xml}"
assert (
    red_data_stack.exists()
), f"Red segmented data not found at path: {red_data_stack}"
assert (
    green_data_stack.exists()
), f"Green segmented data not found at path: {green_data_stack}"

run_trackmate(settings_xml, red_data_stack)
run_trackmate(settings_xml, green_data_stack)
