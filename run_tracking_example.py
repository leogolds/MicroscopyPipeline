from utils import run_trackmate
from pathlib import Path

settings_xml = Path(r"D:\Data\trackmate_models\basic_settings.xml.template")
data_stack = Path(r"D:\Data\MicroscopyPipeline\ser1-1-20\segmented.tiff")

assert settings_xml.exists(), f"Settings not found at path: {settings_xml}"
assert data_stack.exists(), f"Data not found at path: {data_stack}"

run_trackmate(settings_xml, data_stack)
