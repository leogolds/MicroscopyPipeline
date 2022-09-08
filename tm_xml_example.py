from trackmate_utils import TrackmateXML
from pathlib import Path
from shapely.geometry.polygon import Polygon

path = Path(r"D:\Data\MicroscopyPipeline\ser1\segmented_cropped_short.xml")
tm = TrackmateXML(path)

polygon = Polygon(tm.spots.loc[4096].ROI)
x, y = tm.spots.loc[4096].POSITION_X, tm.spots.loc[4096].POSITION_Y

print("asdf")
