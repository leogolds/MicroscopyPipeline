import os
import sys

# from time import sleep

from java.io import File
from java.lang import System

from ij import IJ
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.io import TmXmlReader, TmXmlWriter


# We have to do the following to avoid errors with UTF8 chars generated in
# TrackMate that will mess with our Fiji Jython.
reload(sys)
sys.setdefaultencoding("utf-8")

# -----------------
# Read data stack
# -----------------

# Get currently selected image
# imp = WindowManager.getCurrentImage()
data_path = "/data/" + os.environ.get("TIFF_STACK")
print("reading data from: " + data_path)
# imp = IJ.openImage("https://fiji.sc/samples/FakeTracks.tif")
imp = IJ.openImage(data_path)
print("data read successfully")

# -----------------
# Read settings from XML
# -----------------

settings_path = "/settings/" + os.environ.get("SETTINGS_XML")
print("Reading settings from: " + settings_path)
file = File(settings_path)
reader = TmXmlReader(file)
if not reader.isReadingOk():
    sys.exit(reader.getErrorMessage())
print("Settings read successfully")
# -----------------
# Get a full model
# -----------------

model = reader.getModel()
model.setLogger(Logger.IJ_LOGGER)
# print("Reading data stack at: " + data_stack_path)


settings = reader.readSettings(imp)

# -------------------
# Instantiate plugin
# -------------------

trackmate = TrackMate(model, settings)

# --------
# Process
# --------
# sleep(360)
ok = trackmate.checkInput()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

ok = trackmate.process()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

# Echo results with the logger we set at start:
model.getLogger().log(str(model))

# --------
# Write Results
# --------
print("writing results.xml")
f = File("/data/" + os.environ.get("TIFF_STACK") + ".xml")
xml_writer = TmXmlWriter(f)

xml_writer.appendModel(model)
xml_writer.appendSettings(settings)
xml_writer.writeToFile()
print("finished writing results.xml")


print("Analysis complete. Exiting...")
System.exit(0)
