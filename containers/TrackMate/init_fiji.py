import imagej
import imagej.doctor

imagej.doctor.debug_to_stderr()
ij = imagej.init("/opt/fiji/Fiji.app")
# ij = imagej.init()
print(f"ImageJ version: {ij.getVersion()}")
