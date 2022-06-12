from utils import segment_h5_stack
import h5py
from pathlib import Path

path = Path(r"3pos\pos35\merged_binary_map_short.h5")
assert path.exists()

segment_h5_stack(path)
