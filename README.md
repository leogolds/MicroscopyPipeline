# MicroscopyPipeline
Left empty on purpose

# Pipeline
1. Split channels (phase, C1, C2)
2. Normalize histogram (CLAHE) individually per channel
3. Feed C1, C2 into ilastik using pretrained pipline to create probability maps
4. Feed probability maps CellProfiler using predefined pipeline
