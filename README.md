This repo has been superseeded by [ConfluentFUCCI](https://github.com/leogolds/ConfluentFUCCI) and is now archived

Please check out the now peer-reviewed [ConfluentFUCCI](https://github.com/leogolds/ConfluentFUCCI) or the accomnying [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0305491)

# MicroscopyPipeline
Left empty on purpose

# Pipeline
1. Split channels (phase, C1, C2)
2. Normalize histogram (CLAHE) individually per channel
3. Feed C1, C2 into ilastik using pretrained pipline to create probability maps
4. Feed probability maps CellProfiler using predefined pipeline
