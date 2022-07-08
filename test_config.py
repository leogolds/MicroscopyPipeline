from config import Analysis, Preprocessing, ProbabilityMaps, MergeChannels, SegmentStack
import yaml

a = Analysis(
    left_channel=r"3pos\pos35\C2_short.h5",
    right_channel=r"3pos\pos35\C3_short.h5",
    steps=[
        Preprocessing(),
        ProbabilityMaps(
            model_path_left=r"C:\Data\Code\MicroscopyPipeline\3pos\pos35\MyProject_pos35_red.ilp",
            model_path_right=r"C:\Data\Code\MicroscopyPipeline\3pos\pos35\MyProject_pos35_green.ilp",
        ),
        MergeChannels(),
        SegmentStack(),
    ],
)

# b = Analysis.parse_obj(yaml.safe_load(yaml.dump(a.dict())))
# # print(list(a.steps))
# print(b)
# with open("config-red-channel.yaml.template", "w") as f:
# with open("config-green-channel.yaml.template", "w") as f:
with open("config.yaml.template", "w") as f:
    f.write(yaml.safe_dump(a.dict(), sort_keys=False))

# with open("config-red-channel.yaml.template", "r") as f:
#     b = Analysis.parse_obj(yaml.safe_load(f.read()))
# assert a == b
