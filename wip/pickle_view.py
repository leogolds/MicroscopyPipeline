# import pickle
# import torch

# # with open(r"C:\Users\user\Downloads\cytotorch_1", "r") as f:
f = r"C:\Users\user\Downloads\cytotorch_1"
# # print(pickle.load(f))
# a = torch.load(f, map_location=torch.device("cpu"))

# print(a)
from cellpose.resnet_torch import CPnet
from torchinfo import summary
import torch

nclasses = 3
nbase = [32, 64, 128, 256]
nchan = 2
nbase = [nchan, 32, 64, 128, 256]

a = CPnet(nbase, nclasses, sz=3)
a.load_model(
    f,
    cpu=True,
)
print(summary(a, input_size=[32, 64, 128, 256]))
