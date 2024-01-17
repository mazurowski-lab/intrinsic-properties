from datasetproperties import compute_labelsharpness, compute_intrinsic_datadim, compute_intrinsic_reprdim

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Subset

# first, load dataset
dataset = CIFAR10(root='data', download=True, transform=ToTensor())
classes = [0, 1]
dataset = Subset(dataset, [i for i, s in enumerate(dataset) if s[1] in classes])
# ^ or any torch.utils.data.Dataset

# compute label sharpness and intrinsic dimension of dataset
KF = compute_labelsharpness(dataset)
datadim = compute_intrinsic_datadim(dataset)

# compute intrinsic dimension of dataset representations in some layer of a neural network
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to("cuda")
layer = model.layer4
reprdim = compute_intrinsic_reprdim(dataset, model, layer)

print("label sharpness = {}".format(round(KF, 3)))
print("data intrinsic dim = {}".format(int(datadim)))
print("representation intrinsic dim = {}".format(int(reprdim)))