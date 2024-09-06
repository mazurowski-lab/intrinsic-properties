# The Effect of Intrinsic Dataset Properties on Generalization (ICLR 2024)

#### By [Nicholas Konz](https://nickk124.github.io/) and [Maciej Mazurowski](https://sites.duke.edu/mazurowski/).

[![arXiv Paper](https://img.shields.io/badge/arXiv-2401.08865-orange.svg?style=flat)](https://arxiv.org/abs/2401.08865)



### *Check out our related papers!*
- [Pre-processing and Compression: Understanding Hidden Representation Refinement Across Imaging Domains via Intrinsic Dimension
](https://arxiv.org/abs/2408.08381) (arXiv 2024)
- [The Intrinsic Manifolds of Radiological Images and their Role in Deep Learning](https://arxiv.org/abs/2207.02797) (MICCAI 2022)



<img src='https://github.com/mazurowski-lab/intrinsic-properties/blob/main/figures/teaser.png' width='100%'>

This is the code for our [ICLR 2024 paper](https://arxiv.org/abs/2401.08865) "The Effect of Intrinsic Dataset Properties on Generalization: Unraveling Learning Differences Between Natural and Medical Images". Our paper shows how a neural network's generalization ability (test performance), adversarial robustness, etc., depends on measurable intrinsic properties of its training set, which we find can vary noticeably between imaging domains (e.g., natural images vs. medical images). Also check out our [poster](https://nickk124.github.io/assets/pdf/intrinsicproperties_iclr2024.pdf) for more info.

Using this code, you can measure these intrinsic properties of your dataset: 
1. The **label sharpness** $\hat{K}_F$ of your dataset, our proposed metric which measures the extent to which images in the dataset can resemble each other while still having
different labels.
2. The **intrinsic dimension** $d_{\text{data}}$ of your dataset, i.e., the minimum number of degrees of freedom needed to describe it.
3. The intrinsic dimension $d_{\text{repr}}$ of the **learned representations** of some layer of a network, given the input dataset.

## Citation

Please cite our ICLR 2024 paper if you use our code or reference our work (published version citation forthcoming):
```bib
@inproceedings{konz2024intrinsicproperties,
title={The Effect of Intrinsic Dataset Properties on Generalization: Unraveling Learning Differences Between Natural and Medical Images},
author={Konz, Nicholas and Mazurowski, Maciej A},
booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
year={2024},
url={https://openreview.net/forum?id=ixP76Y33y1}
}
```

## Quickstart
### Code Usage/Installation
Run the following commands in the main directory:
```bash
pip3 install -r requirements.txt
git clone https://github.com/ppope/dimensions.git
cp utils/dimensions_init_fix.py dimensions/estimators/__init__.py
```

### Measure intrinsic properties of your dataset (on GPU)

```python
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
```

Output:
```
label sharpness = 0.106
data intrinsic dim = 19
representation intrinsic dim = 24
```

#### A few notes about label sharpness
1. The label sharpness $\hat{K}_F$ was formulated under the binary classification scenario, where data is either labeled with `0` or `1`. However, it could be extended to the multi-class scenario by simply replacing the $|y_j-y_k|$ term in the numerator of Eq. 1 in the paper with the indicator function $1({y_j\neq y_k})$, as suggested in Appendix A.1 of our paper. This is currently automatically done in our code.
2. When comparing the label sharpness $\hat{K}_F$ of different datasets, use the same image resolution, channel count, and normalization range for all of them. As shown in our paper's appendix, $\hat{K}_F$ is invariant to changes in these transformations besides all datasets’ $\hat{K}_F$ values being multiplied by the same positive constant; i.e., the relative ranking of the $\hat{K}_F$ of each dataset stays the same with respect to such transformations, **as long as they are kept the same for all datasets.**

## Reproducing Our Paper's Results

### Step 1: Dataset Setup
- **Natural image datasets:** the natural image datasets used in our paper and code (ImageNet, CIFAR-10, SVHN and MNIST) are just the [torchvision Datasets](https://pytorch.org/vision/0.16/datasets.html).

- **Medical image datasets:** The medical image datasets are a bit more complicated to install, but step-by-step instructions can be found in step (1) of [the tutorial for our previous paper](https://github.com/mazurowski-lab/radiologyintrinsicmanifolds/blob/main/reproducibility_tutorial.md).

### Step 2: Code Usage

We provide all code used to reproduce the experiments in our paper:
1. `train.py`: run to train multiple models on the different datasets.
2. `estimate_datadim_allmodels.py`: run to estimate the intrinsic dimension of the training sets of multiple models.
3. `estimate_reprdim_allmodels.py`: run to estimate the intrinsic dimension of the learned representations of multiple models, for model layers of choice.
4. `adv_atk_allmodels.py`: run to evaluate the robustness of multiple models to adversarial attack.
