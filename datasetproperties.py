"""
functions for computing main intrinsic properties of a dataset
"""
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dimensionality import estimate_intrinsic_dim

def compute_labelsharpness(dataset, batchsize=500, M=1000, device="cuda"):
    # ^ M is the number of samples used for the Lipschitz constant estimate (See Sec. 3.2 of the paper)
    shuffle = True
    dataloader1 = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
    dataloader2 = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)

    Ks = []

    for bi1, s1 in tqdm(enumerate(dataloader1), desc='computing dataset label sharpness', total=M//batchsize):
        for bi2, s2 in enumerate(dataloader2):
            x1, y1 = s1
            x2, y2 = s2
            x1 = x1.to(device)
            x2 = x2.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)

            K = torch.abs(y1.float() - y2.float()) / torch.norm(x1.float() - x2.float(), p=2, dim=(1, 2, 3))
            # filter out where x1 = x2
            K = K[~K.isnan()]
            K = K.tolist()
            Ks += K

    Kmax = np.max(Ks)

    return Kmax 


def compute_intrinsicdim(dataset, estimator='mle', batchsize=1000, hyperparam_k=20):
    assert estimator in ['mle', 'geomle', 'twonn'], "estimator must be one of 'mle', 'geomle', 'twonn'"
    return estimate_intrinsic_dim(dataset, "dataset", estimator, batchsize=batchsize, hyperparam_k=hyperparam_k)