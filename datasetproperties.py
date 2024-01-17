"""
functions for computing main intrinsic properties of a dataset
"""
from tqdm import tqdm
from random import sample
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from src.dimensionality import estimate_intrinsic_dim
from src.nnutils import LayerActivationsDataset

def makelogdir():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    return

def compute_labelsharpness(
        dataset, 
        batchsize=500,
        M=1000, 
        device="cuda"
        ):
    """
    compute the label sharpness ($\hat{K}_\mathcal{F}$) of a dataset, as defined in Sec. 3.2 of the paper.

    Args:
        dataset: torch.utils.data.Dataset object
        batchsize: batchsize to use for computing the Lipschitz constant estimate
        M: number of samples to use for computing the Lipschitz constant estimate (See Sec. 3.2 of the paper)
        device: device to use for computing the Lipschitz constant estimate
    """

    makelogdir()

    # deal with multiclass version instead of binary
    multiclass = not all([s[1] in [0, 1] for s in enumerate(dataset)])

    dataset_sampled = Subset(dataset, sample(list(range(len(dataset))), M))

    shuffle = True
    dataloader1 = DataLoader(dataset_sampled, batch_size=batchsize, shuffle=shuffle)
    dataloader2 = DataLoader(dataset_sampled, batch_size=batchsize, shuffle=shuffle)

    Ks = []

    for bi1, s1 in tqdm(enumerate(dataloader1), desc='computing dataset label sharpness', total=M//batchsize):
        for bi2, s2 in enumerate(dataloader2):
            x1, y1 = s1
            x2, y2 = s2
            x1 = x1.to(device)
            x2 = x2.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)

            if multiclass:
                # multiclass version
                numerator = (y1 == y2).float()
            else:
                # default binary version
                numerator = torch.abs(y1.float() - y2.float())

            K = numerator / torch.norm(x1.float() - x2.float(), p=2, dim=(1, 2, 3))
            # filter out where x1 = x2
            K = K[~K.isnan()]
            K = K.tolist()
            Ks += K

    Kmax = np.max(Ks)

    return Kmax 


def compute_intrinsic_datadim(
        dataset, 
        estimator='mle', 
        estimator_batchsize=1000, 
        hyperparam_k=20, 
        dataset_name="dataset"
        ):
    """
    compute the intrinsic dimension of a dataset, as defined in Sec. 3.1 of the paper.

    Args:
        dataset: torch.utils.data.Dataset object
        estimator: estimator to use for computing the intrinsic dimension. Must be one of 'mle', 'geomle', 'twonn'
        estimator_batchsize: batchsize to use for computing the intrinsic dimension estimate
        hyperparam_k: hyperparameter k to use for computing the intrinsic dimension (See Sec. 3.1 of the paper)
        dataset_name: name of the dataset, used for saving the intrinsic dimension estimate
    """
    assert estimator in ['mle', 'geomle', 'twonn'], "estimator must be one of 'mle', 'geomle', 'twonn'"

    makelogdir()

    return estimate_intrinsic_dim(dataset, dataset_name, estimator, batchsize=estimator_batchsize, hyperparam_k=hyperparam_k)

def compute_intrinsic_reprdim(
        dataset, 
        model, 
        layer, 
        estimator='mle', 
        batchsize=256,
        estimator_batchsize=1000,
        hyperparam_k=20, 
        dataset_name="dataset", 
        device="cuda"
        ):
    """
    compute the intrinsic dimension of a neural network's learned representations of a dataset, in one of it's layers.

    Args:
        dataset: torch.utils.data.Dataset object
        model: torch.nn.Module object; the neural network to use for computing the intrinsic dimension
        layer: torch.nn.Module of the layer to use for computing the intrinsic dimension, an attribute of model (e.g. model.layer1 for resnet18)
        estimator: estimator to use for computing the intrinsic dimension. Must be one of 'mle', 'geomle', 'twonn'
        batchsize: batchsize to use for computing the neural network's activations given input data
        estimator_batchsize: batchsize to use for computing the intrinsic dimension estimate
        hyperparam_k: hyperparameter k to use for computing the intrinsic dimension (See Sec. 3.1 of the paper)
        dataset_name: name of the dataset, used for saving the intrinsic dimension estimate
        device: device to use for computing the neural network's activations given input data
    """
    assert estimator in ['mle', 'geomle', 'twonn'], "estimator must be one of 'mle', 'geomle', 'twonn'"

    makelogdir()

    input_dataloader = DataLoader(dataset, batch_size=batchsize)

    # register hook to save activations
    activations = []
    def hook(model, input, output):
        activations.append(output.detach().cpu())
    handle = layer.register_forward_hook(hook)

    # compute activations by passing data through net
    for batch_idx, (x_in, _) in tqdm(enumerate(input_dataloader), 
                                        desc='completing forward passes...',
                                        total=len(dataset)//batchsize
                                    ):
        x_in = x_in.to(device)
        output = model(x_in) 
        # memory management/get things off GPU
        del output

    activation_data = torch.cat(activations)

    handle.remove() # remove hook so earlier layers aren't tracked
    
    activation_data = activation_data.to('cpu')
    # memory management/get things off GPU

    # load activations into dataset
    lbls = [dataset[i][1] for i in range(len(dataset))]
    activation_dataset = LayerActivationsDataset(activation_data, lbls)


    # compute intrinsic dim
    try:
        layer_activations_intrinsic_dim = estimate_intrinsic_dim(activation_dataset, 
                                                            dataset_name, estimator, 
                                                            batchsize=estimator_batchsize, hyperparam_k=hyperparam_k)
        return layer_activations_intrinsic_dim

    except (ValueError, OverflowError) as e:
        # NaN or inf result for ID
        print(e)
        return np.nan
