"""
Compute intrinsic dimension of representations in neural nets (all models/datasets)
"""
# environment setup

# imports
from src.utils import *
from src.vizutils import *
from src.nnutils import *
from src.dimensionality import *
from src.dataset import *

import os
from tqdm import tqdm
import numpy as np
import random

# torch
import torch
from torchvision.models import resnet18, resnet34, resnet50, vgg13, vgg16, vgg19

# GPU settings
device_ids = [0] # indices of devices for models, data and otherwise
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
print('running on {}'.format(device))

# set random seed
seed = 1337
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# hyperparams
# use if you have statedicts saved from nn.DataParallel training:
parallel_trained_statedicts = False

# model and input dataset
models = [resnet18, resnet34, resnet50, vgg13, vgg16, vgg19]
dataset_names = ['brats', 'dbc', 'oai', 'chexpert', 'mura', 'rsna', 'prostate'] + ['MNIST', 'CIFAR10', 'SVHN', 'ImageNet']
#dataset_names = ['isic']

labelings = ['default']
training_sizes = list(range(500, 1750+250, 250))
test_size = 750
input_dataset_batchsize = 64  # to compute activations for input dataset

img_size = 224

# activation intrinsic dim estimation
id_estimators = ['twonn'] # or 'mle'
estimator_batchsize = 1000

# options
which_repr_layer = "penultimate"

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# logger
log_dir = 'logs/repr_dimensionality/allmodels_alldata'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger = Logger('custom', log_dir)
header = ' '.join([
    'dataset',
    'n_train',
    'labeling',
    'model',
    'layer',
    'layer_depth',
    'estimator',
    'ID'
])
logger.write_msg(header)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# master loop for all experiments:
for intrinsic_dim_estimator in id_estimators:
    for dataset_name in dataset_names:
        for train_size in training_sizes:
            for labeling in labelings:
                for model in models:
                    # find statedict fname 
                    checkpoint_dir = "saved_models/generalization/{}".format(dataset_name)
                    statedict_fname = None
                    statedict_fnames = [f for f in os.listdir(checkpoint_dir) if f.endswith(".h5")]
                    keep_statedict_fnames = []
                    for fn in statedict_fnames:
                        fn_split = fn.split("_")
                        if (fn_split[:3] == [model.__name__, str(train_size), str(test_size)]) and (fn_split[5] == labeling):
                            keep_statedict_fnames.append(fn)
                    
                    if len(keep_statedict_fnames) == 0:
                        raise FileNotFoundError("no saved model found for setting: {}".format([dataset_name, model.__name__, train_size, test_size, labeling]))

                    for statedict_fname in keep_statedict_fnames:
                        if dataset_name in natural_dataset_names:
                            chosen_classes = statedict_fname.split("_")[6:8]
                            chosen_classes = [int(c) for c in chosen_classes]
                            
                        # load data
                        if dataset_name in natural_dataset_names:
                            trainset, _ = get_datasets(dataset_name, 
                                                subset_size=train_size+test_size, 
                                                test_size=test_size,
                                                labeling=labeling,
                                                img_size=img_size,
                                                class1=chosen_classes[0],
                                                class2=chosen_classes[1]
                                            )
                        else:
                            trainset, _ = get_datasets(dataset_name, 
                                                subset_size=train_size+test_size, 
                                                test_size=test_size,
                                                labeling=labeling,
                                                img_size=img_size,
                                                special_medicalisrgb = dataset_name in special_rgb_medical_dataset_names
                                            )

                        input_dataloader = DataLoader(trainset,
                                batch_size=input_dataset_batchsize)
                        
                        # instantiate model, put on device
                        net = model()
                        net.eval()

                        # load checkpoint
                        net_path = os.path.join(checkpoint_dir, statedict_fname)
                        state_dict = torch.load(net_path, map_location='cpu')['net']
                        if not parallel_trained_statedicts:
                            # for loading models created in parallel, but not in parallel
                            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                        
                        # fix first lyr if one channel needed
                        if dataset_name in natural_dataset_names and dataset_name != "MNIST":
                            print("net takes 3 channels as input.")
                        elif dataset_name in special_rgb_medical_dataset_names:
                            print("net takes 3 channels as input.")
                        else:
                            print("making net take 1 input channel.")
                            make_netinput_onechannel(net, model)

                        net.load_state_dict(state_dict, strict=False)
                        print('network loaded: {} for {}.'.format(os.path.basename(net_path), dataset_name))


                        # get layers of interest for the given model
                        layers, layer_names, layer_depths = get_activation_layers(net, model)          
                        # iterate through the layers

                        net = net.to(device)
                        if parallel_trained_statedicts:
                            net = torch.nn.DataParallel(net, device_ids = range(len(device_ids)))
                        
                        for layer_idx, layer in enumerate(layers):
                            if layer_depths[layer_idx] != get_repr_layer_depth(model.__name__, mode=which_repr_layer):
                                continue
                            if type(layer) != str: # first "layer" may just be a str.
                                # register hook to save activations
                                activations = []
                                def hook(net, input, output):
                                    activations.append(output.detach().cpu())
                                handle = layer.register_forward_hook(hook)

                                # compute activations by passing data through net
                                for batch_idx, (x_in, _) in tqdm(enumerate(input_dataloader), 
                                                                 desc='completing forward passes...',
                                                                 total=len(trainset)//input_dataset_batchsize
                                                                ):
                                    x_in = x_in.to(device)
                                    output = net(x_in) 
                                    # memory management/get things off GPU
                                    del output

                                activation_data = torch.cat(activations)

                                handle.remove() # remove hook so earlier layers aren't tracked
                                
                                activation_data = activation_data.to('cpu')
                                # memory management/get things off GPU

                                # load activations into dataset
                                if dataset_name in natural_dataset_names:
                                    # get labels without filename placeholder
                                    lbls = [l[1] for l in trainset.dataset.labels]
                                    activation_dataset = LayerActivationsDataset(activation_data, lbls)
                                else:
                                    activation_dataset = LayerActivationsDataset(activation_data, trainset.dataset.labels)


                                # compute intrinsic dim
                                try:
                                    layer_activations_intrinsic_dim = estimate_intrinsic_dim(activation_dataset, 
                                                                                        dataset_name, intrinsic_dim_estimator, 
                                                                                        batchsize=estimator_batchsize)
                                except (ValueError, OverflowError) as e:
                                    # NaN or inf result for ID
                                    print(e)
                                    continue
                                
                                # log it
                                log_msg = ' '.join([
                                    dataset_name,
                                    str(train_size),
                                    labeling,
                                    model.__name__,
                                    layer_names[layer_idx],
                                    str(layer_depths[layer_idx]),
                                    intrinsic_dim_estimator,
                                    str(layer_activations_intrinsic_dim)
                                ])

                                logger.write_msg(log_msg)
