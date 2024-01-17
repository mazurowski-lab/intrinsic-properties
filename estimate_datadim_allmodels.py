
"""
perform and evaluate adversarial attacks on a bunch of trained models.
"""
# imports
from src.vizutils import *
from src.utils import *
from src.dataset import *
from src.nnutils import *
from src.dimensionality import *
import os
import random

# torch
import torch
from torchvision.models import resnet18, resnet34, resnet50, vgg13, vgg16, vgg19

# GPUs
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

# settings
#img_sizes = [32, 64, 128, 256]
img_sizes = [224]
test_size = 750
test_batchsize = 256

intrinsic_dim_estimators = ['mle'] # or twonn
estimator_batchsize = 1024

training_sizes = list(range(500, 1750+250, 250))

dataset_names = ['brats', 'dbc', 'oai', 'chexpert', 'mura', 'rsna', 'prostate'] + ['MNIST', 'CIFAR10', 'SVHN', 'ImageNet']
#dataset_names = ["isic"]

models = [vgg13, vgg16, vgg19, resnet18, resnet34, resnet50]
labeling = 'default'

# logger
log_dir = 'logs/data_dim/allmodels_alldata'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger = Logger('custom', log_dir)
header = ' '.join([
    'dataset',
    'n_train',
    'labeling',
    'model',
    'estimator',
    'ID',
    'img_size'
])
logger.write_msg(header)

for img_size in img_sizes:
    for intrinsic_dim_estimator in intrinsic_dim_estimators:
        for model in models:
            for dataset_name in dataset_names: 
                for training_subset_size in training_sizes:
                    # find statedict fname 
                    checkpoint_dir = "saved_models/generalization/{}".format(dataset_name)
                    statedict_fname = None
                    statedict_fnames = [f for f in os.listdir(checkpoint_dir) if f.endswith(".h5")]
                    keep_statedict_fnames = []
                    for fn in statedict_fnames:
                        fn_split = fn.split("_")
                        if (fn_split[:3] == [model.__name__, str(training_subset_size), str(test_size)]) and (fn_split[5] == labeling):
                            keep_statedict_fnames.append(fn)
                    
                    if len(keep_statedict_fnames) == 0:
                        raise FileNotFoundError("no saved model found for setting: {}".format([dataset_name, model.__name__, training_subset_size, test_size, labeling]))

                    for statedict_fname in keep_statedict_fnames:
                        if dataset_name in natural_dataset_names:
                            chosen_classes = statedict_fname.split("_")[6:8]
                            chosen_classes = [int(c) for c in chosen_classes]
                            
                        # load data
                        if dataset_name in natural_dataset_names:
                            trainset, _ = get_datasets(dataset_name, 
                                                subset_size=training_subset_size+test_size, 
                                                test_size=test_size,
                                                labeling=labeling,
                                                img_size=img_size,
                                                class1=chosen_classes[0],
                                                class2=chosen_classes[1]
                                            )
                        else:
                            trainset, _ = get_datasets(dataset_name, 
                                                subset_size=training_subset_size+test_size, 
                                                test_size=test_size,
                                                labeling=labeling,
                                                img_size=img_size,
                                                special_medicalisrgb = dataset_name in special_rgb_medical_dataset_names
                                            )

                        trainset.labels = trainset.dataset.labels
                        intrinsic_dim = estimate_intrinsic_dim(trainset, dataset_name, intrinsic_dim_estimator, batchsize=estimator_batchsize)
                        
                        # log it
                        log_msg = ' '.join([
                            dataset_name,
                            str(training_subset_size),
                            labeling,
                            model.__name__,
                            intrinsic_dim_estimator,
                            str(intrinsic_dim),
                            str(img_size)
                        ])

                        logger.write_msg(log_msg)
