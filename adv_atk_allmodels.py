"""
perform and evaluate adversarial attacks on a bunch of trained models.
"""
# imports
from src.vizutils import *
from src.utils import *
from src.dataset import *
from src.nnutils import *
from src.adversarial_utils import *
import os
import random
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

# torch
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, vgg13, vgg16, vgg19
from torchvision.utils import make_grid

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
img_size = 224
test_size = 750
test_batchsize = 256

training_sizes = list(range(500, 1750+250, 250))

dataset_names = ['brats', 'dbc', 'prostate', 'rsna', 'chexpert', 'mura', 'oai'] + ['MNIST', 'CIFAR10', 'SVHN', 'ImageNet']
#dataset_names = ['isic']

models = [resnet18, resnet34, resnet50, vgg13, vgg16, vgg19]

# for medical image datasets, choose which diagnostic task if applicable (see dataset.py for more info)
labeling = 'default'

attack_type = "FGSM"
criterion = nn.CrossEntropyLoss()
all_atk_eps = [2]
# NOTE: since images are in [0, 255], the eps would be atk_eps/255 is scaled to [0, 1]

# logger
log_dir = 'logs/adv_atk/allmodels_alldata'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger = Logger('custom', log_dir)
header = ' '.join([
    'dataset',
    'n_train',
    'labeling',
    'model',
    'atk_type',
    'atk_eps',
    'clean_loss',
    'clean_acc',
    'atk_loss',
    'atk_acc',
    'class0',
    'class1'
])
logger.write_msg(header)

plot_eg_atks = False
if plot_eg_atks:
    eg_atks = torch.empty((2, len(dataset_names), 1, img_size, img_size))
    eg_clean_accs = []
    eg_atk_accs = []

for atk_eps in all_atk_eps:
    for model in models:
        for dset_idx, dataset_name in enumerate(dataset_names):
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
                    else:
                        chosen_classes = [0, 1]
                        
                    print(statedict_fname)
                    # load data
                    if dataset_name in natural_dataset_names:
                        _, testset = get_datasets(dataset_name, 
                                            subset_size=training_subset_size+test_size, 
                                            test_size=test_size,
                                            img_size=img_size,
                                            labeling=labeling,
                                            class1=chosen_classes[0],
                                            class2=chosen_classes[1]
                                        )
                    else:
                        _, testset = get_datasets(dataset_name, 
                                            subset_size=training_subset_size+test_size, 
                                            test_size=test_size,
                                            img_size=img_size,
                                            labeling=labeling,
                                            special_medicalisrgb = dataset_name in special_rgb_medical_dataset_names
                                        )

                    testloader = DataLoader(testset, 
                            batch_size=test_batchsize)

                    # load model
                    net = model()
                    # fix first lyr if one channel needed
                    if dataset_name in natural_dataset_names and dataset_name != "MNIST":
                        print("net takes 3 channels as input.")
                    elif dataset_name in special_rgb_medical_dataset_names:
                        print("net takes 3 channels as input.")
                    else:
                        print("making net take 1 input channel.")
                        make_netinput_onechannel(net, model)

                    net = net.to(device)
                    net = torch.nn.DataParallel(net, device_ids = range(len(device_ids)))
                    
                    # load statedict
                    statedict = torch.load(os.path.join(checkpoint_dir, statedict_fname))
                    try:
                        net.load_state_dict(statedict)
                    except RuntimeError:
                        net.load_state_dict(statedict["net"])

                    net.eval()

                    # EVALUATION:

                    # get clean acc/loss
                    total_examples = 0
                    correct_examples = 0
                    test_loss = 0
                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(testloader):
                            # Copy inputs to device
                            inputs = inputs.to(device)
                            targets = targets.to(device)

                            # Generate output from the DNN.
                            outputs = net(inputs)
                            loss = criterion(outputs, targets)            
                            # Calculate predicted labels
                            _, predicted = outputs.max(1)
                            total_examples += predicted.size(0)
                            correct_examples += predicted.eq(targets).sum().item()
                            test_loss += loss.item()

                    clean_loss = test_loss / len(testloader)
                    clean_acc = correct_examples / total_examples

                    # get adv accuracy
                    total_examples = 0
                    correct_examples = 0
                    test_loss = 0
                    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
                        # Copy inputs to device
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        inputs.requires_grad = True

                        # generate attack
                        adv_images = generate_attack(attack_type, net, criterion, inputs, targets, atk_eps).detach()
                        outputs = net(adv_images).detach()
                        loss = criterion(outputs, targets)

                        
                        if plot_eg_atks:
                            if batch_idx == 0:
                                #for j in range(5):
                                j = 4
                                #eg_atks += [inputs[j], adv_images[j]]
                                eg_atks[:, dset_idx, :, :, :] = torch.stack((inputs[j], adv_images[j]))
                                #quicksave_image(inputs[0], "clean{}.png".format(batch_idx))
                                #quicksave_image(adv_images[0], "atk{}.png".format(batch_idx))

                        # Calculate predicted labels
                        _, predicted = outputs.max(1)
                        total_examples += predicted.size(0)
                        correct_examples += predicted.eq(targets).sum().item()
                        test_loss += loss.item()

                    atk_loss = test_loss / len(testloader)
                    atk_acc = correct_examples / total_examples

                    # log it
                    log_msg = ' '.join([
                        dataset_name,
                        str(training_subset_size),
                        labeling,
                        model.__name__,
                        attack_type,
                        str(atk_eps),
                        str(clean_loss),
                        str(clean_acc),
                        str(atk_loss),
                        str(atk_acc),
                        str(chosen_classes[0]),
                        str(chosen_classes[1])
                    ])
                    
                    if plot_eg_atks:
                        eg_clean_accs.append(clean_acc)
                        eg_atk_accs.append(atk_acc)

                    logger.write_msg(log_msg)

if plot_eg_atks:
    # create example attack plot
    nrow = len(dataset_names)
    eg_atks = eg_atks.reshape(nrow*2, *eg_atks.shape[2:])
    #save_image((eg_atks - eg_atks.min()) / (eg_atks.max() - eg_atks.min()), "results/imgs/eg_atks.png", nrow=nrow) 
    eg_atks_norm = (eg_atks - eg_atks.min()) / (eg_atks.max() - eg_atks.min())
    grid = make_grid(eg_atks_norm, nrow=nrow)

    plt.figure(figsize=(nrow*2, 2*2))
    plt.imshow(torch.permute(grid, (1, 2, 0)).cpu(), cmap="gray")
    plt.axis('off')

    # plot accuracies
    for dset_idx in range(len(dataset_names)):
        for i in range(2):
            plt.text(20 + dset_idx * (img_size + 2),
                    30 + i*(img_size + 2),
                    "acc. = {}%".format(round(100 * [eg_clean_accs, eg_atk_accs][i][dset_idx], 2)),
                    color="k",
                    weight="bold",
                    bbox=dict(facecolor=['cornflowerblue', 'lightcoral'][i], edgecolor='w')
                    )
    outdir = "results/imgs"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig("results/imgs/eg_atks.pdf", bbox_inches = "tight")
