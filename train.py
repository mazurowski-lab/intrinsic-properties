"""
train multiple models on multiple datasets
"""
# imports
from src.vizutils import *
from src.utils import *
from src.dataset import *
from src.nnutils import *
import os
import random
from tqdm import tqdm
import datetime

# torch
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50, vgg13, vgg16, vgg19

# GPUs
device_ids = [0] # indices of devices for models, data and otherwise
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
print('running on {}'.format(device))

# settings
train_with_augmentation = []
change_model_output_logit = False

training_sizes = list(range(500, 1750+250, 250))

# random seed
seed = 1337
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset_names = ['brats', 'dbc', 'oai', 'chexpert', 'mura', 'rsna', 'prostate'] + ['MNIST', 'CIFAR10', 'SVHN', 'ImageNet']
#dataset_names = ['isic']

models = [vgg13, resnet18, vgg16, vgg19, resnet34, resnet50]

# hyperparams
epochs = 100
test_size = 750
img_size = 224

learning_rates = {
          'resnet18' : 0.001, 
          'resnet34' : 0.001, 
          'resnet50' : 0.001,
           'vgg13' : 0.0001, 
           'vgg16' : 0.0001, 
           'vgg19' : 0.0001, 
}

batch_size_factors = {
          'resnet18' : 200, 
          'resnet34' : 128, 
          'resnet50' : 64,
           'vgg13' : 32, 
           'vgg16' : 32, 
           'vgg19' : 32, 
           'squeezenet1_1' : 32, 
           'densenet121' : 32,
           'densenet169' : 32
}

labelings = ['default']
#labelings = ['Edema']

for labeling in labelings:
    for model in models:#
        for dataset_name in dataset_names: 
            for training_subset_size in training_sizes:
                checkpoint_path_prev = None

                # load dataset and loader
                train_batchsize = batch_size_factors[model.__name__] * len(device_ids) * special_batchsize_factor

                if dataset_name in train_with_augmentation:
                    train_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomResizedCrop(size=img_size)
                    ])
                    print('training with augmentations')
                else:
                    train_transform = transforms.Compose([])

                trainset, testset = get_datasets(dataset_name, 
                                                 subset_size=training_subset_size+test_size, 
                                                 test_size=test_size,
                                                 img_size=img_size,
                                                 labeling=labeling,
                                                 special_medicalisrgb = dataset_name in special_rgb_medical_dataset_names
                                                )

                # if natural dataset, binary class choices are random
                if dataset_name in natural_dataset_names:
                    chosen_classes = trainset.dataset.chosen_classes

                trainloader = DataLoader(trainset, 
                                         batch_size=train_batchsize // 5,
                                         shuffle=True)
                testloader = DataLoader(testset, 
                                        batch_size=64)


                N_train = len(trainset)
                N_test = len(testset)
                print('{} training data, {} testing data'.format(N_train, N_test))


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

                if change_model_output_logit:
                    change_net_output_logit_count(net, model, 1)

                net = net.to(device)
                net = torch.nn.DataParallel(net, device_ids = range(len(device_ids)))


                checkpoint_dir = "saved_models/generalization/{}".format(dataset_name)

                # loss and optim.
                if change_model_output_logit:
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    criterion = nn.CrossEntropyLoss()

                # Your code: use an optimizer
                lr = learning_rates[model.__name__]
                if 'vgg' in model.__name__ and dataset_name == "SVHN":
                    lr = 0.000001

                print("using learning rate of {}".format(lr))
                optimizer = torch.optim.Adam(net.parameters(),
                                             lr=lr,
                                             weight_decay=0.0001     
                                            )
                start_epoch = 0

                log_dir = 'logs/generalization/{}'.format(dataset_name)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                logname = "{}_{}_{}_{}".format(model.__name__, N_train, N_test, labeling)
                if dataset_name in natural_dataset_names:
                    logname += "_" + "_".join([str(c) for c in chosen_classes])
                logger = Logger('custom', log_dir, custom_name=logname)

                # training

                for epoch in range(start_epoch, epochs):
                    net.train()
                    logger.write_msg("Epoch {}:".format(epoch))

                    total_examples = 0
                    correct_examples = 0

                    train_loss = 0
                    train_acc = 0

                    # train for one epoch
                    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), 
                                                             total=len(trainloader.dataset)//train_batchsize):
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                        # apply transformations
                        inputs = train_transform(inputs)

                        # reset gradients
                        optimizer.zero_grad()

                        # inference
                        with torch.cuda.amp.autocast():
                            outputs = net(inputs)
                            if change_model_output_logit:
                                outputs = outputs.flatten()
                                targets = targets.float()

                            # backprop
                            loss = criterion(outputs, targets)
                            
                        loss.backward()

                        # iterate
                        optimizer.step()

                        # Calculate predicted labels
                        if change_model_output_logit:
                            total_examples += targets.size(0)
                            correct_examples += (torch.round(torch.sigmoid(outputs)) == targets).sum().item()

                        else:
                            _, predicted = outputs.max(1)
                            total_examples += predicted.size(0)
                            correct_examples += predicted.eq(targets).sum().item()
                        train_loss += loss

                    # results
                    avg_loss = train_loss / (batch_idx + 1)
                    avg_acc = correct_examples / total_examples
                    logger.write_msg("Training loss: %.4f, Training accuracy: %.4f" %(avg_loss, avg_acc))

                    logger.write_msg(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
                    logger.write_msg("Test...")
                    total_examples = 0
                    correct_examples = 0

                    net.eval()

                    test_loss = 0
                    test_acc = 0
                    with torch.no_grad():
                        for batch_idx, (inputs, targets) in enumerate(testloader):
                            # Copy inputs to device
                            inputs = inputs.to(device)
                            targets = targets.to(device)
                            # Generate output from the DNN.
                            outputs = net(inputs)
                            if change_model_output_logit:
                                outputs = outputs.flatten()
                                targets = targets.float()

                            loss = criterion(outputs, targets)            

                            # Calculate predicted labels
                            if change_model_output_logit:
                                total_examples += targets.size(0)
                                correct_examples += (torch.round(torch.sigmoid(outputs)) == targets).sum().item()

                            else:
                                _, predicted = outputs.max(1)
                                total_examples += predicted.size(0)
                                correct_examples += predicted.eq(targets).sum().item()
                            test_loss += loss

                    avg_loss = test_loss / len(testloader)
                    avg_acc = correct_examples / total_examples

                    logger.write_msg("Test loss: %.4f, Test accuracy: %.4f" % (avg_loss, avg_acc))

                    # Save checkpoint
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    logger.write_msg("Saving ...")
                    state = {'net': net.state_dict(),
                             'epoch': epoch}

                    # delete older checkpoint
                    if checkpoint_path_prev:
                        os.remove(checkpoint_path_prev)

                    # save new checkpoint
                    checkpoint_path = "{}_{}_{}_{}_{}_{}".format(model.__name__,
                        N_train, N_test, avg_acc, epoch, labeling)
                    if dataset_name in natural_dataset_names:
                        checkpoint_path += "_" + "_".join([str(c) for c in chosen_classes])

                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
                    torch.save(state, checkpoint_path)

                    checkpoint_path_prev = checkpoint_path