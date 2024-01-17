"""
various utils for working with PyTorch.nn modules
"""
import numpy as np
from torch import nn
from torch.utils.data import Dataset
import os

def get_net_checkpoint(model, train_dataset_name, N_train, N_test=750, trainset_labeling='default',
                      suppress_nomodel_warning=False):
    """
    get saved net checkpoint following certain params
    
    args:
        model: torch.nn Module. model class, e.g. resnet18.
        train_dataset_name: str. name of dataset used to train the model, e.g. dbc.
        N_train: int. size of train set.
        trainset_labeling: str. name of labeling of classification task used to train model
    """
    model_name = model.__name__
    model_dir = 'saved_models/generalization'
    ckpts_path = os.path.join(model_dir, train_dataset_name)
    
    ckpt_fnames = []
    for fname in os.listdir(ckpts_path):
        if fname.endswith('.h5'):
            fname_splt = fname.split('_')
            
            old_fname_format = False
            # for old fname format, no explicit mention of labeling as default
            if 'squeezenet' in fname:
                if len(fname_splt) == 7:
                    old_fname_format = True # false if 8
            else:
                if len(fname_splt) == 6:
                    old_fname_format = True # false if 7
            
            if 'squeezenet' in fname:
                # fixes issue with squeezenet model name including '_' delimiter
                correct_crossref_modelname = '_'.join([fname_splt[0], fname_splt[1]])
                if old_fname_format:
                    correct_crossref_labeling = 'default'
                else:
                    correct_crossref_labeling = fname_splt[6]
                    
                crossref_items = [
                    correct_crossref_modelname, 
                                  int(fname_splt[2]), 
                                  int(fname_splt[3]), 
                                  correct_crossref_labeling
                                 ]
            else:
                if old_fname_format:
                    correct_crossref_labeling = 'default'
                else:
                    correct_crossref_labeling = fname_splt[5]
                    
                crossref_items = [
                    fname_splt[0], 
                    int(fname_splt[1]), 
                    int(fname_splt[2]), 
                    correct_crossref_labeling
                ]
            
            target_items = [model_name, N_train, N_test, trainset_labeling]
            if crossref_items == target_items:
                ckpt_fnames.append(fname)
            
            
    if len(ckpt_fnames) == 0:
        if not suppress_nomodel_warning:
            print('WARNING: No saved models found for these settings: {} {} {} {}.'.format(
                model.__name__, 
                train_dataset_name, 
                N_train, 
                trainset_labeling
            ))
        return 0
    elif len(ckpt_fnames) > 1:
        print('WARNING: Multiple models found for these settings: {} {} {} {}.'.format(
            model.__name__, 
            train_dataset_name, 
            N_train, 
            trainset_labeling
        ))
        return 1
    
    ckpt_path = os.path.join(ckpts_path, ckpt_fnames[0])
    
    return ckpt_path

class LayerActivationsDataset(Dataset):
    """
    dataset for batch-loading outputs of some layer of a neural network,
    given a dataset of activations (layer outputs) given input to the network.
    
    args:
        TODO
    """
    def __init__(self, activation_data, labels):
        self.activation_data = activation_data
        self.labels = labels
 
    def __len__(self):
        return self.activation_data.shape[0]
        
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.activation_data[idx], self.labels[idx]#[1]
        else:
            return self.activation_data[idx]

def make_netinput_onechannel(net, model):
    # fix nets to take one channel as input
    name = model.__name__
    if 'resnet' in name:
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif 'vgg' in name:
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    elif 'squeezenet' in name:
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))
    elif 'densenet' in name:
        net.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif 'alexnet' in name:
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    else:
        raise NotImplementedError

def change_net_output_logit_count(net, model, n_logit):
    # fix nets to take one channel as input
    name = model.__name__
    if name in ["resnet18", "resnet34"]:
        net.fc = nn.Linear(in_features=512, out_features=n_logit, bias=True)
    elif name == "resnet50":
        net.fc = nn.Linear(in_features=2048, out_features=n_logit, bias=True)
    elif "vgg" in name:
        net.classifier[6] = nn.Linear(in_features=4096, out_features=n_logit, bias=True)
    else:
        raise NotImplementedError

def get_repr_layer_depth(model_name, mode="ultimate"):
    name = model_name
    if mode == "ultimate":
        if ('resnet' in name) or ('vgg' in name):
            layer_depth = int(name[-2:]) + 1
        else:
            raise NotImplementedError
    elif mode == "penultimate":
        if ('resnet' in name):
            layer_depth = int(name[-2:]) - 1
        elif ('vgg' in name):
            layer_depth = int(name[-2:])
        else:
            raise NotImplementedError

        
    return layer_depth
        
#-------------------------------------------------------------------------------------------
# neural network layer utils; 
# everything in this enclosure taken directly from https://github.com/ansuini/IntrinsicDimDeep/blob/master/scripts/pretrained/hunchback.py

archs_ansuinietal = ['alexnet', 'vgg11', 'vgg13', 'vgg16','vgg19',
                    'vgg11_bn', 'vgg13_bn', 'vgg16_bn','vgg19_bn',
                    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

def getDepths(model):    
    count = 0    
    modules = []
    names = []
    depths = []    
    modules.append('input')
    names.append('input')
    depths.append(0)    
    
    for i,module in enumerate(model.features):       
        name = module.__class__.__name__
        if 'Conv2d' in name or 'Linear' in name:
            count += 1
        if 'MaxPool2d' in name:
            modules.append(module)
            depths.append(count)
            names.append('MaxPool2d')            
    for i,module in enumerate(model.classifier):
        name = module.__class__.__name__
        if 'Linear' in name:
            modules.append(module)    
            count += 1
            depths.append(count + 1)
            names.append('Linear')                       
    depths = np.array(depths)   
    return modules, names, depths

def getLayerDepth(layer):
    count = 0
    for m in layer:
        for c in m.children():
            name = c.__class__.__name__
            if 'Conv' in name:
                count += 1
    return count

def getResNetsDepths(model):    
    modules = []
    names = []
    depths = []  
    
    # input
    count = 0
    modules.append('input')
    names.append('input')
    depths.append(count)           
    # maxpooling
    count += 1
    modules.append(model.maxpool)
    names.append('maxpool')
    depths.append(count)     
    # 1 
    count += getLayerDepth(model.layer1)
    modules.append(model.layer1)
    names.append('layer1')
    depths.append(count)         
    # 2
    count += getLayerDepth(model.layer2)
    modules.append(model.layer2)
    names.append('layer2')
    depths.append(count)      
    # 3
    count += getLayerDepth(model.layer3)
    modules.append(model.layer3)
    names.append('layer3')
    depths.append(count)     
    # 4 
    count += getLayerDepth(model.layer4)
    modules.append(model.layer4)
    names.append('layer4')
    depths.append(count)      
    # average pooling
    count += 1
    modules.append(model.avgpool)
    names.append('avgpool')
    depths.append(count)     
    # output
    count += 1
    modules.append(model.fc)
    names.append('fc')
    depths.append(count)                      
    depths = np.array(depths)    
    return modules, names, depths

#-------------------------------------------------------------------------------------------

def getNetDepths(net, model):
    """ 
    convenience function for using above code
    to get information for studying net layers and depth.
    
    args:
        net: PyTorch nn.Module instance.
        model: specific model class used to instantiate net (different than net.__class__)
    
    returns:
        modules: list of torch.nn modules (layers)
        names: list of the names of these modules (strings)
        depths: list of depths of each module/layer in net (ints)
    """
    
    if 'resnet' in model.__name__:
        modules, names, depths = getResNetsDepths(net)
    else:
        modules, names, depths = getDepths(net)
        
    return modules, names, depths
        
def get_activation_layers(net, model):
    """
    get layers from a net that we want to analyze the activations of,
    dependent on the type of the specific net architecture (model).
    
    args:
        net: PyTorch nn.Module instance.
        model: specific model class used to instantiate net (different than net.__class__)
    """
    
    # follow same procedure as Ansuini et al. 2019:

    # model_name = model.__name__
    # if model_name in archs_ansuinietal:
    #     # follow same procedure as Ansuini et al. 2019
    #     # for archs that they defined
    layers, layer_names, layer_depths = getNetDepths(net, model)
        
    # [print(m, '\n\n++++++++++++++++++++++++\n\n') for m in modules]
    # [print(p) for p in [names, depths]]
        
    return layers, layer_names, layer_depths
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
