"""
utilities for creating and loading datasets
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from skimage import io
import pandas as pd
import numpy as np
import os
from random import sample
from tqdm import tqdm
from PIL import Image

# random seed
seed = 1337

# constants
label_csvs = {
    'chexpert' :'data/chexpert/CheXpert-v1.0/train_subset.csv',
    'isic' : 'data/isic/ISIC_2019_Training_GroundTruth.csv',
    'mura' :'data/mura/MURA-v1.1/train_image_paths.csv',
    'rsna' : 'data/rsna/stage_2_train.csv'
}

data_dirs = {
    'chexpert' : 'data/chexpert/CheXpert-v1.0/subset/train',
    'dbc' : 'data/dbc/png_subset',
    'oai' : 'data/oai/OAI_img',
    'brats' : 'data/brats/Brats_normalized/flair',
    'mura' : 'data/mura/',
    'isic' : 'data/isic/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
    'rsna' : 'data/rsna/stage_2_train_png',
    'prostate' : 'data/prostate/train_png'
}

natural_dataset_names = ['MNIST', 'CIFAR10', 'SVHN', 'ImageNet']
medical_dataset_names = ['brats', 'dbc', 'oai', 'chexpert', 'mura', 'rsna', 'prostate']
special_rgb_medical_dataset_names = ['isic']


cur_dir = os.getcwd()
for dct in label_csvs, data_dirs:
    dct = {k: os.path.join(cur_dir, v) for k, v in dct.items()}

class NaturalDatasetBinary(Dataset):
    def __init__(self, dataset_name, img_size, split, transform=None, class1=None, class2=None, min_size=None):
        self.img_size = img_size
        self.transform = transform

        assert split in ["train", "test"]

        if dataset_name == 'MNIST':
            self.dataset = datasets.MNIST(root='data', train=(split=="train"), download=True)
            self.dataset.data = self.dataset.data.numpy()
        elif dataset_name == 'CIFAR10':
            self.dataset = datasets.CIFAR10(root='data', train=(split=="train"), download=True)
        elif dataset_name == 'SVHN':
            self.dataset = datasets.SVHN(root='data', split=split, download=True)
            self.dataset.targets = self.dataset.labels
        elif dataset_name == 'ImageNet':
            # -v if data downloaded from imagenet site
            #self.dataset = datasets.ImageNet(root='data/imagenet', split='train')
            # -v if data downloaded from kaggle
            self.dataset = datasets.ImageFolder('data/ILSVRC/Data/CLS-LOC/{}'.format("train" if split == "train" else "val"))
        else:
            raise NotImplementedError
        
        self.dataset_name = dataset_name

        if dataset_name == 'ImageNet':
            all_classes = range(1000)
        else:
            all_classes = range(10)
        all_classes = list(all_classes)


        # load only two classes
        if class1 is None and class2 is None:
            # random
            chosen_classes = sample(all_classes, 2)
            print('Chosen classes: ', chosen_classes)
        else:
            chosen_classes = [class1, class2]

        # sample only these classes from dset
        chosen_indices = [i for i, label in enumerate(self.dataset.targets) if int(label) in chosen_classes]

        if min_size is not None:
            # make sure there is enough data for these class choices
            # otherwise choose new classes
            if len(chosen_indices) < min_size:
                if class1 is None and class2 is None:
                    while len(chosen_indices) < min_size:
                        chosen_classes = sample(all_classes, 2)
                        print('Chosen classes: ', chosen_classes)
                        chosen_indices = [i for i, label in enumerate(self.dataset.targets) if int(label) in chosen_classes]

                else:
                    raise RuntimeError("chosen classes don't have enough examples for the subset size you want")

        # save chosen classes
        self.chosen_classes = chosen_classes
#
        if dataset_name != 'ImageNet':
            # imagenet too large to store in memory
            self.dataset.data = self.dataset.data[chosen_indices]

        self.dataset.targets = [self.dataset.targets[i] for i in chosen_indices] 
        
        # for usage with functions designed for MedicalDataset:
        # also convert to binary labels of 0 and 1
        # empty filepath
        self.labels = []
        for idx, label in enumerate(self.dataset.targets):
            if int(label) == chosen_classes[0]:
                if dataset_name == 'ImageNet':
                    # need image file path for imagenet
                    self.labels.append((self.dataset.samples[idx][0], 0))
                else:
                    self.labels.append((None, 0))
            elif int(label) == chosen_classes[1]:
                if dataset_name == 'ImageNet':
                    # need image file path for imagenet
                    self.labels.append((self.dataset.samples[idx][0], 1))
                else:
                    self.labels.append((None, 1))
            else:
                raise ValueError

                 
    def normalize(self, img):
        return MedicalDataset.normalize(self, img)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.dataset_name == 'ImageNet':
            # load from file
            img_arr = np.array(Image.open(self.labels[idx][0]).convert('RGB'))
            # convert to tensor from PIL
        else:
            img_arr = self.dataset.data[idx]

        if self.dataset_name in ["ImageNet", "CIFAR10"]:
            # need to rearrange/transpose
            img_arr = np.transpose(img_arr, (2, 0, 1))

        # normalize
        img_arr = self.normalize(img_arr)

        # convert to tensor
        data = torch.from_numpy(img_arr)
        data = data.type(torch.FloatTensor) 
        
        # add channel dim if needed
        if len(data.shape) == 2:
            data = torch.unsqueeze(data, 0)

        # resize img to standard dimensionality
        data = transforms.Resize((self.img_size, self.img_size))(data)
        # bilinear by default
        
        # do any data augmentation/training transformations
        if self.transform:
            data = self.transform(data)
        
        target = self.labels[idx][1]

        return data, target

class MedicalDataset(Dataset):
    def __init__(self, label_csv, data_dir, img_size, transform, make_3_channel=False):
        self.label_csv = label_csv
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        
        self.make_3_channel = make_3_channel
        
        # to be initialized by child class
        self.labels = None

        # special
        self.special_isrgb = False
                 
    def normalize(self, img):
        # normalize to range [0, 255]
        # img expected to be array
                 
        # uint16 -> float
        img = img.astype(float) * 255. / img.max()
        # float -> unit8
        img = img.astype(np.uint8)
        
        return img
    
    def __getitem__(self, idx):
        
        fpath, target  = self.labels[idx]
        
        # print(fpath)
        
        # load img from file (png or jpg)
        if self.special_isrgb:
            img_arr = io.imread(fpath)
            img_arr = np.transpose(img_arr, (2,0,1))
        else:
            img_arr = io.imread(fpath, as_gray=True)
        
        # normalize
        img_arr = self.normalize(img_arr)
        
        # convert to tensor
        data = torch.from_numpy(img_arr)
        data = data.type(torch.FloatTensor) 
       
        # add channel dim
        if not self.special_isrgb:
            data = torch.unsqueeze(data, 0)
        
        # resize to standard dimensionality
        data = transforms.Resize((self.img_size, self.img_size))(data)
        # bilinear by default
        
        # make 3-channel (testing only)
        if self.make_3_channel and not self.special_isrgb:
            data = torch.cat([data, data, data], dim=0)
        
        # do any data augmentation/training transformations
        if self.transform:
            data = self.transform(data)
        
        return data, target
    
    def __len__(self):
        return len(self.labels)
    
    def get_avg_extrinsic_dim(self):
        EDs = []
        for label in tqdm(self.labels):
            fpath, _  = label
            img_arr = io.imread(fpath, as_gray=True)
            EDs.append(img_arr.size)
            
        return np.mean(EDs)
    
    def resize_imgs_on_disk(self, targetsize=224):
        print('resizing images to {}x{}...'.format(targetsize,targetsize))
        for label in tqdm(self.labels):
            fpath, _  = label
            
            im = Image.open(fpath)
            im = im.resize((targetsize, targetsize), Image.BILINEAR)
            im.save(fpath)

# individual datasets                   
class CheXpertDataset(MedicalDataset):
    def __init__(self, img_size, labeling='Pleural Effusion', train_transform=None, make_3_channel=False):
        super(CheXpertDataset, self).__init__(label_csvs['chexpert'], data_dirs['chexpert'], img_size, train_transform, make_3_channel=make_3_channel)
        
        label_df = pd.read_csv(self.label_csv)
        labels = [] 
        # (fname, value = label (0 = neg, 1 = pos) )
        
        pos_ct = 0
        neg_ct = 0
        
        print('building CheXpert dataset.')
        for row_idx, row in label_df.iterrows():
            fname = row['New Path']
            fpath = os.path.join(self.data_dir, fname)
            
            target = row[labeling]
            if np.isnan(target):
                target = 0
                neg_ct += 1
            elif int(target) == -1:
                target = 0 
                neg_ct += 1
            else:
                target = 1
                pos_ct += 1
            assert target in [0, 1]
            
            labels.append((fpath, target))
            
        self.labels = labels
        print('{} positives, {} negatives.'.format(pos_ct, neg_ct))
        
        
class DBCDataset(MedicalDataset):
    def __init__(self, img_size, labeling='default', train_transform=None, make_3_channel=False, unique_patients=False):
        super(DBCDataset, self).__init__(None, data_dirs['dbc'], img_size, train_transform, make_3_channel=make_3_channel)

        labels = []
        patient_IDs_used = []
        # (fname, value = label (0 = neg, 1 = pos) )
        print('building DBC dataset.')
        if labeling == 'default':
            for target, target_label in enumerate(['neg', 'pos']):
                case_dir = os.path.join(self.data_dir, target_label)
                for fname in os.listdir(case_dir):
                    if '.png' in fname:
                        
                        patient_ID = fname.split('-')[2].replace('.png', '')
                        if unique_patients:
                            # if we only one one datapoint per patient
                            if patient_ID in patient_IDs_used:
                                continue
                            else:
                                patient_IDs_used.append(patient_ID)
                        
                        fpath = os.path.join(case_dir, fname)
                        labels.append((fpath, target))
        else:
            raise NotImplementedError
            
        self.labels = labels
         
            
class OAIDataset(MedicalDataset):
    def __init__(self, img_size, labeling='default', train_transform=None, make_3_channel=False):
        super(OAIDataset, self).__init__(None, data_dirs['oai'], img_size, train_transform, make_3_channel=make_3_channel)

        labels = []
        # (fname, value = label (0 = neg, 1 = pos) )
        print('building OAI dataset.')
        if labeling == 'default':
            for target in range(2):
                case_dir = os.path.join(self.data_dir, str(target))
                for fname in os.listdir(case_dir):
                    if '.png' in fname:
                        fpath = os.path.join(case_dir, fname)
                        labels.append((fpath, target))
        else:
            raise NotImplementedError
            
        self.labels = labels
        
        
class BraTSDataset(MedicalDataset):
    def __init__(self, img_size, labeling='default', train_transform=None, make_3_channel=False):
        super(BraTSDataset, self).__init__(None, data_dirs['brats'], img_size, train_transform, make_3_channel=make_3_channel)

        labels = []
        # (fname, value = label (0 = neg, 1 = pos) )
        print('building BraTS dataset.')
        if labeling == 'default':
            for target in range(2):
                case_dir = os.path.join(self.data_dir, str(target))
                for fname in os.listdir(case_dir):
                    if '.jpg' in fname:
                        fpath = os.path.join(case_dir, fname)
                        labels.append((fpath, target))
        else:
            raise NotImplementedError
            
        self.labels = labels
        
        
class MURADataset(MedicalDataset):
    def __init__(self, img_size, labeling='default', train_transform=None, make_3_channel=False):
        super(MURADataset, self).__init__(label_csvs['mura'], data_dirs['mura'], img_size, train_transform, make_3_channel=make_3_channel)
        
        label_df = pd.read_csv(self.label_csv, names=['path'])
        labels = [] 
        # (fname, value = label (0 = neg, 1 = pos) )
        print('building MURA dataset')
        for row_idx, row in label_df.iterrows():
            fname = row['path']
            fpath = os.path.join(self.data_dir, fname)
            
            target = None
            if 'negative' in fname:
                target = 0
            elif 'positive' in fname:
                target = 1
                
            assert target in [0, 1]
            
            labels.append((fpath, target))
            
        self.labels = labels
        
# NV == Melanocytic nevus
class ISICDataset(MedicalDataset):
    def __init__(self, img_size, labeling='NV', train_transform=None, make_3_channel=False):
        super(ISICDataset, self).__init__(label_csvs['isic'], data_dirs['isic'], img_size, train_transform, make_3_channel=make_3_channel)
        
        label_df = pd.read_csv(self.label_csv)
        labels = [] 
        # (fname, value = label (0 = neg, 1 = pos) )
        
        pos_ct = 0
        neg_ct = 0
        
        print('building ISIC dataset.')
        for row_idx, row in label_df.iterrows():
            fname = row['image'] + '.jpg'
            fpath = os.path.join(self.data_dir, fname)
            
            target = int(row[labeling])
                
            if target == 0:
                neg_ct += 1
            elif target == 1:
                pos_ct += 1
                
            assert target in [0, 1]
            
            labels.append((fpath, target))
            
        self.labels = labels
        print('{} positives, {} negatives.'.format(pos_ct, neg_ct))
        
# RSNA intracranial hemorrhage brain CT
class RSNADataset(MedicalDataset):
    def __init__(self, img_size, labeling='default', train_transform=None, make_3_channel=False):
        super(RSNADataset, self).__init__(label_csvs['rsna'], data_dirs['rsna'], img_size, train_transform, make_3_channel=make_3_channel)
        
        
        # default labeling  = 0 for no hemorrage, 1 for any hemorrage
        if labeling == 'default':
            labeling = 'any'
        label_df = pd.read_csv(self.label_csv)
        label_df = label_df[label_df['ID'].str.contains(labeling)] 
        # get rid of irrelevant labels
        
        labels = [] 
        # (fname, value = label (0 = neg, 1 = pos) )
        print('building RSNA dataset')
        for row_idx, row in label_df.iterrows():
            fname = row['ID'].split('_')[:2]
            fname = '_'.join(fname) + '.png'
            fpath = os.path.join(self.data_dir, fname)
            
            target = int(row['Label'])         
            assert target in [0, 1]
            
            labels.append((fpath, target))
            
        self.labels = labels
        

class ProstateMRIDataset(MedicalDataset):
    def __init__(self, img_size, labeling='default', train_transform=None, make_3_channel=False):
        super(ProstateMRIDataset, self).__init__(None, data_dirs['prostate'], img_size, train_transform, make_3_channel=make_3_channel)
        
        labels = []
        # (fname, value = label (0 = neg, 1 = pos) )
        print('building ProstateMRI dataset.')
        for score in range(6):
            case_dir = os.path.join(self.data_dir, str(score))

            if labeling == 'default':
                if score < 2:
                    target = 0
                else:
                    target = 1
                    
                    
            elif labeling == 'hard':
                # negative is least sever cancer, positive is more sever
                if score == 2:
                    target = 0
                elif score > 2:
                    target = 1
                else:
                    continue
            else:
                raise NotImplementedError
                    
            for fname in os.listdir(case_dir):
                if '.png' in fname:
                    fpath = os.path.join(case_dir, fname)
                    labels.append((fpath, target))
            
        self.labels = labels




# utils
def get_datasets(dataset_name, labeling='default', subset_size=None, train_frac=None, 
                 test_size=None, val_size=None, img_size=224, make_3_channel=False,
                 class1=None, class2=None, split=None,
                 unique_DBC_patients=False, special_medicalisrgb=False):
    # either (1) specify train_frac, which split of subset to create train and test sets, or
    # (2) specify test_size

    if split is not None:
        assert dataset_name in natural_dataset_names, "predefined splits only given for natural datasets."

    if labeling != 'default':
        print('using non-default {} labeling.'.format(labeling))

    # first, option of getting subset of full dataset stored
    # then, option of splitting what's left into train and test
    # create dataset
    if dataset_name == 'chexpert':
        # default labeling is by Pleural Effusion state
        if labeling == 'default':
            labeling = 'Pleural Effusion'
        dataset = CheXpertDataset(img_size, labeling, make_3_channel=make_3_channel)
    elif dataset_name == 'dbc':
        dataset = DBCDataset(img_size, labeling, make_3_channel=make_3_channel, unique_patients=unique_DBC_patients)
    elif dataset_name == 'oai':
        dataset = OAIDataset(img_size, labeling, make_3_channel=make_3_channel)
    elif dataset_name == 'brats':
        dataset = BraTSDataset(img_size, labeling, make_3_channel=make_3_channel)
    elif dataset_name == 'mura':
        dataset = MURADataset(img_size, labeling, make_3_channel=make_3_channel)
    elif dataset_name == 'isic':
        if labeling == 'default':
            labeling = 'NV'
        dataset = ISICDataset(img_size, labeling, make_3_channel=make_3_channel)
    elif dataset_name == 'rsna':
        dataset = RSNADataset(img_size, labeling, make_3_channel=make_3_channel)
    elif dataset_name == 'prostate':
        dataset = ProstateMRIDataset(img_size, labeling, make_3_channel=make_3_channel)
    # natural images
    elif dataset_name in natural_dataset_names:
        # default behavior: choose 2 random classes to sample from
        if split is None:
            split = "train"
        dataset = NaturalDatasetBinary(dataset_name, img_size, min_size=subset_size, class1=class1, class2=class2, split=split)
    else:
        raise NotImplementedError

    if not dataset_name in natural_dataset_names:
        dataset.special_isrgb = special_medicalisrgb
        
    if subset_size:
        # each class should have subset_size//2 instances
        # so remove extras
        pos_ct = 0
        neg_ct = 0
        class_size = subset_size//2
        new_labels = []
        for idx, label in enumerate(dataset.labels):
            if label[1] == 1 and pos_ct < class_size:
                new_labels.append(label)
                pos_ct += 1
            elif label[1] == 0 and neg_ct < class_size:
                new_labels.append(label)
                neg_ct += 1
                
        assert len(new_labels) == subset_size
        dataset.labels = new_labels
        print('{} positive examples, {} negative examples.'.format(pos_ct, neg_ct))
            
    # split into train and test if chosen

    # or just one dataset
    if ((not train_frac) and (val_size is None) and (test_size is None)):
        return dataset

    if train_frac:
        train_size = int(train_frac * len(dataset))
        test_size = (len(dataset) - train_size) // 2
        val_size = test_size

    if val_size is not None:
        train_size = len(dataset) - test_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))
        return train_dataset, val_dataset, test_dataset

    else:
        train_size = len(dataset) - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
        return train_dataset, test_dataset