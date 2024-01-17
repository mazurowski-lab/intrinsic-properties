"""
miscellaneous/general purpose utilities
"""
import os
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image

# general utils
def find_model_checkpoint(
    checkpoint_dir,
    model_name,
    train_size,
    test_size,
    labeling
    ):

    statedict_fname = None
    statedict_fnames = [f for f in os.listdir(checkpoint_dir) if f.endswith(".h5")]
    for fn in statedict_fnames:
        fn_split = fn.split("_")
        if (fn_split[:3] == [model_name, str(train_size), str(test_size)]) and (fn_split[5:] == [labeling, "best.h5"]):
            statedict_fname = fn
            break

    return statedict_fname


# visualization
def plot_imgbatch(imgs, nrow=5):
    imgs = imgs.cpu()
    imgs = imgs.type(torch.IntTensor) 
    plt.figure(figsize=(15, 3*(imgs.shape[0])))
    grid_img = make_grid(imgs, nrow=nrow)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

def quicksave_image(img_tensor, fname="img.png"):
    # convert to [0, 1]
    save_image(img_tensor.float() / 255, fname)


# logging
class Logger():
    def __init__(self, mode, log_dir, custom_name=''):
        assert mode in ['custom']
        self.mode = mode
        
        # create log file
        now = datetime.now()
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        logfname = 'log_{}_{}.txt'.format(custom_name, now.strftime("%m-%d-%Y_%H:%M:%S"))
        self.logfname = os.path.join(log_dir, logfname)
        print(self.logfname)
        
        with open(self.logfname, 'w') as fp: # create file
            pass
        
        # log intro message
        start_msg = 'beginning {} on {}.\n'.format(self.mode, now.strftime("%m/%d/%Y, %H:%M:%S"))

            
        if mode == 'custom':
            start_msg += '--------------------------\n'
            start_msg += 'custom log.\n'
        
        self.write_msg(start_msg)
        print(start_msg)
        
    def write_msg(self, msg, print_=True):
        if print_:
            print(msg)
            
        if not msg.endswith('\n'):
            msg += '\n'
            
        log_f = open(self.logfname, 'a')
        log_f.write(msg)
        log_f.close()
        
        return
