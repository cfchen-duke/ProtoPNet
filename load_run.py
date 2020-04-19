##### MODEL AND DATA LOADING
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from dataHelper import DatasetFolder
import re
import numpy as np
import os
import copy
from skimage.transform import resize
from helpers import makedir, find_high_activation_crop
import model
import push
import train_and_test as tnt
import save
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse


# load the model
check_test_accu = True

load_model_dir = '/usr/project/xtmp/ct214/saved_models/vgg16/thresholdlogits25_spiculated_with_negs_0415//'
load_model_name = '40_7push0.8829.pth'

#if load_model_dir[-1] == '/':
#    model_base_architecture = load_model_dir.split('/')[-3]
#    experiment_run = load_model_dir.split('/')[-2]
#else:
#    model_base_architecture = load_model_dir.split('/')[-2]
#    experiment_run = load_model_dir.split('/')[-1]

model_base_architecture = load_model_dir.split('/')[-3]
experiment_run = load_model_dir.split('/')[-2]



load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

print('load model from ' + load_model_path)
print('model base architecture: ' + model_base_architecture)
print('experiment run: ' + experiment_run)

ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

img_size = ppnet_multi.module.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = False

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# load the test data and check test accuracy
test_dir = "/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_test_spiculated/"
if check_test_accu:
    test_batch_size = 100

    test_dataset = DatasetFolder(
        test_dir,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    print('test set size: {0}'.format(len(test_loader.dataset)))

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader, class_specific=class_specific, log=print)
    print(accu)