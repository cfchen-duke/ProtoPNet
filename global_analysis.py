import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import cv2
import matplotlib.pyplot as plt

import re

import os

from helpers import makedir
import model
import find_nearest
import train_and_test as tnt

from preprocess import preprocess_input_function

import argparse

# Usage: python3 global_analysis.py -modeldir='./saved_models/' -model=''
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
#parser.add_argument('-dataset', nargs=1, type=str, default='cub200')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
load_model_dir = args.modeldir[0]
load_model_name = args.model[0]

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

# load the model
print('load model from ' + load_model_path)
ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

img_size = ppnet_multi.module.img_size

# load the data
# must use unaugmented (original) dataset
from settings import train_push_dir, test_dir
train_dir = train_push_dir

batch_size = 100

# train set: do not normalize
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=False)

# test set: do not normalize
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=False)

root_dir_for_saving_train_images = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_train')
root_dir_for_saving_test_images = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_test')
makedir(root_dir_for_saving_train_images)
makedir(root_dir_for_saving_test_images)

# save prototypes in original images
load_img_dir = os.path.join(load_model_dir, 'img')
prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+str(start_epoch_number), 'bb'+str(start_epoch_number)+'.npy'))
def save_prototype_original_img_with_bbox(fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    #plt.imshow(p_img_rgb)
    #plt.axis('off')
    plt.imsave(fname, p_img_rgb)

for j in range(ppnet.num_prototypes):
    makedir(os.path.join(root_dir_for_saving_train_images, str(j)))
    makedir(os.path.join(root_dir_for_saving_test_images, str(j)))
    save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_train_images, str(j),
                                                             'prototype_in_original_pimg.png'),
                                          epoch=start_epoch_number,
                                          index=j,
                                          bbox_height_start=prototype_info[j][1],
                                          bbox_height_end=prototype_info[j][2],
                                          bbox_width_start=prototype_info[j][3],
                                          bbox_width_end=prototype_info[j][4],
                                          color=(0, 255, 255))
    save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_test_images, str(j),
                                                             'prototype_in_original_pimg.png'),
                                          epoch=start_epoch_number,
                                          index=j,
                                          bbox_height_start=prototype_info[j][1],
                                          bbox_height_end=prototype_info[j][2],
                                          bbox_width_start=prototype_info[j][3],
                                          bbox_width_end=prototype_info[j][4],
                                          color=(0, 255, 255))

k = 5

find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=train_loader, # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
        k=k+1,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_train_images,
        log=print)

find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=test_loader, # pytorch dataloader (must be unnormalized in [0,1])
        prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
        k=k,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_test_images,
        log=print)
