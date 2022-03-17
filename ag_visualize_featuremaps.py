#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:58:39 2022

PROVIAMO A VISUALIZZARE LE FEATURE MAPS USCENTI DA ALCUNI LAYER


@author: si-lab
"""
import argparse
import torch
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model_path', nargs=1, type=str)
parser.add_argument('-i','--img_path', nargs=1, type=str)
args = parser.parse_args()


# model init
load_model_path = args.model_path[0] #'./10_18push0.7822.pth'
ppnet = torch.load(load_model_path)
#ppnet = ppnet.cuda()
.. non funziona ancora
print(ppnet)
print()

nodes, _ = get_graph_node_names(ppnet)
print(nodes)
print()

feature_extractor = create_feature_extractor(
	ppnet, return_nodes=['features.conv1'])
# `out` will be a dict of Tensors, each representing a feature map
out = feature_extractor(torch.zeros(1, 3, 32, 32))
print(out.shape)

from matplotlib import pyplot as plt
plt.imshow(out['features.conv1'][0,:,:,:],title='features.conv1',cmap='gray')
plt.show()

# # fetch test image
# test_image_path = args.img_path[0]
# img_pil = Image.open(test_image_path).convert('RGB') #TODO dobbiamo farla diventare una finta rgb
# to_T = transforms.Compose(
#             transforms.Resize(size=(224, 224)),
#             transforms.ToTensor(),
#     )
# img_tns = to_T(img_pil)

# # show results
# res1 = out[0,:,]


