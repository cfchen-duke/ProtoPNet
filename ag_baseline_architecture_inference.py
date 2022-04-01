#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:15:42 2022

@author: si-lab
"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from time import gmtime,strftime
import argparse

from settings import img_size, num_classes
from preprocess import mean, std, preprocess_input_function

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('path_to_model_pth', type=str) #TODO
parser.add_argument('path_to_test_dir', type=str) #TODO





import random
#torch.cuda.empty_cache()
# SEED FIXED TO FOSTER REPRODUCIBILITY
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed=1)


# def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
def train_model(model, dataloaders, criterion,output_dir):

    since = time.time()

    model.eval()   # Set model to evaluate mode
                
    running_loss = 0.0
    running_corrects = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        # Iterate over data.
        for inputs, labels in tqdm(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # # zero the parameter gradients
            # optimizer.zero_grad()
    
            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
    
                outputs = model(inputs)
                loss = criterion(outputs, labels)                       
    
                _, preds = torch.max(outputs, 1)

                preds_npy = preds.data.cpu().numpy()
                y_pred.extend(preds_npy)
                
                labels_npy = labels.data.cpu().numpy()
                y_true.extend(labels_npy)    
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    # classes = ('Benign','Malignant') 
    # cf_mat_norm = confusion_matrix(y_true, y_pred, normalize='true')
    # cf_mat = confusion_matrix(y_true, y_pred).astype(int)
    # np.save(os.path.join(output_dir,'confusion_matrix_norm.npy'),cf_mat_norm)
    # np.save(os.path.join(output_dir,'confusion_matrix.npy'),cf_mat)
    
    # df = pd.DataFrame(cf_mat, index = [i for i in classes],
    #                  columns = [i for i in classes])
    # plt.figure()
    # sn.heatmap(df, annot=True, linewidths=.5, cmap='Blues', linecolor='black', fmt="d", vmin=0)
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    # plt.savefig(os.path.join(output_dir,'confusion_matrix.pdf'),bbox_inches='tight')
    
    # df_norm = pd.DataFrame(cf_mat_norm, index=[i for i in classes], columns=[i for i in classes])
    # plt.figure()
    # sn.heatmap(df_norm, annot=True, linewidths=.5, cmap='Blues', linecolor='black', vmin=0, vmax=1)
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    # plt.savefig(os.path.join(output_dir,'confusion_matrix_norm.pdf'),bbox_inches='tight')

    #
    epoch_loss = running_loss / len(dataloaders.dataset)
    epoch_acc = running_corrects.double() / len(dataloaders.dataset)

    # with open(os.path.join(output_dir,'test_metrics.txt'),'w') as f_out:
    #           f_out.write(f'loss,accuracy\n{epoch_loss},{epoch_acc}')

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_loss, epoch_acc))

    
    print()
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, epoch_acc, epoch_loss





def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            

def initialize_model(model_name, num_classes, feature_extract, dropout_rate, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = img_size  

    if model_name == "resnet50":
        """ Resnet50
        """
        # model_ft = models.resnet50(pretrained=use_pretrained)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = img_size   
        
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(

            #Fully connected
            nn.Linear(num_ftrs,4096),
            nn.ReLU(),
            
            #Dropout
            nn.Dropout(p=dropout_rate),
            
            # Classification layer
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)            
            )
        input_size = img_size  
    
    elif model_name == "vgg19":
        """ VGG19
        """
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    
    elif model_name == "densenet121":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    return model_ft, input_size





#%% CODICE


   
# experiment_run = f'CBIS_baseline_massCalcification_{model_name}_{strftime("%a_%d_%b_%Y_%H:%M:%S", gmtime())}' #TODO
args = parser.parse_args()
path_to_model = args.path_to_model_pth
test_dir = args.path_to_test_dir


output_dir = os.path.dirname(path_to_model)

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=30, shuffle=False,#False
    num_workers=4, pin_memory=False)



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu" #TODO attenzione!


# Initialize the model for this run
# model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, dropout_rate, use_pretrained=True)
model_ft = torch.load(path_to_model,map_location=device)
input_size=img_size

# Send the model to GPU
model_ft = model_ft.to(device)

# params_to_update = model_ft.parameters()
# print("Params to learn:")
# if feature_extract:
#     params_to_update = []
#     for name,param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             params_to_update.append(param)
#             print("\t",name)
# else:
#     for name,param in model_ft.named_parameters():
#         if param.requires_grad == True:
#             print("\t",name)


# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# evaluate
model_ft, val_accs, val_loss= train_model(model_ft, test_loader, criterion,output_dir)


print('Test, done!')
