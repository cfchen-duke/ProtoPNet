#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:25:55 2022
Script per creare una piccola rete CNN
@author: si-lab
"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn.functional as F
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
from settings import train_dir, test_dir,img_size, num_classes \
                     # train_batch_size, test_batch_size
from preprocess import mean, std, preprocess_input_function

# from sklearn.metrics import confusion_matrix
# import seaborn as sn
# import pandas as pd



num_epochs = 200

import random
torch.cuda.empty_cache()
# SEED FIXED TO FOSTER REPRODUCIBILITY
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed=1)



# lr = [1e-3, 1e-4, 1e-5, 1e-6]
# lr = [1e-5, 1e-6]
lr = [1e-5]
# lr=[1e-6]

# lr = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]     #TODO
# wd = [1e-1, 1e-2, 0]
wd = [1e-3]
dropout_rate = [0.7, 0]
#dropout_rate=[0,0.4,0.7]
batch_size = [15]

# joint_lr_step_size = [2, 5, 10]
# gamma_value = [0.10, 0.50, 0.25]

N = len(lr) * len(wd) * len(dropout_rate) * len(batch_size)
#* len(joint_lr_step_size) * len(gamma_value)

# def get_N_HyperparamsConfigs(N=0, lr=lr, wd=wd, joint_lr_step_size=joint_lr_step_size,
#                              gamma_value=gamma_value):
def get_N_HyperparamsConfigs(N=0, lr=lr, wd=wd, dropout_rate=dropout_rate):
    configurations = {}
    h = 1
    for i in lr:
        for j in wd:
            for k in dropout_rate:
                for l in batch_size:
                    configurations[f'config_{h}'] = [i, j, k, l]
                    h += 1
    
                     
    configurations_key = list(configurations.keys())
    chosen_configs = sorted(random.sample(configurations_key,N)) 
    return [configurations[x] for x in chosen_configs]


def train_model(model, dataloaders, criterion, optimizer, num_epochs=100):
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    val_loss=[]
    train_loss=[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)                       

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            with open(os.path.join(output_dir,f'{phase}_metrics.txt'),'a') as f_out:
                      f_out.write(f'{epoch},{epoch_loss},{epoch_acc}\n')

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                torch.save(obj=model, f=os.path.join(output_dir,('epoch_{}_acc_{:.4f}.pth').format(epoch, best_acc)))
                
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss.append(epoch_loss)
                # joint_lr_scheduler.step()
        print()
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, val_loss, train_loss, best_acc
    

class Net(nn.Module):
    # def __init__(self,input_fts1=3,output_fts1=64,dropout_rate=0.4):
    #   super(Net, self).__init__()
    #   self.conv1 = nn.Conv2d(input_fts1, output_fts1, 3, 1)
    #   self.conv2 = nn.Conv2d(output_fts1, output_fts1*2, 3, 1)
    #   # self.conv3 = nn.Conv2d(output_fts1*2, output_fts1*4, 3, 1)

    #   self.dropout = nn.Dropout2d(dropout_rate)

    #   self.fc1 = nn.Linear((54**2)*output_fts1*2, 256) #TODO 224>222>111>109>54 --> (54**2)*output_fts1*2

    #   self.fc2 = nn.Linear(256, 2)

    # # x represents our data
    # def forward(self, x):
      
    #   x = self.conv1(x)
    #   x = F.relu(x)
    #   x = F.max_pool2d(x, 2)

    #   x = self.conv2(x)
    #   x = F.relu(x)
    #   x = F.max_pool2d(x, 2)

    #   # x = self.conv3(x)
    #   # x = F.relu(x)
      
    #   # x = F.max_pool2d(x, 2)
     
    #   x = self.dropout(x)

    #   x = torch.flatten(x, 1)

    #   x = self.fc1(x)
    #   x = F.relu(x)
    #   x = self.dropout(x)
    #   x = self.fc2(x)

    #   output = F.softmax(x, dim=1)
    #   return output  


    def __init__(self,input_fts1=3,output_fts1=64,dropout_rate=0.4):
      super(Net, self).__init__()
      
      self.conv1 = nn.Sequential(
          nn.Conv2d(input_fts1, output_fts1, 3, 1), #DIM = DIM-2
          nn.ReLu(),
          nn.MaxPool2d(2) #DIM = DIM/2
          )
      self.conv2 = nn.Sequential(
          nn.Conv2d(output_fts1, output_fts1*2, 3, 1), #DIM = DIM-2
          nn.ReLu(),
          nn.MaxPool2d(2) #DIM = DIM/2
          )
      self.dropout = nn.Dropout2d(dropout_rate)
      self.flatten = nn.Flatten()
      self.classifier = nn.Sequential(
          nn.Linear((54**2)*output_fts1*2, 256), #TODO 224>222>111>109>54 --> (54**2)*output_fts1*2
          nn.ReLu(),
          self.dropout,
          nn.Linear(256, 2),
          nn.Softmax(dim=1)
          )


    # x represents our data
    def forward(self, x):
      
      x = self.conv1(x)
      x = self.conv2(x)     
      x = self.dropout(x)
      x = self.flatten(x)
      output = self.classifier(x)

      return output          
    
    
# my_nn = Net()
# print(my_nn)


#%% CODICE




# chosen_configurations = get_N_HyperparamsConfigs(N=N)
chosen_configurations = get_N_HyperparamsConfigs(N=N) #TODO




for idx,config in enumerate(chosen_configurations): 
    print(f'Starting config {idx}: {config}')
    lr = config[0]
    wd = config[1]
    dropout_rate = config[2]
    batch_size = config[3]
    train_batch_size = batch_size
    test_batch_size = batch_size
    
    # joint_lr_step_size = config[2]
    # gamma_value = config[3]
    
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "cnn_custom" #TODO
    
    experiment_run = f'CBIS_baseline_massBenignMalignant_{model_name}_{strftime("%a_%d_%b_%Y_%H:%M:%S", gmtime())}'
    # experiment_run = f'CBIS_baseline_massCalcification_{model_name}_{strftime("%a_%d_%b_%Y_%H:%M:%S", gmtime())}' #TODO

    output_dir = f'./saved_models_baseline/{model_name}/{experiment_run}'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    
    with open(os.path.join(output_dir,'train_metrics.txt'),'w') as f_out:
        f_out.write('epoch,loss,accuracy\n')
    
    with open(os.path.join(output_dir,'val_metrics.txt'),'w') as f_out:
        f_out.write('epoch,loss,accuracy\n')
    
    
    
    # all datasets
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    # train set
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Grayscale(num_output_channels=3), #TODO
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=False) #TODO cambiare num_workers=4*num_gpu

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
        test_dataset, batch_size=test_batch_size, shuffle=True,#False
        num_workers=4, pin_memory=False)

    # Create training and validation dataloaders
    dataloaders_dict = {
        'train': train_loader,
        'val': test_loader    
        }
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    # Initialize the model for this run
    model_ft = Net(3,32,dropout_rate) #TODO
    
    with open(os.path.join(output_dir,'model_architecture.txt'),'w') as f_out:
        f_out.write(f'{model_ft}')
   
    # Send the model to GPU
    model_ft = model_ft.to(device)
    
 
    joint_optimizer_specs = [{'params':model_ft.parameters(),'lr': lr, 'weight_decay': wd}]# bias are now also being regularized
    optimizer_ft = torch.optim.Adam(joint_optimizer_specs)
    # joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=joint_lr_step_size, gamma=gamma_value)
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss() #TODO due classi; oppure usare BinaryCross una classe con Sigmoide.
    
    # Train and evaluate
    model_ft, val_accs, train_accs, val_loss, train_loss, best_accuracy= train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
    

    with open(f'./saved_models_baseline/{model_name}/experiments_setup_massBenignMalignant.txt', 'a') as out_file: #TODO ricordati di cambiare il nome del txt se cambia esperimento

    # with open(f'./saved_models_baseline/{model_name}/experiments_setup_massCalcification.txt', 'a') as out_file: #TODO ricordati di cambiare il nome del txt se cambia esperimento
        # out_file.write(f'{experiment_run},{lr},{wd},{joint_lr_step_size},{gamma_value},{img_size},{num_classes},{train_batch_size},{test_batch_size},{num_train_epochs},{best_accuracy}\n')
        out_file.write(f'{experiment_run},{lr},{wd},{dropout_rate},{batch_size},{best_accuracy}\n')

    
    
    
    # Plots
    val_accs_npy = [elem.detach().cpu() for elem in val_accs]
    train_accs_npy = [elem.detach().cpu() for elem in train_accs]
    x_axis = range(0,len(val_accs_npy))
    plt.figure()
    plt.plot(x_axis,train_accs_npy,'*-k',label='Training')
    plt.plot(x_axis,val_accs_npy,'*-b',label='Validation')
    # plt.ylim(bottom=0.5,top=1)
    plt.legend()
    b_acc = best_accuracy.detach().cpu()
    plt.title(f'Model: {model_name}\nLR: {lr}, WD: {wd}, dropout: {dropout_rate},\nbest validation accuracy: {np.round(b_acc,decimals=2)}, batch size: {batch_size}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig(os.path.join(output_dir,experiment_run+'_acc.pdf'))
    
    
    
    # val_loss_npy = [elem.detach().cpu() for elem in val_loss]
    # train_loss_npy = [elem.detach().cpu() for elem in train_loss]
    x_axis = range(0,len(val_loss))
    plt.figure()
    plt.plot(x_axis,train_loss,'*-k',label='Training')
    plt.plot(x_axis,val_loss,'*-b',label='Validation')
    # plt.ylim(bottom=-0.5)
    plt.legend()
    plt.title(f'Model: {model_name}\nLR: {lr}, WD: {wd}, dropout: {dropout_rate}, batch size: {batch_size}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(os.path.join(output_dir,experiment_run+'_loss.pdf'))
    
    # with open(os.path.join(output_dir,experiment_run+'_accuracies.txt'),'w') as f_out:
    #           f_out.write(train_accs_npy+'\n'+val_accs_npy)
    print(f'End of config {idx}: {config}')

print('All experiments saved!')

