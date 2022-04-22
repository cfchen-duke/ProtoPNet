#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22 Aprile 2022
baseline usando le ResNet per la classificazione di masse MALIGNANT v. BENIGN
- Versione con k-fold Cross Validation:
    -- per il momento implementare versione con training e validation (senza test), si salvano le performance sul validation set ad ogni fold, e se mediano alla fine
    -- poi, implementare versione con training-validation e infine test, si salvano le performance del modello sul test set ad ogni fold, e se mediano alla fine

@author: si-lab
"""

#%% Imports

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import argparse
from tqdm import tqdm
from time import gmtime,strftime
from settings import train_dir, test_dir #test_dir is actually validation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# from preprocess import mean, std 
#TODO prenderli corretamente col rispettivo valore calcolato:
mean = np.float32(np.uint8(np.load('./datasets/mean.npy'))/255)
std = np.float32(np.uint8(np.load('./datasets/std.npy'))/255)
# mean = 0.5
# std = 0.5



#%% Argparse and initial instances


parse = argparse.ArgumentParser(description="")
parse.add_argument('model_name', help='Name of the baseline architecture: resnet18, resnet34, resnet50, vgg..',type=str)
parse.add_argument('lr', help='learning rate',type=float)
parse.add_argument('wd', help='weight decay',type=float)
parse.add_argument('dr', help='dropout rate',type=float)
parse.add_argument('num_dropouts',help='Number of dropout layers in the bottleneck of ResNet18, if 1 uses one, if 2 uses two.', type=int)
#Add string of information about the specific experiment run, as dataset used, images specification, etc
parse.add_argument('run_info', help='Plain-text string of information about the specific experiment run, as the dataset used, the images specification, etc. This is saved in run_info.txt',type=str)
args = parse.parse_args()

model_names = [args.model_name+'_esperimenti_sistematici_squared224']#TODO
lr = [args.lr]
wd = [args.wd]
dropout_rate = [args.dr]
num_dropouts = args.num_dropouts
run_info_to_be_written = args.run_info
assert(type(run_info_to_be_written)==str)

img_size = 224 #564 #224 #TODO
num_epochs = 1000 #TODO
batch_size = [40]
batch_size_valid = 2
num_classes = 1 #TODO
window = 20
patience = int(np.ceil(50/window)) # sono 3*20 epoche ad esempio



#%% Functions definition

def set_seed(seed):
    # SEED FIXED TO FOSTER REPRODUCIBILITY
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, window=20, patience=3):
    since = time.time()
    val_acc_history = []
    train_acc_history = []
    val_loss=[]
    train_loss=[]
    count = 0
    prima_volta = True
    to_be_stopped = False
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

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
            y_true = []
            y_pred = []

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                
                labels=labels.float()
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1))                       

                    # preds = np.round(outputs.detach().cpu())
                    # preds = torch.round(outputs)
                    
                    # PREDICTIONS 
                    pred = np.round(outputs.detach().cpu())
                    target = np.round(labels.detach().cpu())             
                    y_pred.extend(pred.tolist())
                    y_true.extend(target.tolist())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == torch.round(labels.data))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            epoch_acc = accuracy_score(y_true,y_pred)

            with open(os.path.join(output_dir,f'{phase}_metrics.txt'),'a') as f_out:
                      f_out.write(f'{epoch},{epoch_loss},{epoch_acc}\n')

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                torch.save(obj=model, f=os.path.join(output_dir,('epoch_{}_acc_{:.4f}.pth').format(epoch, best_val_acc)))
                
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss.append(epoch_loss)
                
                ## EARLY STOPPING
                if ((epoch+1) >= 2*(window)) and ((epoch+1) % window==0):
                    loss_npy = np.array(val_loss,dtype=float)
                    windowed = [np.mean(loss_npy[i:i+window]) for i in range(0,len(val_loss),window)]
                    # windowed_epochs = range(window,len(val_loss)+window,window)
                    windowed_2 = windowed[1:]
                    windowed_2.append(0.0)
                    deriv_w = [(e[0]-e[1])/window for e in zip(windowed,windowed_2)]
                    deriv_w = deriv_w[:-1]
                    
                    if prima_volta:
                        prima_volta = False
                        thresh = deriv_w[0]*0.10 #TODO
                    
                    if deriv_w[-1] < thresh:
                        print(f'DETECTION DI VALORE SOTTOSOGLIA, con soglia {thresh}')
                        count += 1
                        if count ==1:
                            model_wts_earlyStopped = copy.deepcopy(model.state_dict())                        
                            torch.save(obj=model, f=os.path.join(output_dir,('earlyStopped_epoch_{}_acc_{:.4f}.pth').format(epoch, epoch_acc)))
                            print('Modello salvato per quando verrà stoppato allenamento')
                        if count >= patience:
                            to_be_stopped = True
                            print(f'PAZIENZA SUPERATA EPOCA {epoch}, ESCO')
                            break
                        else:
                            print(f'EPOCA {epoch} - ASPETTO ANCORA IN PAZIENZA ({patience-count}) EPOCHE')
                            continue
                        
                        
                    else:
                        count = 0
                
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss.append(epoch_loss)
                # joint_lr_scheduler.step()
        if to_be_stopped:
            break
        
        print()
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_val_acc))

    # load best model weights
    if to_be_stopped==False: #the case when EarlyStop never occurs
        model.load_state_dict(best_model_wts)
    else:
        model.load_state_dict(model_wts_earlyStopped)

    return model, val_acc_history, train_acc_history, val_loss, train_loss, best_val_acc


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            

def initialize_model(model_name, num_classes, feature_extract, dropout_rate, num_dropouts, use_pretrained=True):
    model_ft = None
    input_size = 0
    

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        ##TODO 11 aprile 2022: idea di semplificare la base architecture per ridurre il numero di out_features uscente e di conseguenza il numero di filtri necessari ai successivi layer FC
        if num_dropouts==1:
            model_ft.fc = nn.Sequential(
    
                #Fully connected
                nn.Linear(num_ftrs,256),
                nn.ReLU(),
                
                nn.Linear(256,128),
                nn.ReLU(),
                
                #Dropout
                nn.Dropout(p=dropout_rate),
                
                # Classification layer
                nn.Linear(128, num_classes),
                # nn.Softmax() 
                nn.Sigmoid()
                )
        
        elif num_dropouts==2:
            model_ft.fc = nn.Sequential(

                #Fully connected
                nn.Linear(num_ftrs,256),
                nn.ReLU(),
                
                nn.Dropout(p=dropout_rate),

                nn.Linear(256,128),
                nn.ReLU(),
                
                #Dropout
                nn.Dropout(p=dropout_rate),
                
                # Classification layer
                nn.Linear(128, num_classes),
                # nn.Softmax() 
                nn.Sigmoid()
                )
            
        else:
            print('Attention please, invalid value for num_dropouts')
            raise ValueError
            
            
            
            # #Fully connected
            # nn.Linear(num_ftrs,128),
            # nn.ReLU(),
            
            # # nn.Dropout(p=dropout_rate), #TODO
            
            # nn.Linear(128,10),
            # nn.ReLU(),
            
            # #Dropout
            # nn.Dropout(p=dropout_rate),
            
            # # Classification layer
            # nn.Linear(10, num_classes),
            # # nn.Softmax() 
            # nn.Sigmoid()
            # )
        
        
            
            # #Fully connected
            # nn.Linear(num_ftrs,10),
            # nn.ReLU(),
            
            # #Dropout
            # nn.Dropout(p=dropout_rate),
            
            # # Classification layer
            # nn.Linear(10, num_classes),
            # # nn.Softmax() 
            # nn.Sigmoid()
            # )
        
        input_size = img_size  

    if model_name == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        ##TODO 8 aprile 2022: idea di semplificare la base architecture per ridurre il numero di out_features uscente e di conseguenza il numero di filtri necessari ai successivi layer FC
        model_ft.fc = nn.Sequential(

            #Fully connected
            nn.Linear(num_ftrs,256),
            nn.ReLU(),
                       
            nn.Linear(256,128),
            nn.ReLU(),
            
            #Dropout
            nn.Dropout(p=dropout_rate),
            
            # Classification layer
            nn.Linear(128, num_classes),
            # nn.Softmax() 
            nn.Sigmoid()
            )
            
            
            
            # # #TODO consiglio di fare 512>128>10>1. Fully connected
            # nn.Linear(num_ftrs,128),
            # nn.ReLU(),
                       
            # nn.Linear(128,10),
            # nn.ReLU(),
            
            # # #Dropout
            # nn.Dropout(p=dropout_rate),
            
            # # # Classification layer
            # nn.Linear(10, num_classes),
            # # # nn.Softmax() 
            # nn.Sigmoid()
            # )
            
            
            #TODO 14 aprile 2022 mattina
            #nn.Linear(num_ftrs,20),
            #nn.ReLU(),
            
            #Dropout
            #nn.Dropout(p=dropout_rate),
            
            # Classification layer
            #nn.Linear(20, num_classes),
            # nn.Softmax() 
            #nn.Sigmoid()
            #)
        
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
            #nn.Linear(num_ftrs,1024),
            #nn.ReLU(),
                       
            #nn.Linear(1024,512),
            #nn.ReLU(),
            
            #Dropout
            #nn.Dropout(p=dropout_rate),
            
            # Classification layer
            #nn.Linear(512, num_classes),
            # nn.Softmax() 
            #nn.Sigmoid()
            #)
            
            ## VERSIONE CON SOLO DUE FC E NON TRE:
            # #Fully connected
            nn.Linear(num_ftrs,512),
            nn.ReLU(),
                                  
            # #Dropout
            nn.Dropout(p=dropout_rate),
            
            # # Classification layer
            nn.Linear(512, num_classes),
            # # nn.Softmax() 
            nn.Sigmoid()
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

#%% Code

#Flush memory before each new run
torch.cuda.empty_cache()

#Seed reproducibility
set_seed(seed=1)

#Configurations
N = len(lr) * len(wd) * len(dropout_rate) * len(batch_size)
chosen_configurations = get_N_HyperparamsConfigs(N=N) #TODO

for model_name in model_names:
    for idx,config in enumerate(chosen_configurations): 
        
        lr = config[0]
        wd = config[1]
        dropout_rate = config[2]
        batch_size = config[3]
        train_batch_size = batch_size
        valid_batch_size = batch_size_valid
        
        experiment_run = f'CBIS_{model_name}_{strftime("%a_%d_%b_%Y_%H:%M:%S", gmtime())}_cv'
        output_dir = f'./saved_models_baseline/{model_name}/{experiment_run}'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)            
        
        with open(os.path.join(output_dir,'train_metrics.txt'),'w') as f_out:
            f_out.write('epoch,loss,accuracy\n')
        
        with open(os.path.join(output_dir,'val_metrics.txt'),'w') as f_out:
            f_out.write('epoch,loss,accuracy\n')
        
        with open(os.path.join(output_dir,'run_info.txt'),'w') as f_out:
             f_out.write(run_info_to_be_written)      
             
             
        # when True we only update the reshaped layer params
        feature_extract = True        
                
        normalize = transforms.Normalize(mean=mean,
                                         std=std)
             
        ## Datasets
        # training
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.Grayscale(num_output_channels=3), #TODO
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        # validation
        valid_dataset = datasets.ImageFolder(
            test_dir, # from the import, it is actually validation directory
            transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        
        # #Versione senza Cross Validation
        # ## Dataloaders
        # # training
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=train_batch_size, shuffle=True,
        #     num_workers=4, pin_memory=False) #TODO cambiare num_workers=4*num_gpu
        # # validation       
        # valid_loader = torch.utils.data.DataLoader(
        #     valid_dataset, batch_size=valid_batch_size, shuffle=True,#False
        #     num_workers=4, pin_memory=False)
        # # Create training and validation dataloaders
        # dataloaders_dict = {
        #     'train': train_loader,
        #     'val': valid_loader    
        #     }

        
        # Versione k-fold Cross Validation
        ## Set up the Cross Vaidation framework:
        dataset = ConcatDataset([train_dataset, valid_dataset])
        k=10
        splits=KFold(n_splits=k,shuffle=True,random_state=42)
        
        ## Prepare a list to store performance of each fold        
        best_val_accuracy_each_fold=[]
        
        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

            print(f'Model: {model_name}; start config {idx}: {config}; Fold {fold+1}/{k}')

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = torch.utils.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            valid_loader = torch.utils.DataLoader(dataset, batch_size=batch_size_valid, sampler=valid_sampler)
            dataloaders_dict = {
                'train': train_loader,
                'val': valid_loader    
                }
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Init the model for this run
            model_ft, input_size = initialize_model(model_name[:8], num_classes, feature_extract, dropout_rate, num_dropouts, use_pretrained=True) #TODO modelname
            
            if not os.path.exists(os.path.join(output_dir,'model_architecture.txt')):                   
                with open(os.path.join(output_dir,'model_architecture.txt'),'w') as f_out:
                    f_out.write(f'{model_ft}')
                    
            # Send the model to GPU
            model_ft = model_ft.to(device)
                      
            params_to_update = model_ft.parameters()
            print("Params to learn:")
            if feature_extract:
                params_to_update = []
                for name,param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        print("\t",name)
            else:
                for name,param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        print("\t",name)
        
     
            joint_optimizer_specs = [{'params': params_to_update, 'lr': lr, 'weight_decay': wd}]# bias are now also being regularized
            optimizer_ft = torch.optim.Adam(joint_optimizer_specs)
        
            # Setup the loss fxn
            criterion = nn.BCELoss()
        
       
            # Train and evaluate
            model_ft, val_accs, train_accs, val_loss, train_loss, best_val_accuracy = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"),window=window, patience=patience)
            
            
            #
             ## TODO:  Qui ci sarà il pezzo per testare il modello sul test set effettivo, e salvo quelle performance per ogni fold
                     # In questo caso, modificare il nome delle variabili nell header dei txt sotto
            #
            
            # Save performance on validation set for each fold
            best_val_accuracy_each_fold.append(best_val_accuracy) #TODO val->test
            
            

            if not os.path.exists(f'./saved_models_baseline/{model_name}/experiments_crossValidation.txt'):
                with open(f'./saved_models_baseline/{model_name}/experiments_crossValidation.txt', 'w') as out_file:
                    out_file.write('{experiment_run},{fold},{lr},{wd},{dropout_rate},{num_dropouts},{batch_size},{best_val_accuracy}\n') #TODO val->test

            with open(f'./saved_models_baseline/{model_name}/experiments_crossValidation.txt', 'a') as out_file:
                    out_file.write(f'{experiment_run},{fold}.{lr},{wd},{dropout_rate},{num_dropouts},{batch_size},{best_val_accuracy}\n') #TODO val-> test
    
            Al momento, le variabili :
                val_accs, train_accs, val_loss, train_loss
            si perdono --> valutare di tenerne traccia nel ciclo esterno per poi plottare tutte le curve di accuracy (una per fold) in una unica figura finale
            
            #TODO Qui sotto il codice per i due plot nella vecchia versione senza CV
            # non vorrei salvare il plot di ogni fold di CV, quindi salto 
            
            # # Plots
            # val_accs_npy = val_accs
            # train_accs_npy = train_accs
            # x_axis = range(0,len(val_accs_npy))
            # plt.figure()
            # plt.plot(x_axis,train_accs_npy,'*-k',label='Training')
            # plt.plot(x_axis,val_accs_npy,'*-b',label='Validation')
            # # plt.ylim(bottom=0.5,top=1)
            # plt.legend()             
            # plt.title(f'Model: {model_name[:8]} Loss: BCE\nLR: {lr}, WD: {wd}, dropout: {dropout_rate},\nbest validation accuracy: {np.round(best_val_accuracy,decimals=2)}, batch size: {batch_size}')
            # plt.xlabel('Epochs')
            # plt.ylabel('Accuracy')
            # plt.grid()
            # plt.savefig(os.path.join(output_dir,experiment_run+'_acc.pdf'))
            
            # x_axis = range(0,len(val_loss))
            # plt.figure()
            # plt.plot(x_axis,train_loss,'*-k',label='Training')
            # plt.plot(x_axis,val_loss,'*-b',label='Validation')
            # # plt.ylim(bottom=-0.5)
            # plt.legend()
            # plt.title(f'Model: {model_name[:8]} Loss: BCE\nLR: {lr}, WD: {wd}, dropout: {dropout_rate}, batch size: {batch_size}')
            # plt.xlabel('Epochs')
            # plt.ylabel('Loss')
            # plt.grid()
            # plt.savefig(os.path.join(output_dir,experiment_run+'_loss.pdf'))
            
            # with open(os.path.join(output_dir,experiment_run+'_accuracies.txt'),'w') as f_out:
            #           f_out.write(train_accs_npy+'\n'+val_accs_npy)
            
            
            print(f'Model: {model_name}; ended config {idx}: {config}; Fold {fold+1}/{k}. Best validation accuracy = {np.round(best_val_accuracy,decimals=3)}')

        A questo livello andrebbe la figura di cui alla riga 630
        
        # Finally:
        AVG_VAL_ACC = np.round(np.mean(np.array(best_val_accuracy_each_fold)),decimals=3)
        print(f'Model: {model_name}. All {k} folds done.\nList of the best val accuracy for each_fold: {best_val_accuracy_each_fold}\nAVERAGE VAL ACC = {AVG_VAL_ACC}')
        
print('All experiments saved!')
