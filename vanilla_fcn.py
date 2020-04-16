from fcn_model import FCN16s

import matplotlib
import matplotlib.pyplot as plt
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features
import argparse
import torch.nn as nn
from dataHelper import DatasetFolder
from torchvision import transforms
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_auc_score
import os

matplotlib.use("Agg")

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str)
parser.add_argument("-train_dir", type=str, default="/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_train_spiculated_augmented_crazy_with_rot/")
parser.add_argument("-test_dir", type=str, default="/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_test_spiculated/")
parser.add_argument("-name", type=str)
parser.add_argument("-lr", type=lambda x: float(x))
parser.add_argument("-wd", type=lambda x: float(x))
args = parser.parse_args()
model_name = args.model
train_dir = args.train_dir
test_dir = args.test_dir
task_name = args.name
lr = args.lr
wd = args.wd
print(lr, wd)

if not os.path.exists(task_name):
    os.mkdir(task_name)

writer = SummaryWriter()

model = FCN16s()

# load data
# train set
train_dataset = DatasetFolder(
    train_dir,
    augmentation=False,
    loader=np.load,
    extensions=("npy",),
    target_size=(224, 224),
    transform = transforms.Compose([
        torch.from_numpy,
    ]))
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=50, shuffle=True,
    num_workers=4, pin_memory=False)

# test set
test_dataset =DatasetFolder(
    test_dir,
    loader=np.load,
    extensions=("npy",),
    target_size=(224, 224),
    transform = transforms.Compose([
        torch.from_numpy,
    ]))
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=50, shuffle=False,
    num_workers=4, pin_memory=False)


# start training
epochs = 1000

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

device = torch.device("cuda")
model.to(device)

train_losses = []
test_losses = []
train_auc = []
test_auc = []
curr_best = 0


for epoch in range(epochs):
    # train
    confusion_matrix = [0, 0, 0, 0]
    total_output = []
    total_one_hot_label  = []
    running_loss = 0
    model.train()
    for inputs, labels, id in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(inputs)
        # print("logits are ", logps.shape)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        one_hot_label = np.zeros(shape=(len(labels), 2))
        for k in range(len(labels)):
            one_hot_label[k][labels[k].item()] = 1
        # roc_auc_score()
        total_output.extend(logps.cpu().detach().numpy())
        total_one_hot_label.extend(one_hot_label)
        # confusion matrix
        _, predicted = torch.max(logps.data, 1)
        for t_idx, t in enumerate(labels):
            if predicted[t_idx] == t and predicted[t_idx] == 1:  # true positive
                confusion_matrix[0] += 1
            elif t == 0 and predicted[t_idx] == 1:
                confusion_matrix[1] += 1  # false positives
            elif t == 1 and predicted[t_idx] == 0:
                confusion_matrix[2] += 1  # false negative
            else:
                confusion_matrix[3] += 1

    auc_score = roc_auc_score(np.array(total_one_hot_label), np.array(total_output))
    train_losses.append(running_loss / len(trainloader))
    train_auc.append(auc_score)
    print("=======================================================")
    print("\t at epoch {}".format(epoch))
    print("\t train loss is {}".format(train_losses[-1]))
    print("\t train auc is {}".format(auc_score))
    print('\tthe confusion matrix is: \t\t{0}'.format(confusion_matrix))
    # test
    confusion_matrix = [0, 0, 0, 0]
    test_loss = 0
    total_output = []
    total_one_hot_label  = []
    model.eval()
    with torch.no_grad():
        for inputs, labels, id in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            one_hot_label = np.zeros(shape=(len(labels), 2))
            for k in range(len(labels)):
                one_hot_label[k][labels[k].item()] = 1
            # roc_auc_score()
            total_output.extend(logps.cpu().numpy())
            total_one_hot_label.extend(one_hot_label)
            # confusion matrix
            _, predicted = torch.max(logps.data, 1)
            for t_idx, t in enumerate(labels):
                if predicted[t_idx] == t and predicted[t_idx] == 1:  # true positive
                    confusion_matrix[0] += 1
                elif t == 0 and predicted[t_idx] == 1:
                    confusion_matrix[1] += 1  # false positives
                elif t == 1 and predicted[t_idx] == 0:
                    confusion_matrix[2] += 1  # false negative
                else:
                    confusion_matrix[3] += 1
    auc_score = roc_auc_score(np.array(total_one_hot_label), np.array(total_output))
    test_losses.append(test_loss / len(testloader))
    test_auc.append(auc_score)
    print("===========================")
    if auc_score > curr_best:
        curr_best = auc_score
    print("\t test loss is {}".format(test_losses[-1]))
    print("\t test auc is {}".format(auc_score))
    print("\t current best is {}".format(curr_best))
    print('\tthe confusion matrix is: \t\t{0}'.format(confusion_matrix))
    print("=======================================================")

    # save model
    if auc_score > 0.7:
        torch.save(model, task_name + "/" + str(auc_score) + "_at_epoch_" + str(epoch))

    # plot graphs
    plt.plot(train_losses, "b", label="train")
    plt.plot(test_losses, "r", label="test")
    plt.ylim(0, 4)
    plt.legend()
    plt.savefig(task_name+'/train_test_loss_vanilla' + ".png")
    plt.close()

    plt.plot(train_auc, "b", label="train")
    plt.plot(test_auc, "r", label="test")
    plt.ylim(0.4, 1)
    plt.legend()
    plt.savefig(task_name + '/train_test_auc_vanilla' + ".png")
    plt.close()



writer.close()

