import matplotlib.pyplot as plt
import numpy as np
import torch
from dataHelper import DatasetFolder
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score



train_dir = "/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_train_spiculated_augmented_crazy/"
test_dir = "/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_test_spiculated/"
#
# train_dir = "/usr/project/xtmp/mammo/rawdata/Sept2019/JM_Dataset_Final/normalized_rois/binary_context_roi/binary_train_spiculated_augmented/"
# test_dir = "/usr/project/xtmp/mammo/rawdata/Sept2019/JM_Dataset_Final/normalized_rois/binary_context_roi/binary_test_spiculated/"

# train set
train_dataset = DatasetFolder(
    train_dir,
    augmentation=False,
    loader=np.load,
    extensions=("npy",),
    transform = transforms.Compose([
        torch.from_numpy,
    ]))
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=True,
    num_workers=4, pin_memory=False)

# test set
test_dataset =DatasetFolder(
    test_dir,
    loader=np.load,
    extensions=("npy",),
    transform = transforms.Compose([
        torch.from_numpy,
    ]))
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=100, shuffle=False,
    num_workers=4, pin_memory=False)



device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")

model = models.resnet152(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(512, 2),
                         nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

model.to(device)

epochs = 100
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
curr_best = 0
for epoch in range(epochs):
    total_output = []
    total_one_hot_label  = []
    for inputs, labels, id in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
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
    auc_score = roc_auc_score(np.array(total_one_hot_label), np.array(total_output))
    train_losses.append(running_loss / len(trainloader))
    test_losses.append(test_loss / len(testloader))

    print("=======================================================")
    if auc_score>curr_best:
        curr_best = auc_score
    print("\t at epoch {}".format(epoch))
    print("\t train loss is {}".format(train_losses[-1]))
    print("\t test loss is {}".format(test_losses[-1]))
    print("\t auc is {}".format(auc_score))
    print("\t current best is {}".format(curr_best))
    running_loss = 0
    model.train()
