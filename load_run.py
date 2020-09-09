import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from dataHelper import DatasetFolder
import re
import numpy as np
import os
import train_and_test as tnt
from sklearn.metrics import roc_curve, auc
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse


def main(test_dir, model_path):
    # load the model
    check_test_accu = True


    ppnet = torch.load(model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    class_specific = True

    # load the test data and check test accuracy
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

def draw_roc_curve(data_path, model_path, image_name, target_class, num_classes):
    model = torch.load(model_path)
    test_dataset = DatasetFolder(
        data_path,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=True,
        num_workers=4, pin_memory=False)


    total_one_hot_label, total_output = [], []
    for i, (image, label, patient_id) in enumerate(test_loader):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.no_grad()
        with grad_req:
            output, min_distances = model(input)

            # one hot label for AUC
            one_hot_label = np.zeros(shape=(len(target), num_classes))
            for k in range(len(target)):
                one_hot_label[k][target[k].item()] = 1

            prob = torch.nn.functional.softmax(output, dim=1)
            total_output.extend(prob.data.cpu().numpy())
            total_one_hot_label.extend(one_hot_label)

    total_output = np.array(total_output)
    total_one_hot_label = np.array(total_one_hot_label)
    print(total_output[:5])
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(total_one_hot_label[:, i], total_output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    print("saving!!!!")
    lw = 2
    plt.plot(fpr[target_class], tpr[target_class], color='red',
             lw=lw, label='ROC curve with context (area = %0.2f)' % roc_auc[target_class])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('AUC')
    # plt.legend(loc="lower right")
    # plt.savefig(image_name)



    model = torch.load("/usr/project/xtmp/ct214/saved_models/resnet152/5class_DDSM_1024_0517_neglogit-0.5sep-0.08/50_9push0.6072.pth")
    test_dataset = DatasetFolder(
        "/usr/project/xtmp/mammo/DDSM-context-test/",
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=True,
        num_workers=4, pin_memory=False)

    total_one_hot_label, total_output = [], []
    for i, (image, label, patient_id) in enumerate(test_loader):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.no_grad()
        with grad_req:
            output, min_distances = model(input)

            # one hot label for AUC
            one_hot_label = np.zeros(shape=(len(target), num_classes))
            for k in range(len(target)):
                one_hot_label[k][target[k].item()] = 1

            prob = torch.nn.functional.softmax(output, dim=1)
            total_output.extend(prob.data.cpu().numpy())
            total_one_hot_label.extend(one_hot_label)

    total_output = np.array(total_output)
    total_one_hot_label = np.array(total_one_hot_label)
    print(total_output[:5])
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(total_one_hot_label[:, i], total_output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    print("saving!!!!")
    lw = 2
    plt.plot(fpr[target_class], tpr[target_class], 'b--',
             lw=lw, label='ROC curve no context(area = %0.2f)' % roc_auc[target_class])
    plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('AUC')
    plt.legend(loc="lower right")
    plt.savefig(image_name)


def confusion_matrix(model_path, data_path, num_classes=5):
    # predicted * true
    model = torch.load(model_path)
    test_dataset = DatasetFolder(
        data_path,
        augmentation=False,
        loader=np.load,
        extensions=("npy",),
        transform=transforms.Compose([
            torch.from_numpy,
        ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=True,
        num_workers=4, pin_memory=False)

    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for i, (image, label, patient_id) in enumerate(test_loader):
        input = image.cuda()\

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.no_grad()
        with grad_req:
            output, min_distances = model(input)
            res = torch.argmax(output, dim=1)
            for j in range(len(res)):
                confusion_matrix[res[j]][label[j]] += 1 # cm[predicted][true] += 1

    print("confusion matrix is", confusion_matrix)

if __name__=="__main__":
    main(test_dir='/usr/project/xtmp/mammo/Lo1136i/validation/',
         model_path='/usr/project/xtmp/ct214/saved_models/resnet152/5class_Lo1136_1024_0831_neglogit-1/90_2push0.7511.pth')
    # draw_roc_curve("/usr/project/xtmp/mammo/DDSM-context-test/",
    #                "/usr/project/xtmp/ct214/saved_models/resnet152/DDSM_context_1024_0618/90_6push0.6458.pth",
    #                image_name="withContextCompare", target_class=4, num_classes=5)
    # draw_roc_curve("/usr/project/xtmp/mammo/DDSM-context-test/",
    #                "/usr/project/xtmp/ct214/saved_models/resnet152/5class_DDSM_1024_0517_neglogit-0.5sep-0.08/50_9push0.6072.pth",
    #                image_name="noContext", target_class=4, num_classes=5)
    # confusion_matrix('/usr/project/xtmp/ct214/saved_models/resnet152/5class_Lo1136_1024_0826_neglogit-0/60_7push0.7425.pth', '/usr/xtmp/mammo/Lo1136i/test_DONOTTOUCH/')
