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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse


def main(test_dir, model_dir, model_name):
    # load the model
    check_test_accu = True

    load_model_dir = model_dir
    load_model_name = model_name

    #if load_model_dir[-1] == '/':
    #    model_base_architecture = load_model_dir.split('/')[-3]
    #    experiment_run = load_model_dir.split('/')[-2]
    #else:
    #    model_base_architecture = load_model_dir.split('/')[-2]
    #    experiment_run = load_model_dir.split('/')[-1]

    model_base_architecture = load_model_dir.split('/')[-3]
    experiment_run = load_model_dir.split('/')[-2]

    load_model_path = os.path.join(load_model_dir, load_model_name)

    print('load model from ' + load_model_path)
    print('model base architecture: ' + model_base_architecture)
    print('experiment run: ' + experiment_run)

    ppnet = torch.load(load_model_path,map_location=torch.device('cpu'))
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    img_size = ppnet_multi.module.img_size
    prototype_shape = ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    class_specific = False

    normalize = transforms.Normalize(mean=mean,
                                     std=std)

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

def draw_roc_curve(data_path, model_path, image_name):
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
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(input)

            # one hot label for AUC
            one_hot_label = np.zeros(shape=(len(target), 2))
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
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(total_one_hot_label[:, i], total_output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    print("saving!!!!")
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")
    plt.savefig(image_name)

if __name__=="__main__":
    main(test_dir="/usr/project/xtmp/mammo/binary_Feb/DDSM_five_class_test/",
         model_dir="/usr/project/xtmp/ct214/saved_models/resnet152/5class_DDSM_1024_0506_pushonLo/",
         model_name="50_7push0.6512.pth")
    # draw_roc_curve("/usr/xtmp/mammo/binary_Feb/lesion_or_not_test/",
    #                "/usr/project/xtmp/ct214/saved_models/vgg16/thresholdlogits25_lesion_512_0419/10_3push0.9792.pth",
    #                image_name="lesion")
    # draw_roc_curve("/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_test_spiculated/",
    #                "/usr/project/xtmp/ct214/saved_models/vgg16/thresholdlogits25_spiculated_with_negs_0415/40_7push0.8829.pth",
    #                image_name="spiculation")
