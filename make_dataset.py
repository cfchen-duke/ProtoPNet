import torch
import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import settings as config


class ImageDataset(Dataset):
    def __init__(self, csv, train, test, transform):
        self.csv = csv
        self.train = train
        self.test = test
        self.image_names=[]
        self.all_image_names = self.csv[:]['img']
        self.all_labels = np.array(self.csv.drop(['img'], axis=1))
        self.train_ratio = int(0.8 * len(self.csv))
        self.test_ratio = len(self.csv) - self.train_ratio
        self.label_names = csv.keys()[1::]
        self.class_to_idx = {self.label_names[i]:i for i in range (len(self.label_names))}
        self.idx_to_class = {i:self.label_names[i] for i in range (len(self.label_names))}
        # set the training data images and labels
        if self.train:
            print(f"Number of training images: {self.train_ratio}")
            self.image_names = list(self.all_image_names[:self.train_ratio])
            self.labels = list(self.all_labels[:self.train_ratio])
            self.transform = transform

        # set the validation data images and labels
        # elif self.train == False and self.test == False:
        #     print(f"Number of validation images: {self.valid_ratio}")
        #     self.image_names = list(self.all_image_names[-self.valid_ratio:-10])
        #     self.labels = list(self.all_labels[-self.valid_ratio:])
        #     # define the validation transforms
        #     self.transform = transforms.Compose([
        #         transforms.ToPILImage(),
        #         transforms.Resize((400, 400)),
        #         transforms.ToTensor(),
        #     ])

        elif self.test == True and self.train == False:
            self.image_names = list(self.all_image_names[self.train_ratio::])
            self.labels = list(self.all_labels[self.train_ratio::])
            # define the test transforms
            self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = cv2.imread(config.data_img_path + '/' + self.image_names[index])
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        targets = self.labels[index]

        return {
            # 'image': torch.tensor(image, dtype=torch.float32),
            'image': image.clone().detach(),
            'label': torch.tensor(targets, dtype=torch.float32),
            'key'  : self.image_names[index]
        }