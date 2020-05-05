import os
import pandas as pd
import cv2

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

source_dir = '../CUB_200_2011/'  # original CUB_200_2011 dataset directory

datasets_root_dir = './datasets/cub200_cropped/'
train_dir = datasets_root_dir + 'train_cropped/'
test_dir = datasets_root_dir + 'test_cropped/'
makedir(train_dir)
makedir(test_dir)

classes = pd.read_csv(source_dir + 'classes.txt', sep=' ', names=['id', 'classname'], index_col='id')
for classname in classes['classname']:
    makedir(train_dir + classname)
    makedir(test_dir + classname)

images = pd.read_csv(source_dir + 'images.txt', sep=' ', names=['id', 'path'], index_col='id')
bounding_boxes = pd.read_csv(source_dir + 'bounding_boxes.txt', sep=' ', names=['id', 'x', 'y', 'weight', 'height'], index_col='id')
train_test_split = pd.read_csv(source_dir + 'train_test_split.txt', sep=' ', names=['id', 'train'], index_col='id')

for idx in images.index:
    print(idx)

    imgpath, = images.loc[idx]
    x, y, weight, height = bounding_boxes.loc[idx]
    is_train, = train_test_split.loc[idx]
    x, y, weight, height = int(x), int(y), int(weight), int(height)

    img = cv2.imread(source_dir + 'images/' + imgpath)
    basepath = train_dir if is_train else test_dir
    cv2.imwrite(basepath + imgpath, img[y:y+height, x:x+weight, :])


