from __future__ import division
import numpy as np
import os
import pandas as pd
import argparse
import sys
import random
import png
from matplotlib.pyplot import imsave, imread
import matplotlib
from PIL import Image
matplotlib.use("Agg")
import torchvision.datasets as datasets
from skimage.transform import resize
import ast
import pickle
import csv
import pydicom as dcm
import Augmentor


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)
    return images


def random_flip(input, axis):
    ran = random.random()
    if ran > 0.5:
        return np.flip(input, axis=axis)
    else:
        return input


def random_crop(input):
    ran = random.random()
    if ran > 0.2:
        # find a random place to be the left upper corner of the crop
        rx = int(random.random() * input.shape[0] // 10)
        ry = int(random.random() * input.shape[1] // 10)
        return input[rx: rx + int(input.shape[0] * 9 // 10), ry: ry + int(input.shape[1] * 9 // 10)]
    else:
        return input


def random_rotate_90(input):
    ran = random.random()
    if ran > 0.5:
        return np.rot90(input)
    else:
        return input


def random_shift(input, axis, range):
    ran = random.random()


def random_rotation(x, chance):
    ran = random.random()
    img = Image.fromarray(x)
    if ran > 1 - chance:
        # create black edges
        angle = np.random.randint(0, 90)
        img = img.rotate(angle=angle, expand=1)
        return np.asarray(img)
    else:
        return np.asarray(img)


class DatasetFolder(datasets.DatasetFolder):
    def __init__(self, root, loader, augmentation=False, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, target_size=(224, 224)):

        super(DatasetFolder, self).__init__(root, loader, ("npy",),
                                            transform=transform,
                                            target_transform=target_transform, )
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                extensions)))
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.augment = augmentation
        self.target_size = target_size
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        patient_id = path.split("/")[-1][:-4]
        sample = self.loader(path)
        if self.target_size:
            sample = resize(sample, self.target_size)
        #  normalize to 0 to 1
        # sample = (sample - np.amin(np.abs(sample))) / (np.amax(np.abs(sample))-np.amin(np.abs(sample)))
        #  imagenet normalization
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # temp = []
        # for i in range(3):
        #     temp += [sample-mean[i] / std[i]]

        # print("before transform", sample.shape)
        if self.augment:
            sample = random_rotation(sample, 0.7)
        temp = []
        for i in range(3):
            # sample = random_rotation(sample)
            # temp.append(dog(sample))
            temp.append(sample)
        # temp = [sample, sample, sample]
        n = np.stack(temp)

        if self.transform is not None:
            sample = self.transform(n)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # print("after transform", sample.shape)
        return sample.float(), target, patient_id


def dataAugmentation(in_dir, out_dir, num_of_sample):
    """
    Augment images from train_dir and save it to out_dir
    :param train_dir: image source dir
    :param out_dir: image target dir
    :return: None
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for root, dirs, files in os.walk(in_dir):
        for dir in dirs:
            dir += "/"
            out = os.path.join(out_dir, dir)
            train_dir = os.path.join(in_dir, dir)
            if not os.path.exists(out):
                os.makedirs(out)
            p = Augmentor.Pipeline(train_dir, output_directory=out, save_format="PNG")
            p.rotate(probability=0.9, max_left_rotation=20, max_right_rotation=20)
            # p.zoom(probability=1, min_factor=0.8, max_factor=1.2)
            p.rotate90(probability=0.5)
            p.rotate180(probability=0.5)
            p.rotate270(probability=0.5)
            # p.skew(probability=0.7)
            # p.random_distortion(probability=0.7, grid_width=5, grid_height=5, magnitude=5)
            p.flip_random(probability=0.5)
            # p.shear(1, 0.2, 0.2)
            p.sample(num_of_sample)


def cleanup(dir):
    """
    throw out all the images that has over 80% black, 80% white, and size smaller than 10kb
    :param dir: the dir to clean
    :return: None
    """
    removed = 0
    if not os.path.exists("/usr/xtmp/ct214/removed/"):
        os.mkdir("/usr/xtmp/ct214/removed/")
    for root, dir, files in os.walk(dir):
        for file in files:
            if "png" in file:
                path = os.path.join(root, file)
                image = imread(path)
                black = image.shape[0] * image.shape[1] - np.count_nonzero(np.round(image, 3))
                whites = image.shape[0] * image.shape[1] - np.count_nonzero(np.round(image - np.amax(image), 4))
                if image.shape[0] < 10 or image.shape[1] < 10 or black >= image.shape[0] * image.shape[1] * 0.8 or \
                        whites >= image.shape[0] * image.shape[1] * 0.8 or os.path.getsize(path) / 1024 < 10:
                    imsave("/usr/xtmp/ct214/removed/" + file[:-4], image)
                    os.remove(path)
                    print("file removed!")
            if "npy" in file:
                path = os.path.join(root, file)
                image = np.load(path)
                whites = image.shape[0] * image.shape[1] - np.count_nonzero(np.round(image - np.amax(image), 4))
                black = image.shape[0] * image.shape[1] - np.count_nonzero(np.round(image, 3))
                if image.shape[0] < 10 or image.shape[1] < 10 or black >= image.shape[0] * image.shape[1] * 0.8 or \
                        whites >= image.shape[0] * image.shape[1] * 0.8 or os.path.getsize(path) / 1024 < 10:
                    imsave("/usr/xtmp/ct214/removed/" + file[:-4], image)
                    os.remove(path)
                    removed += 1
                    print("file removed!")
    print(removed)


def dataAugNumpy(path, targetNumber, targetDir, skip=None, rot=True):
    classes = os.listdir(path)
    if not os.path.exists(targetDir):
        os.mkdir(targetDir)
    for class_ in classes:
        if not os.path.exists(targetDir + class_):
            os.makedirs(targetDir + class_)

    for class_ in classes:
        count, round = 0, 0
        while count < targetNumber:
            round += 1
            for root, dir, files in os.walk(os.path.join(path, class_)):
                for file in files:
                    if skip and skip in file:
                        continue
                    filepath = os.path.join(root, file)
                    arr = np.load(filepath)
                    try:
                        arr = random_crop(arr)
                        if rot:
                            arr = random_rotation(arr, 0.9)
                        arr = random_flip(arr, 0)
                        arr = random_flip(arr, 1)
                        arr = random_rotate_90(arr)
                        arr = random_rotate_90(arr)
                        arr = random_rotate_90(arr)
                        # arr = random_rotation(arr, 0.9)

                        whites = arr.shape[0] * arr.shape[1] - np.count_nonzero(np.round(arr - np.amax(arr), 2))
                        black = arr.shape[0] * arr.shape[1] - np.count_nonzero(np.round(arr, 2))

                        if arr.shape[0] < 10 or arr.shape[1] < 10 or black >= arr.shape[0] * arr.shape[1] * 0.8 or \
                                whites >= arr.shape[0] * arr.shape[1] * 0.8:
                            print("illegal content")
                            continue

                        if count % 1500 == 0:
                            if not os.path.exists("./visualizations_of_augmentation/" + class_ + "/"):
                                os.makedirs("./visualizations_of_augmentation/" + class_ + "/")
                            imsave("./visualizations_of_augmentation/" + class_ + "/" + str(count), arr,
                                   cmap="gray")

                        np.save(targetDir + class_ + "/" + file[:-4] + "aug" + str(round), arr)
                        count += 1
                        print(count)
                    except:
                        if not os.path.exists("./error_of_augmentation/" + class_ + "/"):
                            os.makedirs("./error_of_augmentation/" + class_ + "/")
                        np.save("./error_of_augmentation/" + class_ + "/" + str(count), arr)
                    if count > targetNumber:
                        break
    print(count)


def window_adjustment(wwidth, wcen):
    if wcen == 2047 and wwidth == 4096:
        return wwidth, wcen
    else:
        new_wcen = np.random.randint(-100, 300)
        new_wwidth = np.random.randint(-200, 300)
        wwidth += new_wwidth
        wcen += new_wcen
        return wwidth, wcen

def read_16bit_img(path):
    # read image into np
    reader = png.Reader(path)
    data = reader.read()
    pixels = data[2]
    image = []
    for row in pixels:
        row = np.asarray(row, dtype=np.float32)
        image.append(row)
    return np.stack(image, 1)

def windowing(wcen, wwidth, img):
    image = ((img - wcen) / float(wwidth)) + 0.5
    return np.clip(image, 0, 1)

def cropROI(target, csvpath, augByWindow=False, numAugByWin=5,
            datapath="/usr/project/xtmp/mammo/rawdata/Jan2020/PenRad_Dataset_SS_Final/sorted_by_mass_edges_Jan_in/train/"):
    """Crops out the ROI of the image as defined in the spreadsheet provided by Yinhao."""
    df = pd.read_excel(csvpath)
    # classes = df["Class"]
    locations = df['Box_List']
    win_width = df['Win_Width']
    win_cen = df['Win_Center']
    names = list(df["File_Name"])
    did = set()
    if augByWindow:
        target = target[:-1] + "_augByWin/"
    count, max_shape0, max_shape1 = 0, 0, 0
    avg_shape0, avg_shape1 = 0, 0
    file_count = 0
    for root, dir, files in os.walk(datapath):
        for file in files:
            file_count += 1
            # find the index of the name
            path = os.path.join(root, file)
            name_list = file.split("_")
            name = "_".join([name_list[-4][-5:]] + name_list[-3:])
            if len(name.split("_")[0]) != 5:
                name = "_".join([name_list[-5][-2:]] + name_list[-4:])
            name = name[:-4] + ".png"
            if name in names:
                i = names.index(name)
            else:
                print("failed to find ", name)
                continue
            # find the class of the file
            margin = path.split("/")[-2]
            if not os.path.exists(target + margin):
                os.makedirs(target + margin)
            did.add(name)
            # read image into np
            image = read_16bit_img(path)

            # ds = dcm.read_file(path)
            # image = ds.pixel_array

            if augByWindow:
                for k in range(numAugByWin):
                    wwidth = np.asarray(ast.literal_eval(win_width[i])).max()
                    wcen = np.median(np.asarray(ast.literal_eval(win_cen[i])))

                    wwidth, wcen = window_adjustment(wwidth, wcen)

                    image = ((image - wcen) / wwidth) + 0.5
                    image = np.clip(image, 0, 1)

                    # read the location
                    location = locations[i]
                    j, curr, temp = 0, "", []
                    while j < len(location):
                        if location[j] in "1234567890":
                            curr += location[j]
                        else:
                            if curr:
                                temp.append(int(curr))
                                curr = ""
                        j += 1
                    location = temp
                    if len(location) % 4 != 0:
                        print("Failed because of Illegal location information ", location, " for name ", name)
                        continue
                    for j in range(len(location) // 4):
                        # if j not in mass_index:
                        #     continue
                        x1, y1, x2, y2 = location[4 * j:4 * (j + 1)]
                        x1, y1, x2, y2 = max(0, min(x1, x2) - 100), max(0, min(y1, y2) - 100), \
                                         min(image.shape[0], max(x1, x2) + 100), min(image.shape[1], max(y1, y2) + 100)
                        # x1, y1 = midx - target_size//2, midy - target_size//2
                        # x2, y2 = x1 + target_size, y1 + target_size
                        roi = image[x1:x2, y1:y2]

                        # print(roi.shape)
                        np.save(target + margin + "/" + name[:-4] + "#" + str(j) + "#" + str(k) + ".npy", roi)
                        avg_shape0 += roi.shape[0]
                        avg_shape1 += roi.shape[1]
                        max_shape0 = max(max_shape0, roi.shape[0])
                        max_shape1 = max(max_shape1, roi.shape[1])
                        count += 1
                        print("successfully saved ", name, " . Have saved ", count, " total, seen ", file_count,
                              " files in total")

            else:
                wwidth = np.asarray(ast.literal_eval(win_width[i])).max()
                wcen = np.median(np.asarray(ast.literal_eval(win_cen[i])))

                image = windowing(wcen, wwidth, image)
                # read the location
                location = locations[i]
                j, curr, temp = 0, "", []
                while j < len(location):
                    if location[j] in "1234567890":
                        curr += location[j]
                    else:
                        if curr:
                            temp.append(int(curr))
                            curr = ""
                    j += 1
                location = temp
                if len(location) % 4 != 0:
                    print("Failed because of Illegal location information ", location, " for name ", name)
                    continue
                for j in range(len(location) // 4):
                    # if j not in mass_index:
                    #     continue
                    x1, y1, x2, y2 = location[4 * j:4 * (j + 1)]
                    x1, y1, x2, y2 = max(0, min(x1, x2) - 100), max(0, min(y1, y2) - 100), \
                                     min(image.shape[0], max(x1, x2) + 100), min(image.shape[1], max(y1, y2) + 100)
                    # x1, y1 = midx - target_size//2, midy - target_size//2
                    # x2, y2 = x1 + target_size, y1 + target_size
                    roi = image[x1:x2, y1:y2]

                    # print(roi.shape)
                    np.save(target + margin + "/" + name[:-4] + "#" + str(j) + ".npy", roi)
                    # np.save(target + name[:-4] + "#" + str(j) + "#" + str(j) + ".npy", roi)
                    avg_shape0 += roi.shape[0]
                    avg_shape1 += roi.shape[1]
                    max_shape0 = max(max_shape0, roi.shape[0])
                    max_shape1 = max(max_shape1, roi.shape[1])
                    # Use pypng to write z as a color PNG.
                    # imsave(target + margin + "/" + name, roi)
                    # with open(target + margin + "/" + name, 'wb') as f:
                    ##writer = png.Writer(width=roi.shape[1], height=roi.shape[0], bitdepth=16)
                    ## Convert z to the Python list of lists expected by
                    ## the png writer.
                    # roi2list = roi.tolist()
                    # writer.write(f, roi2list)
                    count += 1
                    print("successfully saved ", name, " . Have saved ", count, " total, seen ", file_count,
                          " files in total")

    print(max_shape0, max_shape1)
    print(avg_shape0 / count, avg_shape1 / count)


def crop_negative_patches(target, datapath):
    """Crops out the regions around ROI of the image as defined in the spreadsheet provided by Yinhao."""
    if not os.path.exists(target):
        os.makedirs(target)
    # df = pd.read_excel("/usr/project/xtmp/mammo/rawdata/Sept2019/JM_Dataset_Final/no_PHI_Sept.xlsx")
    df = pd.read_excel("/usr/project/xtmp/mammo/rawdata/Jan2020/Anotation_Master_adj.xlsx")
    # datapath = "/usr/project/xtmp/mammo/rawdata/Sept2019/JM_Dataset_Final/sorted_by_mass_edges_Sept/train/"
    # datapath = "/usr/project/xtmp/mammo/rawdata/Jan2020/PenRad_Dataset_SS_Final/sorted_by_mass_edges_Jan_in/train/"
    locations = df['Box_List']
    win_width = df['Win_Width']
    win_cen = df['Win_Center']
    names = list(df["File_Name"])
    did = set()
    count, max_shape0, max_shape1 = 0, 0, 0
    for root, dir, files in os.walk(datapath):
        for file in files:
            # find the index of the name
            path = os.path.join(root, file)
            name_list = file.split("_")
            name = "_".join([name_list[-4][-5:]] + name_list[-3:])
            if len(name.split("_")[0]) != 5:
                name = "_".join([name_list[-5][-2:]] + name_list[-4:])
            name = name[:-4] + ".png"
            if name in did:
                print("already seen ", name)
                continue
            if name in names:
                i = names.index(name)
            else:
                print("failed to find ", name)
                continue

            # find the class of the file
            for margin in ["spiculated", "circumscribed", "indistinct", "microlobulated", "obscured"]:
                if not os.path.exists(target + "binary_test_" + margin + "/allneg/"):
                    os.makedirs(target + "binary_test_" + margin + "/allneg/")
                did.add(name)
                # read image into np
                reader = png.Reader(path)
                data = reader.read()
                pixels = data[2]
                image = []
                for row in pixels:
                    row = np.asarray(row, dtype=np.uint16)
                    image.append(row)
                image = np.stack(image, 1)

                wwidth = np.asarray(ast.literal_eval(win_width[i])).max()
                wcen = np.median(np.asarray(ast.literal_eval(win_cen[i])))

                image = ((image - wcen) / wwidth) + 0.5
                image = np.clip(image, 0, 1)

                # read the location
                location = locations[i]
                j, curr, temp = 0, "", []
                while j < len(location):
                    if location[j] in "1234567890":
                        curr += location[j]
                    else:
                        if curr:
                            temp.append(int(curr))
                            curr = ""
                    j += 1
                location = temp
                if len(location) % 4 != 0:
                    print("Illegal location information ", location)
                    continue
                for j in range(len(location) // 4):
                    x1, y1, x2, y2 = location[4 * j:4 * (j + 1)]
                    x1, y1, x2, y2 = max(0, min(x1, x2) - 100), max(0, min(y1, y2) - 100), \
                                     min(image.shape[0], max(x1, x2) + 100), min(image.shape[1], max(y1, y2) + 100)
                    # crop out locations around. Make sure that its in the right range
                    neg1 = image[max(0, x1 - 224):x1, y1:y2]
                    neg2 = image[x2:min(x2 + 224, image.shape[0]), y1:y2]
                    neg3 = image[x1:x2, max(0, y1 - 224):y1]
                    neg4 = image[x1:x2, y2:min(y2 + 224, image.shape[1])]
                    neg5 = image[max(0, x1 - 224):x1, y2:min(y2 + 224, image.shape[1])]
                    neg6 = image[max(0, x1 - 224):x1, max(0, y1 - 224):y1]
                    neg7 = image[x2:min(x2 + 224, image.shape[0]), y2:min(y2 + 224, image.shape[1])]
                    neg8 = image[max(0, x1 - 224):x1, max(0, y1 - 224):y1]

                    for index, roi in enumerate([neg1, neg2, neg3, neg4, neg5, neg6, neg7, neg8]):
                        if roi.shape[0] > 10 and roi.shape[1] > 10 and np.count_nonzero(roi) > roi.shape[0] * \
                                roi.shape[1] * 0.7:
                            np.save(target + "binary_test_" + margin + "/allneg/" + name[:-4] +
                                    "#" + str(j) + "neg" + str(index) + ".npy",
                                    roi)
                            count += 1

                print("successfully saved neg ", name, " . Have saved ", count, " total")


def move_to_binary(pos, before, target):
    positive_target = target + pos + "/"
    negative_target = target + "allneg/"
    count = 0
    if not os.path.exists(positive_target):
        os.makedirs(positive_target)
    if not os.path.exists(negative_target):
        os.makedirs(negative_target)
    path_list = []
    for root, dirs, files in os.walk(before):
        for file in files:
            path = os.path.join(root, file)

            if pos in path:
                path_list.append(path)
    for path in path_list:
        data = np.load(path)
        file_name = path.split("/")[-1]
        np.save(positive_target + file_name, data)
        count += 1
        print("successfully saved ", file_name, " . Have saved ", count, " total for pos: " + pos)
    seen = set(path_list)
    for root, dirs, files in os.walk(before):
        for file in files:
            path = os.path.join(root, file)
            if path in seen:
                continue
            else:
                data = np.load(path)
                file_name = path.split("/")[-1]
                np.save(negative_target + file_name, data)
                count += 1
                print("successfully saved ", file_name, " . Have saved ", count, " total for neg: " + pos)


def move_DOI_to_training():
    df = pd.read_csv("/usr/project/xtmp/mammo/binary_Feb/CBIS-DDSM/mass.csv")
    margins = df["mass margins"]
    roi_names = [s.split("/")[0] for s in df["cropped image file path"]]
    seen = set()
    count = 0
    for root, dirs, files in os.walk("/usr/project/xtmp/mammo/binary_Feb/CBIS-DDSM/"):
        for file in files:
            path = os.path.join(root, file)
            name = path.split("/")[-4]
            if name in roi_names:
                index = roi_names.index(name)
            else:
                continue
            margin = margins[index]
            # print(margin)
            # find save directory
            # first detect spiculated, then circumscribed, then obscured, then microlobulated, then ill-defined, then other
            if not margin or type(margin) != str:
                continue

            # check file size -- only save file that is smaller than 500k
            size = os.path.getsize(path)
            if size > 500 * 1024:
                continue

            # find save directory
            save_dir = "/usr/xtmp/mammo/binary_Feb/DDSM_five_class_test/" + margin + "/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # dont save duplicates for now
            if name in seen:
                continue
            seen.add(name)

            # read in dcm files
            ds = dcm.read_file(path)
            image = ds.pixel_array

            # save to npy
            np.save(save_dir + name[14:], image)
            count += 1
            print(count)

            # visualize some of it

            if count %50==0:
                imsave("./visualizations_of_augmentation/"+name[14:], image, cmap="gray")

    print("saved ", count, " images in total")
    # print(name[14:])


def Fides_visualization(size):
    paths = []
    save_dir = "Fides" + str(size)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for root, dir, files in os.walk(
            "/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_test_spiculated/spiculated/"):
        for file in files:
            if file.endswith(".npy"):
                path = os.path.join(root, file)
                paths.append(path)

    paths.sort()
    with open("fides_name_list_test.data", "wb") as filehandle:
        pickle.dump(paths, filehandle)
    tosave = np.zeros((size * 250, size * 250))
    index = 0
    index_ = 1
    for path in paths:
        arr = np.load(path)
        arr = resize(arr, (224, 224))
        arr = np.pad(arr, 13, constant_values=0)
        tosave[(index // size) * 250:(index // size) * 250 + 250, (index % size) * 250:(index % size) * 250 + 250] = arr
        index += 1
        if index == size * size:
            imsave(save_dir+ "/"+str(index_), tosave, cmap="gray")
            tosave = np.zeros((size * 250, size * 250))
            index_ += 1
            index = 0
            print("Saved!")
    imsave(save_dir + "/last", tosave, cmap="gray")
    print("Saved!")


def Fidex_visualization_csv(dir):
    with open(dir, "rb") as filehandle:
        paths = pickle.load(filehandle)

    with open("fides_test.csv", "w") as csv_file:
        fieldnames = ["img name", "mega image label", "row", "col", "spiculated? 1 - yes, 0 - no, 2 - unsure",
                      "Good, prototypical example to display? 1 - yes, 0 - no"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i, path in enumerate(paths):
            name = path.split("/")[-1]
            writer.writerow({"img name": name,
                             "mega image label": str(i // 100 + 1),
                             "row": str((i // 10) % 10 + 1),
                             "col": str(i % 10 + 1),
                             })


def remove_duplicates(dir):
    classes = list(os.listdir(dir))
    classes.sort()
    class1, class2 = classes
    print(class1, class2)
    path = os.path.join(dir, class2)
    seen = set()
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".npy"):
                name = file[:file.index("#")]
                seen.add(name)
    count, remove = 0, 0
    path = os.path.join(dir, class1)
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".npy"):
                count += 1
                name = file[:file.index("#")]
                if name in seen:
                    os.remove(os.path.join(root, file))
                    print("remove ", name)
                    remove+=1
    print(count, remove)


def Lo1136iHelper():
    train_num = {'Spiculated': 91, 'Indistinct': 158, 'Microlobulated': 30, 'Obscured': 421, 'Circumscribed': 125}
    test_num = {'Spiculated': 19, 'Indistinct': 34, 'Microlobulated': 6, 'Obscured': 85, 'Circumscribed': 25}
    val_num = {'Spiculated': 15, 'Indistinct': 28, 'Microlobulated': 5, 'Obscured': 73, 'Circumscribed': 21}
    image_dir = '/usr/project/xtmp/mammo/rawdata/Aug2020/mammo_data/'
    csv_dir = '/usr/project/xtmp/mammo/rawdata/Aug2020/mammo_labels.csv'
    csv = pd.read_csv(csv_dir)
    image_names = [full_path.split('/')[-1] for full_path in csv['full path']]
    window_levels = csv['window level']
    margins = csv['lesion margin']
    for margin in margins:
        for cat in ["train", "test_DONOTTOUCH", 'validation']:
            save_dir = "/usr/xtmp/mammo/Lo1136i/" + cat + '/' + margin + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
    paths = []
    bounding_boxes = csv['bounding box']
    print('start going through dirs')
    # load image
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            path = os.path.join(root, file)
            paths.append(path)
    paths.sort()
    for path in paths:
        file = path.split("/")[-1]
        index = image_names.index(file)
        if index < 0:
            print('not found image with name ', file)
            continue
        image = read_16bit_img(path)
        # window level
        win_cen, win_width = ast.literal_eval(window_levels[index])
        print(win_cen, win_width, file)
        image = windowing(win_cen, win_width, image)
        # crop
        x1, y1, x2, y2 = ast.literal_eval(bounding_boxes[index])
        x1, y1, x2, y2 = max(0, min(x1, x2) - 100), max(0, min(y1, y2) - 100), \
                         min(image.shape[0], max(x1, x2) + 100), min(image.shape[1], max(y1, y2) + 100)
        roi = image[x1:x2, y1:y2]

        # save image
        margin = margins[index]
        if train_num[margin] > 0:
            np.save('/usr/xtmp/mammo/Lo1136i/train/' + margin + "/" + file[:-4], roi)
            train_num[margin] -= 1
        elif val_num[margin] > 0:
            np.save('/usr/xtmp/mammo/Lo1136i/validation/' + margin + "/" + file[:-4], roi)
            val_num[margin] -= 1
        elif test_num[margin]>0:
            np.save('/usr/xtmp/mammo/Lo1136i/test_DONOTTOUCH/' + margin + "/" + file[:-4], roi)
            test_num[margin] -= 1
        print('saved!')



if __name__ == "__main__":
    # Lo1136iHelper()
    # cropROI("/usr/project/xtmp/mammo/binary_Feb/five_classes_roi/train_context_roi/", augByWindow=False,
    #         datapath="/usr/project/xtmp/mammo/rawdata/Jan2020/PenRad_Dataset_SS_Final/sorted_by_mass_edges_Jan_in/train/",
    #         csvpath="/usr/project/xtmp/mammo/rawdata/Jan2020/Anotation_Master_adj.xlsx")
    # cropROI("/usr/project/xtmp/mammo/binary_Feb/five_classes_roi/test_context_roi/", augByWindow=False,
    #         datapath="/usr/project/xtmp/mammo/rawdata/Jan2020/PenRad_Dataset_SS_Final/sorted_by_mass_edges_Jan_in/test/",
    #         csvpath="/usr/project/xtmp/mammo/rawdata/Jan2020/Anotation_Master_adj.xlsx")

    # cropROI("/usr/project/xtmp/mammo/binary_Feb/five_classes_roi/train_context_roi/", augByWindow=False,
    #         datapath="/usr/project/xtmp/mammo/rawdata/Sept2019/JM_Dataset_Final/sorted_by_mass_edges_Sept/train/",
    #         csvpath="/usr/project/xtmp/mammo/rawdata/Sept2019/JM_Dataset_Final/no_PHI_Sept.xlsx")
    # cropROI("/usr/project/xtmp/mammo/binary_Feb/five_classes_roi/test_context_roi/", augByWindow=False,
    #         datapath="/usr/project/xtmp/mammo/rawdata/Sept2019/JM_Dataset_Final/sorted_by_mass_edges_Sept/test/",
    #         csvpath="/usr/project/xtmp/mammo/rawdata/Sept2019/JM_Dataset_Final/no_PHI_Sept.xlsx")

    # crop_negative_patches("/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/",
    #                       datapath="/usr/project/xtmp/mammo/rawdata/Jan2020/PenRad_Dataset_SS_Final/sorted_by_mass_edges_Jan_in/test/")

    # for margin in ["spiculated", "circumscribed", "obscured", "microlobulated", "indistinct"]:
    #     dataAugNumpy("/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_train_" + margin + "_augmented_by_win/", 50000,
    #             "/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_train_" + margin + "_augmented_crazy_with_rot/")
    # for margin in ["spiculated", "circumscribed", "obscured", "microlobulated", "indistinct"]:
    #     dataAugNumpy("/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_train_" + margin + "_augmented_by_win/", 10000,
    #             "/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_train_" + margin + "_augmented_more_with_rot/")

    # for pos in list(os.listdir("/usr/xtmp/mammo/binary_Feb/DDSM_five_class_augmented/")):
    #      move_to_binary(pos, "/usr/xtmp/mammo/binary_Feb/DDSM_five_class_augmented/",
    #                     "/usr/xtmp/mammo/binary_Feb/binary_context_roi/" + pos + "_DDSM_binary_train/")
    # for pos in list(os.listdir("/usr/xtmp/mammo/binary_Feb/DDSM_five_class_test/")):
    #     move_to_binary(pos, "/usr/xtmp/mammo/binary_Feb/DDSM_five_class_test/",
    #                    "/usr/xtmp/mammo/binary_Feb/binary_context_roi/" + pos + "_DDSM_binary_test/")

    # dirs = os.listdir("/usr/xtmp/mammo/binary_Feb/binary_context_roi/")
    # # print(dirs)
    # # remove_duplicates("/usr/xtmp/mammo/binary_Feb/binary_context_roi/binary_test_spiculated/")
    # dataAugNumpy(path="/usr/project/xtmp/mammo/binary_Feb/DDSM_five_class/",
    #              targetNumber=3000,
    #              targetDir="/usr/project/xtmp/mammo/binary_Feb/DDSM_five_class_augmented/",
    #              rot=True,
    #              skip=None)

    # print("start data augmenting")
    for pos in ["Spiculated","Circumscribed", "Indistinct", "Microlobulated", "Obscured"]:
        dataAugNumpy(
            path="/usr/xtmp/mammo/Lo1136i/train/",
            targetNumber=5000,
            targetDir="/usr/xtmp/mammo/Lo1136i/train_augmented_5000/",
            rot=True)
    # Fides_visualization(10)
    # Fidex_visualization_csv("fides_name_list_test.data")
    # move_DOI_to_training()
