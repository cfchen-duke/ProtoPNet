import os
import numpy as np
from skimage.transform import resize
import ast
import pandas as pd
import png
import matplotlib
import matplotlib.pyplot as plt
import cv2
import torch
matplotlib.use("Agg")

load_model_dir = '/usr/project/xtmp/ct214/saved_models/vgg16/thresholdlogits25_lesion_512_0419/'
load_model_name = '10_3push0.9792.pth'
load_model_path = os.path.join(load_model_dir, load_model_name)
model = torch.load(load_model_path)

def slide(image, size, stride, heatmap):
    assert(len(image.shape)==2)
    for i in range(0, image.shape[0]-size, stride):
        for j in range(0, image.shape[1]-size, stride):
            roi = image[i:i+size, j:j+size]
            roi = resize(roi, (224, 224))
            roi = np.stack([roi,roi,roi])
            roi = torch.from_numpy(np.expand_dims(roi, axis=0)).float().cuda()
            print("roi shape is ", roi.shape)
            score, _ = model(roi)
            print(score)
            score = score.cpu()[0]
            print(score[1].item())
            heatmap[i:i+size, j:j+size] += score[1].item()


def window_adjustment(wwidth, wcen):
    if wcen == 2047 and wwidth == 4096:
        return wwidth, wcen
    else:
        new_wcen = np.random.randint(-100, 300)
        new_wwidth = np.random.randint(-200, 300)
        wwidth += new_wwidth
        wcen += new_wcen
        return wwidth, wcen


def lesionMap(save_dir, csvpath, datapath="/usr/project/xtmp/mammo/rawdata/Jan2020/PenRad_Dataset_SS_Final/sorted_by_mass_edges_Jan_in/test/"):
    """Make a heatmap lesion detection over a full-field mammogram."""
    df = pd.read_excel(csvpath)
    # classes = df["Class"]
    win_width = df['Win_Width']
    win_cen = df['Win_Center']
    names = list(df["File_Name"])
    did = set()
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

            # ds = dcm.read_file(path)
            # image = ds.pixel_array
            # image=np.ndarray

            wwidth = np.asarray(ast.literal_eval(win_width[i])).max()
            wcen = np.median(np.asarray(ast.literal_eval(win_cen[i])))

            wwidth, wcen = window_adjustment(wwidth, wcen)

            image = ((image - wcen) / wwidth) + 0.5
            image = np.clip(image, 0, 1)

            image_3d = np.stack([image, image, image])

            # slide through
            heatmap = np.zeros(shape=image.shape)
            slide(image, size=224, stride=100, heatmap=heatmap)
            slide(image, size=400, stride=100, heatmap=heatmap)
            print("heatmap non zero count ", np.count_nonzero(heatmap))


            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]
            image_3d = np.transpose(image_3d, (1, 2, 0))
            overlayed_img = 0.5 * image_3d + 0.3 * heatmap
            print(overlayed_img.shape)
            plt.imsave(save_dir + "lesion_map_"+ name, overlayed_img)

            raise


if __name__ == "__main__":
    lesionMap(save_dir="/usr/xtmp/ct214/test/",
              csvpath="/usr/project/xtmp/mammo/rawdata/Jan2020/Anotation_Master_adj.xlsx",
              datapath="/usr/project/xtmp/mammo/rawdata/Jan2020/PenRad_Dataset_SS_Final/sorted_by_mass_edges_Jan_in/test/spiculated/",
              )




