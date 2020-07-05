from fpdf import FPDF
import os
from matplotlib.pyplot import imsave, imread
import numpy as np
import cv2
from skimage.transform import resize
import ast
import pandas as pd
from sklearn.metrics import roc_auc_score
import png
import torch


class PDF(FPDF):
    def __init__(self, dir):
        FPDF.__init__(self)
        #/usr/project/xtmp/ct214/model_visualization_result/resnet34/MassMarginROI_1028_3/resume_from_140/140_3push0.7310.pth/JMAFE_4_RMLO_D-9.npy/
        self.id = dir.split("/")[-2][:-4]
        print("visualizing", self.id)
        # class is hard coded for now
        self.model_class = "lesion"


    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def lookup(patient_id, target, prototype_index, excel_dir, file_dir):
    """
    Look up patient in the database, crop out the region of interest, original image with rectangle,
    and ROI to the current dir
    :param patient: patient ID
    :return: size of ROI, class
    """
    df = pd.read_excel(excel_dir)
    locations = df['Box_List']
    win_width = df['Win_Width']
    win_cen = df['Win_Center']
    names = list(df["File_Name"])
    test_image_name = patient_id + ".png"
    i = names.index(test_image_name)
    for root, dir, files in os.walk(file_dir):
        for file in files:
            # find the index of the name
            path = os.path.join(root, file)
            temp = file.split("_")
            name = temp[-4][-5:] + "_" + temp[-3] + "_" + temp[-2] + "_" + temp[-1]
            if name != test_image_name:
                continue
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
            x1, y1, x2, y2 = location[0:4]
            x1, y1, x2, y2 = max(0, min(x1, x2) - 100), max(0, min(y1, y2) - 100), \
                             min(image.shape[0], max(x1, x2) + 100), min(image.shape[1], max(y1, y2) + 100)
            # x1, y1 = midx - target_size//2, midy - target_size//2
            # x2, y2 = x1 + target_size, y1 + target_size
            start_point, end_point, thickness, color = (y1, x1), (y2, x2), 5, (0, 255, 0)
            roi = image[x1:x2, y1:y2]
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            image = np.rot90(image, k=3)
            imsave(target + "original_prototype_" + str(prototype_index) + ".png", image, cmap="gray")
            imsave(target + "roi_prototype_" + str(prototype_index) + ".png", roi, cmap='gray')
            print(np.amax(image), np.amin(image))
            print("Successfully save original for ", patient_id)
            break
    return [x2-x1, y2-y1]



def generate_pdf(test_dir, num_of_protos, excel_dir, file_dir, classes=['allneg', 'other'], ):
    """
    Generate a pdf visualization of prototypes for a specific testing image
    :param test_dir: the saving directory of local_analysis.py
    :param num_of_protos: number of visualized protos you want to see
    :return: a pdf
    """
    pdf = PDF(dir=test_dir)
    pdf.alias_nb_pages()
    pdf.add_page()

    # title
    # Arial bold 15
    pdf.set_font('Arial', 'B', 13)
    # Title
    pdf.cell(0, 0, 'Visualization for ' + pdf.model_class + " ID " + pdf.id, 0, 1, 'C')

    pdf.set_font('Times', '', 12)
    # original image
    pdf.cell(0,10,"Original Breast Image is: ", 0, 1)
    pdf.image(test_dir + "original_" + pdf.id +".png", h=250)
    pdf.ln(20)

    # ROI image
    pdf.cell(0,10,"Zoom in on the cropped ROI: ", 0, 1)
    pdf.image(test_dir + "original_part_" + pdf.id +".png", w=100)
    pdf.ln(20)

    # prototypes with heatmap (itself and ROI)
    # avoid identical prototypes
    seen_id = set()
    for i in range(1, 20):
        with open(test_dir + "most_activated_prototypes/top-" + str(i) + "_activated_prototype_lookup.txt", "r") as f:
            line = f.readline()
            patient_id = line[:-3]
            class_info = line[-1]
        classes.sort()
        class_info = classes[int(class_info)]

#        if class_info=="other":
#            continue
        if patient_id in seen_id:
            continue
        seen_id.add(patient_id)
        # get size
        try:
          [h, w]= lookup(patient_id, test_dir + "/most_activated_prototypes/" , i, excel_dir, file_dir)
        except ValueError:
          continue
        pdf.add_page()
        pdf.cell(0,10,"activation map of ROI by prototype " + str(i) + ":", 0, 1)
        # read two images and concatenate
        # roi_heat = np.concatenate((roi,heat), axis=1)
        # imsave(test_dir + "roi+heat"+ str(i) + ".png", roi_heat)
        pdf.image(test_dir + "original_part_" + pdf.id + ".png", x=10, y=None, w=80, h=80)
        pdf.image(test_dir + "most_activated_prototypes/prototype_activation_map_by_top-" + str(i) + "_prototype.png",
                  x=100, y=20, w=80, h=80)

        # get similarity score
        with open(test_dir + "most_activated_prototypes/top-" + str(i) + "_activated_prototype.txt","r") as f:
            similarity = f.readline()

        pdf.cell(0, 10, "prototype " + str(i) + " of class " + class_info + " with similarity score " + similarity[:5] +
                 "(patient id " + patient_id + "):", 0, 1)
        pdf.image(test_dir + "/most_activated_prototypes/" + "roi_prototype_" + str(i) + ".png",
                  x=10, y=None,  w=80, h= h*80//w)

        # calculate heatmap
        original_roi = imread(test_dir + "/most_activated_prototypes/" + "roi_prototype_" + str(i) + ".png")
        generated_by_push_heat = imread(test_dir + "most_activated_prototypes/top-" + str(i) +
                                   "_activated_prototype_self_act.png")
        generated_by_push = imread(test_dir + "most_activated_prototypes/top-" + str(i) +
                                   "_activated_prototype.png")
        # heatmap = (generated_by_push_heat - 0.5 * generated_by_push)
        heatmap = (generated_by_push_heat -  generated_by_push)
        # original_roi = resize(original_roi, (224,224))
        generated_heatmap = heatmap/0.3
        imsave(test_dir + "/most_activated_prototypes/" + "roi_prototype_" + str(i) + "heatmap.png", generated_heatmap,
               cmap="gray")

        # load heatmap
        pdf.image(test_dir + "/most_activated_prototypes/" + "roi_prototype_" + str(i) + "heatmap.png",
                  x=100, y=110, w=80, h= h*80//w)
        pdf.ln(30)
        pdf.image(test_dir + "most_activated_prototypes/original_prototype_" + str(i) + ".png", h=250)

        if len(seen_id) == num_of_protos:
            break
    pdf.output(test_dir + pdf.id + '_visualization.pdf', 'F')
    print("pdf created")
    print("saved in ", test_dir)


def draw_box():
    # load image
    base_dir_J = "/usr/project/xtmp/mammo/rawdata/Sept2019/JM_Dataset_Final/sorted_by_mass_edges_Sept/test/"
    base_dir_D = "/usr/project/xtmp/mammo/rawdata/Jan2020/PenRad_Dataset_SS_Final/sorted_by_mass_edges_Jan_in/test/"
    names = ["DP_ADZW_L_CC_2", "DP_AFEX_R_CC_1", "DP_AFXO_L_CC_2"]

    for name in names:
        if name[0] == "J":
            base_dir = base_dir_J
            df = pd.read_excel("/usr/project/xtmp/mammo/rawdata/Sept2019/JM_Dataset_Final/no_PHI_Sept.xlsx")
        else:
            base_dir = base_dir_D
            df = pd.read_excel("/usr/project/xtmp/mammo/rawdata/Jan2020/Anotation_Master_adj.xlsx")

        csv_names = list(df["File_Name"])
        win_width = df['Win_Width']
        win_cen = df['Win_Center']
        locations = df['Box_List']
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if name in file:
                    i = csv_names.index(name+".png")
                    # read image into np
                    path = os.path.join(root, file)
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

                    # read locations
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
                    for j in range(len(location)//4):
                        # if j not in mass_index:
                        #     continue
                        x1, y1, x2, y2 = location[4 * j:4 * (j + 1)]
                        x1, y1, x2, y2 = max(0, min(x1, x2) - 100), max(0, min(y1, y2) - 100), \
                                         min(image.shape[0], max(x1, x2) + 100), min(image.shape[1], max(y1, y2) + 100)
                        start_point = (y1, x1)
                        end_point = (y2, x2)
                        color = (0, 255, 0)
                        thickness = 5
                        print(image.shape)
                        image = cv2.rectangle(image, start_point, end_point, color, thickness)
                    image = np.rot90(image, k=3)
                    imsave(name, image, cmap="gray")
                    print("successfully saved ", name)




if __name__ == "__main__":
    # Instantiation of inherited class
    #test_dir = "/usr/project/xtmp/ct214/model_visualization_result/spiculated/resnet34/MassMarginROI_1028_3/resume_from_140/140_3push0.7310.pth/JMAFE_4_RMLO_D-9.npy/"
    # test_dir = ""[
    # num_of_protos = 3
    # generate_pdf(test_dir, num_of_protos)
    draw_box()

