This code package implements the prototypical part network (ProtoPNet)
from the paper "This Looks Like That: Deep Learning for Interpretable
Image Recognition" by Chaofan Chen*, Oscar Li*, Chaofan Tao, Alina Jade Barnett,
Jonathan Su, and Cynthia Rudin (* denotes equal contribution).

This code package is jointly developed by Chaofan Chen (cfchen-duke)
and Oscar Li (OscarcarLi), and licensed under MIT License (see LICENSE
for more information regarding the use and the distribution of this code
package).

Prerequisites: PyTorch, NumPy, cv2, Augmentor (https://github.com/mdbloice/Augmentor)
Recommended hardware: 4 NVIDIA Tesla P-100 GPUs or 8 NVIDIA Tesla K-80 GPUs

Instructions for preparing the data:
1. Download the dataset CUB_200_2011.tgz from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
2. Unpack CUB_200_2011.tgz
3. Crop the images using information from bounding_boxes.txt (included in the
dataset)
4. Split the (cropped) images into training and test sets, using train_test_split.txt
5. Augment the training set using img_aug.py (included in this code package)

Instructions for training the model:
1. In settings.py, provide the appropriate strings for data_path, train_dir, test_dir,
train_push_dir:
(1) data_path is where the dataset resides
(2) train_dir is the directory containing the augmented training set (the output of
Step 5 in the instructions for preparing the data)
(3) test_dir is the directory containing the test set
(4) train_push_dir is the directory containing the original (unaugmented) training set
2. Run main.py

Instructions for finding the nearest prototypes to a test image:
1. Run local_analysis.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-modeldir is the directory containing the model you want to analyze
-model is the filename of the saved model you want to analyze
-imgdir is the directory containing the image you want to analyze
-img is the filename of the image you want to analyze
-imgclass is the (0-based) index of the correct class of the image

Instructions for finding the nearest patches to each prototype:
1. Run global_analysis.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-modeldir is the directory containing the model you want to analyze
-model is the filename of the saved model you want to analyze

Instructions for pruning the prototypes from a saved model:
1. Run run_pruning.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-modeldir is the directory containing the model you want to prune prototypes from
-model is the filename of the saved model you want to prune prototypes from
Note: the prototypes in the model must already have been projected (pushed) onto
the nearest latent training patches, before running this script

Instructions for combining several ProtoPNet models (Jupyter Notebook required):
1. Run the Jupyter Notebook combine_models.ipynb
