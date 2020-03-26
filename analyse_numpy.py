import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import argparse

# check shape, zeros, visualization, and distribution

parser = argparse.ArgumentParser()
parser.add_argument('-image', nargs=1, type=str, default='0')
args = parser.parse_args()
image_names = args.image[0]

for image_name in image_names.split(" "):
    test_dir = "/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/binary_train_spiculated_augmented/spiculated/" + image_name
    arr = np.load(test_dir)
    print(arr.shape)
    print("Non zero count is " , np.count_nonzero(arr))
    plt.imsave("visualization " + image_name[:-4], arr, cmap="gray")
    whites =  arr.shape[0] * arr.shape[1] - np.count_nonzero(np.round(arr - np.amax(arr), 4))
    print("white count is ", whites)
    print(arr.shape[0] * arr.shape[1] * 0.5)
    arr = arr.flatten()
    arr.sort()
    plt.plot(arr)
    plt.savefig("distribution " + image_name[:-4])
    plt.close()


# DP_AAHR_R_CC_1#0#3.npy
#DP_AAYY_R_MLO_5#0#4.npy
#DP_ABAC_L_MLO_3#1#0.npy
# DP_ABBQ_L_MLO_3#1#3.npy
#DP_ACYS_L_CC_2#0#0.npy
# DP_AGCW_R_CC_5#0#0.npy   DP_AIDR_L_MLO_3#0#3.npy   DP_AJZL_L_MLO_3#0#3.npy