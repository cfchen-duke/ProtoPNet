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
    test_dir = "/usr/project/xtmp/mammo/binary_Feb/binary_context_roi/test_spiculated_temp/spiculated/" + image_name
    print(test_dir)
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


