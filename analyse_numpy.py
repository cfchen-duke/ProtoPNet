import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import argparse
import os


# check shape, zeros, visualization, and distribution
def analyze_image(test_dir, image_name):
    arr = np.load(test_dir)
    print(arr.shape)
    print("Non zero count is ", np.count_nonzero(arr))
    plt.imsave("visualization/" + image_name[:-4], arr, cmap="gray")
    whites = arr.shape[0] * arr.shape[1] - np.count_nonzero(np.round(arr - np.amax(arr), 4))
    print("white count is ", whites)
    print(arr.shape[0] * arr.shape[1] * 0.5)
    arr = arr.flatten()
    arr.sort()
    plt.plot(arr)
    plt.savefig("distribution/" + image_name[:-4])
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image', nargs=1, type=str, default='0')
    args = parser.parse_args()
    image_names = args.image[0]
    count = 0
    threshold = 20

    if image_names and image_names!="0":
        print(image_names)
        for image_name in image_names.split(" "):
            image_dir = "/usr/project/xtmp/mammo/binary_Feb/sorted_by_Fides_ratings/definite_augmented/definitely_positive/" + image_name
            analyze_image(image_dir, image_name)
    else:
        test_dir = "/usr/project/xtmp/mammo/Lo1136i/train/Spiculated/"
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if not file.endswith('.npy'):
                    continue
                path = os.path.join(root, file)
                analyze_image(path, file)
                if count > threshold:
                    return
                count += 1

if __name__ == "__main__":
    main()
