import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', help='relative path to the images folder as described in CUB 200-2011')
    args = parser.parse_args()

    imgs_txt = Path(args.images_dir).joinpath('../images.txt')
    bb_txt = Path(args.images_dir).joinpath('../bounding_boxes.txt')
    train_test = Path(args.images_dir).joinpath('../train_test_split.txt')
    train_dir = Path(__file__).parent.joinpath('datasets/cub200_cropped/train_cropped')
    test_dir = Path(__file__).parent.joinpath('datasets/cub200_cropped/test_cropped')
    if not train_dir.exists():
        train_dir.mkdir(parents=True)
    if not test_dir.exists():
        test_dir.mkdir(parents=True)

    imgs_to_data = []
    with open(str(imgs_txt)) as imgs:
        img_index = imgs.readlines()
    with open(str(bb_txt)) as bb:
        bb_index = bb.readlines()
    with open(str(train_test)) as tt:
        tt_index = tt.readlines()

    for i, line in enumerate(img_index):
        n1, filename = line.strip().split(' ')
        n2, x, y, width, height = bb_index[i].strip().split(' ')
        n3, is_train = tt_index[i].strip().split(' ')
        if n1 != n2 or n2 != n3:
            raise Exception('something went wrong and indexing on images.txt/bounding_boxes.txt/train_test_split.txt is off')
        imgs_to_data.append([
            Path(args.images_dir).joinpath(filename),
            int(float(x)),
            int(float(y)),
            int(float(width)),
            int(float(height)),
            bool(int(is_train)),
        ])

    for path, x, y, w, h, is_train in imgs_to_data:
        im = cv2.imread(str(path))
        # crop and save
        im = im[y:y+h, x:x+w, :]

        if is_train:
            final_dir = train_dir.joinpath(path.parent.name)
        else:
            final_dir = test_dir.joinpath(path.parent.name)
        final_path = final_dir.joinpath(path.name)
        if not final_dir.exists():
            final_dir.mkdir()

        cv2.imwrite(str(final_path), im)


if __name__ == "__main__":
    main()
