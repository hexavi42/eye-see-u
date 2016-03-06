# downsample.py
# take a file input, downsample it to a size / by a factor,
# dump downsampled image to [image name]_[factor]_small somewhere
import argparse
import os
import numpy as np
from scipy import ndimage
from scipy.misc import imsave, imread


def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res


# lops off chunks of image so that it's divisible by a factor
def lop(ar, fact):
    sx, sy = ar.shape
    [x_offset_l, x_offset_r, y_offset_t, y_offset_b] = [0, sx, 0, sy]
    if sx % fact != 0:
        x_offset = sx % fact
        [x_offset_l, x_offset_r] = [x_offset/2, sx - x_offset/2]
        if x_offset % 2 != 0:
            x_offset_r -= 1
    if sy % fact != 0:
        y_offset = sy % fact
        [y_offset_t, y_offset_b] = [y_offset/2, sy - y_offset/2]
        if y_offset % 2 != 0:
            y_offset_b -= 1
    new_ar = ar[x_offset_l:x_offset_r, y_offset_t:y_offset_b]
    return new_ar


def main():
    parser = argparse.ArgumentParser(description='Downsample image by a factor or to a certain size')
    parser.add_argument('-p', '--path', type=str, default="/Users/harveytang/Desktop/wallpapers/o1832683.jpg",
                        help='filepath of image file to be analyzed')
    parser.add_argument('-f', '--factor', type=int, default=None,
                        help='factor to downscale by')
    args = parser.parse_args()
    image = imread(args.path)
    img_name = filename, file_extension = os.path.splitext(args.path)

    crop = np.array([lop(image[..., layer], args.factor) for layer in range(image.shape[2])])
    print(crop.shape)
    down = np.array([block_mean(layer, args.factor) for layer in crop])
    print(down.shape)
    imsave('{0}_{1}_small.png'.format(img_name, args.factor), down)


if __name__ == "__main__":
    main()
