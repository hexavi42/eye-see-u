import os
import argparse
from PIL import Image

def lop(img, fact):
    sx, sy = img.size
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
    return [x_offset_l, y_offset_t, x_offset_r, y_offset_b]

def main():
    parser = argparse.ArgumentParser(description='Downsample image by a factor or to a certain size')
    parser.add_argument('-p', '--path', type=str, default=None, help='filepath of image file to be analyzed')
    parser.add_argument('-f', '--factor', type=int, default=None, help='factor to downscale by')
    args = parser.parse_args()
    img = Image.open(args.path)
    filename, file_extension = os.path.splitext(args.path)

    fact = args.factor
    box = lop(img,fact)
    new_img = img.crop(box)
    thumbSize = [i/fact for i in new_img.size]
    new_img.thumbnail(thumbSize)
    new_img.save(filename+"Red{0}".format(fact),"PNG")
    return

if __name__ == "__main__":
    main()