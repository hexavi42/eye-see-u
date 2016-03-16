import argparse
from os import path
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Grayscale an Image')
    parser.add_argument('-p', '--path', type=str, default=None, help='filepath of image file to be analyzed')
    parser.add_argument('-o', '--outFile', type=str, default=None, help='filepath to save new image at')
    args = parser.parse_args()
    filename, file_extension = path.splitext(args.path)

    img = Image.open(args.path)
    newImg = img.convert("L")

    if args.outFile is not None:
        filename, file_extension = path.splitext(args.outFile)
    else:
        filename, file_extension = path.splitext(args.path)
        filename+="_Gray"

    newImg.save(filename,"PNG")
    return

if __name__ == "__main__":
    main()