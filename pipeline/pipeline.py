import scipy.misc as misc
import matplotlib.pyplot as plt

class InputImage:
    """
    Input Image class. Given a filepath to an image it will automatically
    load the image and build a low res gray scale version of the image by
    first lopping the image to match a factor (hard coded to 8), then
    grayscaling the image and finally reducing the size by a factor of 8.
    """
    #TODO: Add information about where the stimulus is located, for training.
    def __init__(self,filepath):
        self.fovealImg    = misc.imread(filepath)
        self.peripheryImg = self._buildPeriphery()
        return

    def _buildPeriphery(self):
        loppedImg = self._lop()
        gray = self._toGray(loppedImg)
        reduced = misc.imresize(gray,1/8.)

        return reduced

    def _toGray(self,array):
        red  = array[:,:,0]
        green= array[:,:,1]
        blue = array[:,:,2]

        return 0.299*red + 0.587*green + 0.114*blue

    def _lop(self, fact=8):
        width, height = self.fovealImg.shape[0:2]
        left,right,top,bottom = (0, width, 0, height)
        
        if width % fact != 0:
            x_offset = width % fact
            left, right = (x_offset/2, width - x_offset/2)
            if x_offset % 2 != 0:
                right -= 1
        
        if height % fact != 0:
            y_offset = height % fact
            top, bottom = (y_offset/2, height - y_offset/2)
            if y_offset % 2 != 0:
                bottom -= 1
        
        loppedImg = self.fovealImg[left:right+1,top:bottom+1,:]
        return loppedImg

if __name__ == '__main__':
    test = InputImage('test.png')
    
    plt.subplot(211)
    plt.imshow(test.fovealImg)
    plt.subplot(212)
    plt.imshow(test.peripheryImg,cmap='gray')
    plt.show()