import scipy.misc as misc
import matplotlib.pyplot as plt
import numpy as np

class InputImage:
    """
    Input Image class. Given a filepath to an image it will automatically
    load the image and build a low res gray scale version of the image by
    first lopping the image to match a factor (hard coded to 10), then
    grayscaling the image and finally reducing the size by a factor of 10.
    """
    def __init__(self,filepath):
        self.fovealImg    = misc.imread(filepath)
        self.peripheryImg = self._buildPeriphery()
        return

    def _buildPeriphery(self):
        loppedImg = self._lop()
        gray = self._toGray(loppedImg)
        reduced = misc.imresize(gray,1/10.)
        reduced = reduced[np.newaxis,:,:]
        return reduced

    def _toGray(self,array):
        red  = array[:,:,0]
        green= array[:,:,1]
        blue = array[:,:,2]

        return 0.299*red + 0.587*green + 0.114*blue

    def _lop(self, fact=10):
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

class TrainingImage(InputImage):
    def __init__(self,filepath,stimLoc,testing=False):
        self.fovealImg    = misc.imread(filepath)
        self.peripheryImg = self._buildPeriphery()

        self.stimLocFoveal = map(int,stimLoc*800)
        self.stimLocPeriphery = map(int,stimLoc*4)

        self.foveInd = self._extractIndex(self.stimLocFoveal,800)
        self.periInd = self._extractIndex(self.stimLocPeriphery,4)
        
        self.periTrainVec = np.zeros(16)
        self.periTrainVec[self.periInd] = 1
        self.foveTrainVec = np.zeros(800*800)
        self.foveTrainVec[self.foveInd] = 1

        if testing==True:
            self.periTrainImg = np.zeros((4,4))
            self.periTrainImg[3-self.stimLocPeriphery[1],self.stimLocPeriphery[0]] = 1
            print stimLoc
            plt.subplot(221)
            plt.imshow(self.fovealImg[:,:,:3])
            plt.subplot(222)
            plt.imshow(self.peripheryImg[0])
            plt.subplot(223)
            plt.imshow(self.periTrainImg)
            plt.show()
        return

    def _extractIndex(self,stim,size):
        return stim[0]*size + stim[1]

if __name__ == '__main__':
    test = InputImage('test.png')
    
    plt.subplot(211)
    plt.imshow(test.fovealImg)
    plt.subplot(212)
    #plt.imshow(test.peripheryImg,cmap='gray')
    plt.show()

    test = TrainingImage('test.png',np.random.random(2),testing=True)
    print test.stimLocPeriphery,test.stimLocFoveal