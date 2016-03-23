import numpy as np
import matplotlib.pyplot as plt
from pipeline.pipeline import TrainingImage
from Gen.testImageMaker import randCoord,makeCircle,makeSquare

def makeImagesV2(filepath,numCircles):
    fig=plt.figure(figsize=[8,8])
    axis=fig.gca()
    plt.xticks([])
    plt.yticks([])
    for _ in range(numCircles):
        coords=randCoord()
        makeCircle(0.03,coords,axis)
    coords=randCoord()
    makeSquare(0.05,coords,axis)    
    plt.tight_layout(pad=0,h_pad=0,w_pad=0)
    plt.savefig(filepath)
    plt.close()
    return coords

def loadImage(filepath,coords):
    img = TrainingImage(filepath,np.array(coords))
    foveImg = img.fovealImg
    foveInd = img.foveInd
    periImg = img.peripheryImg
    periInd = img.periInd
    return [(foveImg,foveInd),(periImg,periInd)]

def loadingBar(totalNumberOfActions,i):
    bars = int(float(i)/totalNumberOfActions * 100.)
    space = 100-bars
    print "["+"+"*bars+' '*space+"]"
    return

def main(numImgs):
    fovealImages = []
    peripheryImages = []
    fovealIndexes = []
    peripheryIndexes = []
    for _ in range(numImgs):
        coords = makeImagesV2('data/test.png',20)
        fove, peri = loadImage('data/test.png',coords)
        fovealImages.append(fove[0])
        peripheryImages.append(peri[0])
        fovealIndexes.append(fove[1])
        peripheryIndexes.append(peri[1])
        loadingBar(numImgs,_)
    np.array(fovealImages)
    np.array(peripheryImages)
    np.array(fovealIndexes)
    np.array(peripheryIndexes)

    np.save('data/fovealImages.npy',fovealImages)
    np.save('data/peripheryImages.npy',peripheryImages)
    np.save('data/fovealIndexes.npy',fovealIndexes)
    np.save('data/peripheryIndexes.npy',peripheryIndexes)
        

if __name__ == '__main__':
    main(1000)