import numpy as np
import matplotlib.pyplot as plt
from pipeline.pipeline import TrainingImage
from Gen.testImageMaker import randCoord,makeCircle,makeSquare
from multiprocessing import Process,Pipe
import shutil,os

def makeImagesV3(filepath,numTriangles):
    bigImage = np.zeros((3,800,800))
    tinyImage= np.zeros((1,80,80))

    bigSquare = np.zeros((3,39,39))
    bigSquare[0] = np.ones((39,39))*255
    tinySquare= np.zeros((1,3,3))
    tinySquare[0]= np.ones((3,3))*255
    
    bigTriangle = np.zeros((3,39,39))
    for i in range(39):
        bigTriangle[2,i,:i+1] = np.ones(i+1)*255
    tinyTriangle = np.zeros((1,3,3))
    for i in range(3):
        tinyTriangle[0,i,:i+1] = np.ones(i+1)*255
    for i in range(numTriangles):
        x,y = np.random.randint(20,780,2)
        bigImage[:,y-19:y+20,x-19:x+20] = bigTriangle
        tinyImage[0,y/10-1:y/10+2,x/10-1:x/10+2] = tinyTriangle
    x,y = np.random.randint(20,780,2)
    bigImage[:,y-19:y+20,x-19:x+20] = bigSquare
    tinyImage[0,y/10-1:y/10+2,x/10-1:x/10+2] = tinySquare

    return [(bigImage,y*800+x),(tinyImage,y/200 * 4 + x/200)]

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
    fove, peri = loadImage(filepath,coords)
    return fove, peri

def loadImage(filepath,coords):
    img = TrainingImage(filepath,np.array(coords))
    foveImg = img.fovealImg
    foveInd = img.foveInd
    periImg = img.peripheryImg
    periInd = img.periInd
    return [(foveImg,foveInd),(periImg,periInd)]

def parallelizedLoops(numImgs,num,numDistractors,conn):
    # fovealImages = np.zeros((numImgs,800,800,4)) #FOVEA #This is the wrong shape for the network it should be (3,400,400)
    peripheryImages = np.zeros((numImgs,1,80,80))
    # fovealIndexes = np.zeros(numImgs) #FOVEA
    peripheryIndexes = np.zeros(numImgs)
    
    for i in range(numImgs):
        fove, peri = makeImagesV3('data/tmp/test%s.png'%num,numDistractors)
        # fovealImages[i] = fove[0] #FOVEA
        peripheryImages[i] = peri[0]
        # fovealIndexes[i] = fove[1] #FOVEA
        peripheryIndexes[i] = peri[1]

    conn.send((None,peripheryImages,None,peripheryIndexes))#(fovealImages,peripheryImages,fovealIndexes,peripheryIndexes)) #FOVEA
    return

def refactor(numImgs,numDistractors,iteration):
    numProcess = 4
    subNumImages = numImgs/numProcess
    parentConns, childConns = zip(*[Pipe() for _ in range(numProcess)])
    procs = [Process(target=parallelizedLoops, 
                     args=(subNumImages,_,numDistractors,childConns[_]))
                         for _ in range(numProcess)]
    for proc in procs:
        proc.start()
    data = []
    for proc,conn in zip(procs,parentConns):
        d = conn.recv()
        data.append(d)
        proc.join()

    # fovealImages = np.concatenate([data[_][0] for _ in range(numProcess)]) #FOVEA
    peripheryImages = np.concatenate([data[_][1] for _ in range(numProcess)])
    # fovealIndexes = np.concatenate([data[_][2] for _ in range(numProcess)]) #FOVEA
    peripheryIndexes = np.concatenate([data[_][3] for _ in range(numProcess)])

    # np.save('data/tmp/fovealImages{}.npy'.format(iteration),fovealImages) #FOVEA
    np.save('data/tmp/peripheryImages{}.npy'.format(iteration),peripheryImages)
    # np.save('data/tmp/fovealIndexes{}.npy'.format(iteration),fovealIndexes) #FOVEA
    np.save('data/tmp/peripheryIndexes{}.npy'.format(iteration),peripheryIndexes)


def main(numImgs,numDistractors):
    if not os.path.exists('data/tmp'):
        os.makedirs('data/tmp')
    
    subNumImages = 1000 #100 #FOVEA
    numIter = numImgs/subNumImages
    for i in range(numIter):
        refactor(subNumImages,numDistractors,i)
        print '{} out of {}'.format(i,numIter)

    dataKinds = ['peripheryImages','peripheryIndexes'] # ['fovealImages','peripheryImages','fovealIndexes','peripheryIndexes'] #FOVEA
    for dataType in dataKinds:
        d = np.concatenate([np.load('data/tmp/'+dataType+str(i)+'.npy') for i in range(numIter)])
        np.save('data/{}.npy'.format(dataType),d)
    shutil.rmtree('data/tmp/')

if __name__ == '__main__':
    main(20000,0)