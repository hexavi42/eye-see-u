import numpy as np
import matplotlib.pyplot as plt
from pipeline.pipeline import TrainingImage
from Gen.testImageMaker import randCoord,makeCircle,makeSquare
from multiprocessing import Process,Pipe
import shutil,os

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

def parallelizedLoops(numImgs,num,conn):
    # fovealImages = np.zeros((numImgs,800,800,4)) #FOVEA #This is the wrong shape for the network it should be (3,400,400)
    peripheryImages = np.zeros((numImgs,1,80,80))
    # fovealIndexes = np.zeros(numImgs) #FOVEA
    peripheryIndexes = np.zeros(numImgs)
    
    for i in range(numImgs):
        coords = makeImagesV2('data/tmp/test%s.png'%num,0) #20 circles or 0 circles
        fove, peri = loadImage('data/tmp/test%s.png'%num,coords)
        # fovealImages[i] = fove[0] #FOVEA
        peripheryImages[i] = peri[0]
        # fovealIndexes[i] = fove[1] #FOVEA
        peripheryIndexes[i] = peri[1]

    conn.send((None,peripheryImages,None,peripheryIndexes))#(fovealImages,peripheryImages,fovealIndexes,peripheryIndexes)) #FOVEA
    return

def refactor(numImgs,iteration):
    numProcess = 4
    subNumImages = numImgs/numProcess
    parentConns, childConns = zip(*[Pipe() for _ in range(numProcess)])
    procs = [Process(target=parallelizedLoops, 
                     args=(subNumImages,_,childConns[_]))
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


def main(numImgs):
    if not os.path.exists('data/tmp'):
        os.makedirs('data/tmp')
    
    subNumImages = 1000 #100 #FOVEA
    numIter = numImgs/subNumImages
    for i in range(numIter):
        refactor(subNumImages,i)
        print '{} out of {}'.format(i,numIter)

    dataKinds = ['peripheryImages','peripheryIndexes'] # ['fovealImages','peripheryImages','fovealIndexes','peripheryIndexes'] #FOVEA
    for dataType in dataKinds:
        d = np.concatenate([np.load('data/tmp/'+dataType+str(i)+'.npy') for i in range(numIter)])
        np.save('data/{}.npy'.format(dataType),d)
    shutil.rmtree('data/tmp/')

if __name__ == '__main__':
    main(10000)