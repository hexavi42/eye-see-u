import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process,Pipe
import shutil,os

def makeImages(numTriangles):
    imgSize = 800
    stimSize = imgSize//20 - 1

    bigImage = np.ones((3, imgSize,    imgSize   ), dtype=np.uint8)*255
    tinyImage= np.ones((1, imgSize//10, imgSize//10), dtype=np.uint8)*255

    #Make stimuli
    bigSquare = np.ones((3, stimSize, stimSize), dtype=np.uint8)*255
    bigSquare[0] = np.zeros((stimSize, stimSize), dtype=np.uint8)
    tinySquare= np.ones((1, stimSize//10, stimSize//10), dtype=np.uint8)*255
    tinySquare[0]= np.zeros((stimSize//10, stimSize//10), dtype=np.uint8)
    
    bigTriangle = np.ones((3, stimSize, stimSize), dtype=np.uint8)*255
    for i in range(stimSize):
        bigTriangle[2, i, :i+1] = np.zeros(i+1, dtype=np.uint8)
    tinyTriangle = np.ones((1, stimSize//10, stimSize//10), dtype=np.uint8)*255
    for i in range(stimSize//10):
        tinyTriangle[0, i, :i+1] = np.zeros(i+1, dtype=np.uint8)

    #Place in Images
    bigBorder = (stimSize + 1)//2 #Avoid clipping edge
    tinyBorder = bigBorder//10
    for i in range(numTriangles):
        x,y = np.random.randint(bigBorder , imgSize - bigBorder, 2)
        bigImage[:, y-bigBorder+1:y+bigBorder, x-bigBorder+1:x+bigBorder] = bigTriangle
        tinyImage[0, y//10-tinyBorder+1:y//10+tinyBorder, x//10-tinyBorder+1:x//10+tinyBorder] = tinyTriangle
    x,y = np.random.randint(bigBorder , imgSize - bigBorder, 2)
    bigImage[:, y-bigBorder+1:y+bigBorder, x-bigBorder+1:x+bigBorder] = bigSquare
    tinyImage[0, y//10-tinyBorder+1:y//10+tinyBorder, x//10-tinyBorder+1:x//10+tinyBorder] = tinySquare

    loc = y//(imgSize//4) * 4 + x//(imgSize//4)
    return [(bigImage, loc),(tinyImage, loc)]

def parallelizedLoops(numImgs,num,numDistractors,conn):
    fovealImages = np.zeros((numImgs, 3, 800, 800), dtype=np.uint8)
    peripheryImages = np.zeros((numImgs, 1, 80, 80), dtype=np.uint8)
    fovealIndexes = np.zeros(numImgs, dtype=np.uint8)
    peripheryIndexes = np.zeros(numImgs, dtype=np.uint8)
    
    for i in range(numImgs):
        fove, peri = makeImages(numDistractors)
        fovealImages[i] = fove[0]
        peripheryImages[i] = peri[0]
        fovealIndexes[i] = fove[1]
        peripheryIndexes[i] = peri[1]

    conn.send((fovealImages,peripheryImages,fovealIndexes,peripheryIndexes))
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

    filenames = ['fovealImages{}'.format(iteration), 'peripheryImages{}'.format(iteration),
                 'fovealIndexes{}'.format(iteration), 'peripheryIndexes{}'.format(iteration)]

    for i,filename in enumerate(filenames):
        tmp = np.concatenate([ data[_][i] for _ in range(numProcess)])
        np.save('data/tmp/{}.npy'.format(filename), tmp)
    
def main(numImgs,numDistractors):
    if not os.path.exists('data/tmp'):
        os.makedirs('data/tmp')
    
    subNumImages = 100
    numIter = numImgs/subNumImages
    for i in range(numIter):
        refactor(subNumImages,numDistractors,i)
        print '{} out of {}'.format(i,numIter)

    dataKinds = ['fovealImages','peripheryImages','fovealIndexes','peripheryIndexes']
    for dataType in dataKinds:
        d = np.concatenate([np.load('data/tmp/'+dataType+str(i)+'.npy') for i in range(numIter)])
        np.save('data/{}.npy'.format(dataType),d)
    shutil.rmtree('data/tmp/')

if __name__ == '__main__':
    main(2500,20)