import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process,Pipe
import shutil,os
import random

def makeSquare(layers,length):
    square = np.ones((layers, length, length), dtype=np.uint8)*255
    square[0] = np.zeros((length, length), dtype=np.uint8)
    return square

def makeTriangle(layers, height):
    triangle = np.ones((layers, height, height), dtype=np.uint8)*255
    ind = 2 if layers>1 else 0
    for i in range(height):
        triangle[ind, i, :i+1] = np.zeros(i+1, dtype=np.uint8)
    return triangle

def makePeriImages(triLoc, squaLoc, imgSize=800):
    stimSize = imgSize//20 - 1
    tinyImage= np.ones((1, imgSize//10, imgSize//10), dtype=np.uint8)*255

    #Make stimuli
    tinySquare   = makeSquare(1, stimSize//10)
    tinyTriangle = makeTriangle(1, stimSize//10)

    #Place in Images
    bigBorder = (stimSize + 1)//2 #Avoid clipping edge
    tinyBorder = bigBorder//10
    for x,y in triLoc:
        tinyImage[0, y//10-tinyBorder+1:y//10+tinyBorder, x//10-tinyBorder+1:x//10+tinyBorder] = tinyTriangle
    x,y = squaLoc
    tinyImage[0, y//10-tinyBorder+1:y//10+tinyBorder, x//10-tinyBorder+1:x//10+tinyBorder] = tinySquare

    loc = y//(imgSize//4) * 4 + x//(imgSize//4)
    return (tinyImage, loc)

def makeFoveImages(triLoc, squaLoc, imgSize=800):
    stimSize = imgSize//20 - 1
    bigImage = np.ones((3, imgSize,    imgSize   ), dtype=np.uint8)*255

    #Make stimuli
    bigSquare   = makeSquare(3, stimSize)
    bigTriangle = makeTriangle(3, stimSize)

    #Place in Images
    bigBorder = (stimSize + 1)//2 #Avoid clipping edge
    tinyBorder = bigBorder//10
    for x,y in triLoc:
        bigImage[:, y-bigBorder+1:y+bigBorder, x-bigBorder+1:x+bigBorder] = bigTriangle
    x,y = squaLoc
    bigImage[:, y-bigBorder+1:y+bigBorder, x-bigBorder+1:x+bigBorder] = bigSquare

    loc = y//(imgSize//4) * 4 + x//(imgSize//4)
    return (bigImage, loc)

def parallelizedLoops(numImgs,numDistractors,conn,makeFove=True,makePeri=True):
    np.random.seed(random.randint(0,4294967295))
    triLocs  = np.random.randint(20, 780, (numImgs ,numDistractors, 2))
    squaLocs = np.random.randint(20, 780, (numImgs, 2))
    
    results = {}
    if makeFove:
        results['fovealImages'] = np.zeros((numImgs, 3, 800, 800), dtype=np.uint8)
        results['fovealIndexes'] = np.zeros(numImgs, dtype=np.uint8)
        for i in range(numImgs):
            fove = makeFoveImages(triLocs[i],squaLocs[i])
            results['fovealImages'][i] = fove[0]
            results['fovealIndexes'][i] = fove[1]

    if makePeri:
        results['peripheryImages'] = np.zeros((numImgs, 1, 80, 80), dtype=np.uint8)
        results['peripheryIndexes'] = np.zeros(numImgs, dtype=np.uint8)
        for i in range(numImgs):
            peri = makePeriImages(triLocs[i],squaLocs[i])
            results['peripheryImages'][i] = peri[0]
            results['peripheryIndexes'][i] = peri[1]

    conn.send(results)
    return

def refactor(numImgs, numDistractors, makeFove, makePeri, iteration):
    numProcess = 4
    subNumImages = numImgs/numProcess
    parentConns, childConns = zip(*[Pipe() for _ in range(numProcess)])
    procs = [Process(target=parallelizedLoops, 
                     args=(subNumImages,numDistractors,childConns[_],makeFove,makePeri))
                         for _ in range(numProcess)]
    for proc in procs:
        proc.start()
    
    data = {}
    if makeFove:
        data['fovealImages'] = []
        data['fovealIndexes'] = []
    if makePeri:
        data['peripheryImages'] = []
        data['peripheryIndexes'] = []

    for proc,conn in zip(procs,parentConns):
        d = conn.recv()
        for key in d:
            data[key].append(d[key])
        proc.join()

    for key in data:
        np.save('data/tmp/%s%s.npy'%(key,iteration), np.concatenate(data[key]))
    
def main(numImgs, numDistractors, makeFove=True, makePeri=True):
    if not os.path.exists('data/tmp'):
        os.makedirs('data/tmp')
    
    subNumImages = 100
    numIter = numImgs/subNumImages
    for i in range(numIter):
        refactor(subNumImages,numDistractors,makeFove,makePeri,i)
        print '{} out of {}'.format(i,numIter)

    dataKinds = []
    if makeFove:
        dataKinds.append('fovealImages')
        dataKinds.append('fovealIndexes')
    if makePeri:
        dataKinds.append('peripheryImages')
        dataKinds.append('peripheryIndexes')
    
    for dataType in dataKinds:
        d = np.concatenate([np.load('data/tmp/'+dataType+str(i)+'.npy') for i in range(numIter)])
        np.save('data/{}.npy'.format(dataType),d)
    shutil.rmtree('data/tmp/')

if __name__ == '__main__':
    main(1200,20,makeFove=False,makePeri=True)