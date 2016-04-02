# Helper functions

import matplotlib.pyplot as plt
import numpy as np

def show_predictions(model_predict, data, answers):
    for i, j in enumerate(model_predict):
        plt.subplot(211)
        plt.imshow(data[i, 0, :, :], cmap='gray')
        plt.xlabel('We know the stimulus to be at position {}'.format(np.argmax(answers[i])))
        plt.subplot(212)
        color = ['b']*16
        color[np.argmax(model_predict[i])] = 'r'
        color[np.argmax(answers[i])] = 'g'
        plt.bar(range(16), j, color=color)
        plt.show()

def splitSectors(np_matrix, objHalf=20, numSectors=[4, 4]):
    sectors = [[] for i in range(numSectors[0]*numSectors[1])]
    for layer in np_matrix:
        padShape = np.array(layer.shape)+objHalf*2
        beforeSect = np.ones(padShape, dtype=np.uint8)*255
        size = layer.shape/np.array(numSectors)
        # error will happen if size*num < np_matrix.shape
        # currently not handled or needed
        beforeSect[objHalf:objHalf+layer.shape[0], objHalf:objHalf+layer.shape[1]] = layer
        for i in range(numSectors[0]):
            for j in range(numSectors[1]):
                sectors[i*4+j].append(beforeSect[i*size[0]:(i+1)*size[0]+objHalf*2, j*size[1]:(j+1)*size[1]+objHalf*2])
    return np.array(sectors)


def formRGBImage(np_matrix):
    assert np_matrix.shape[0] == 3 and len(np_matrix.shape) == 3,\
        "shape ({0}) of input matrix does not match (3, M, N)".format(np_matrix.shape)
    return np.stack([np_matrix[0], np_matrix[1], np_matrix[2]], axis=2)

def plotSector(sector):
    plt.imshow(formRGBImage(sector))
    plt.show()

def processForValidation(images, indexes, objHalf=20):
    data  = []
    answers=[]
    for img,ind in zip(images,indexes):
        sectors = splitSectors(img, objHalf=objHalf)
        for i,sec in enumerate(sectors):
            ans = np.array([1, 0]) if i==ind else np.array([0, 1])
            data.append(sec)
            answers.append(ans)
    return np.array(data), np.array(answers)

def visualizeKernels(model, wFname, layerNum):
    model.load(wFname)
    layer = model.model.layers[layerNum]
    kernels = layer.get_weights()[0]
    side = (len(kernels)**0.5 + 1) //1
    for i,k in enumerate(kernels):
        plt.subplot(side,side,1+i)
        plt.imshow(k[0], cmap='gray', interpolation='nearest')
    plt.show()
    return