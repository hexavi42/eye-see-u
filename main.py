import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import nnModels

# fix for python 2->3
try:
    input = raw_input
except NameError:
    pass

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
    sectors = []
    for layer in np_matrix:
        layer_sect = []
        padShape = np.array(layer.shape)+objHalf*2
        beforeSect = np.zeros(padShape, dtype=np.int8)
        size = layer.shape/np.array(numSectors)
        # error will happen if size*num < np_matrix.shape
        # currently not handled or needed
        beforeSect[objHalf:objHalf+layer.shape[0], objHalf:objHalf+layer.shape[1]] = layer
        for i in range(numSectors[0]):
            for j in range(numSectors[1]):
                layer_sect.append(beforeSect[i*size[0]:(i+1)*size[0]+objHalf*2, j*size[1]:(j+1)*size[1]+objHalf*2])
        sectors.append(np.array(layer_sect))
    retShape = [numSectors[0]*numSectors[1], np_matrix.shape[0], size[0]+objHalf*2, size[1]+objHalf*2]
    return np.array(sectors).reshape(retShape)


def plotSector(sector):
    plt.imshow(sector.reshape([sector.shape[1], sector.shape[2], sector.shape[0]]))
    plt.show()


def main():
    # Fetch Data
    data = np.load('data/peripheryImages.npy')
    answers = np.load('data/peripheryIndexes.npy')
    answers = np_utils.to_categorical(answers, 16)

    periModel = nnModels.PeripheryNet()
    periModel.fit(data[:len(data)*3/4], answers[:len(answers)*3/4], nb_epoch=30, batch_size=128)

    predictions = periModel.predict(data[len(data)*3/4:])
    right = 0
    topHalf = 0
    for i, j in enumerate(predictions):
        if np.argmax(j) == np.argmax(answers[i]):
            right += 1
        if np.argmax(answers[i]) in np.argpartition(j, -8)[-8:]:
            topHalf += 1
    print("First choice cases: {0}".format(float(right)/len(predictions)))
    print("Top half of cases: {0}".format(float(topHalf)/len(predictions)))
    show_predictions(predictions[:10], data, answers)
    name = input("If you'd like to save the weights, please enter a savefile name now: ")
    if name:
        periModel.save(name)
    return


if __name__ == '__main__':
    main()