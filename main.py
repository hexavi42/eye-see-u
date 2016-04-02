import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import nnModels
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainPeri', help='Train the peripheral network', action='store_true')
    parser.add_argument('--trainFove', help='Train the foveal network', action='store_true')
    args = parser.parse_args()

    if args.trainPeri:
        # Train PeripheryNet
        periModel = nnModels.PeripheryNet()
        periModel.fit_generator(batch_size=128, samples_per_epoch=60032, nb_epoch=20)
        
        # Load Validation Data
        data    = np.load('data/peripheryImages.npy')
        answers = np.load('data/peripheryIndexes.npy')
        data, answers = processForValidation(data, answers, objHalf=4)

        # Validate Periphery Net
        predictions = periModel.predict(data)
        right = np.sum(np.argmax(predictions, axis=1) == np.argmax(answers, axis=1))
        print("First choice cases: {0}".format(float(right)/len(predictions)))

        name = input("If you'd like to save the weights, please enter a savefile name now: ")
        if name:
            periModel.save(name)

    if args.trainFove:
        # Train FovealNet
        foveModel = nnModels.FoveaNet()
        foveModel.fit_generator(batch_size=128, samples_per_epoch=12800, nb_epoch=100)

        # Load Validation Data
        data    = np.load('data/fovealImages.npy')
        answers = np.load('data/fovealIndexes.npy')
        data, answers = processForValidation(data, answers, objHalf=20)

        # Validate Foveal Net
        predictions = foveModel.predict(data)
        right = np.sum(np.argmax(predictions, axis=1) == np.argmax(answers, axis=1))
        print("First choice cases: {0}".format(float(right)/len(predictions)))

        name = input("If you'd like to save the weights, please enter a savefile name now: ")
        if name:
            foveModel.save(name)
    return


if __name__ == '__main__':
    main()
