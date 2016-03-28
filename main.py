import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt

# fix for python 2->3
try:
    input = raw_input
except NameError:
    pass


class PeripheryNet(object):
    model = None

    def __init__(self, input_shape=[1, 80, 80], sectors=16):
        # Build model
        periModel = Sequential()
        periModel.add(Convolution2D(4, 8, 8, input_shape=input_shape, init='uniform'))
        periModel.add(Activation('relu'))
        periModel.add(Dropout(0.1))
        # periModel.add(MaxPooling2D(pool_size=(4, 4)))
        periModel.add(Flatten())
        periModel.add(Dense(output_dim=sectors))
        periModel.add(Activation('softmax'))

        sgd = SGD(lr=1e-5, momentum=0.9, nesterov=True)
        periModel.compile(optimizer=sgd, loss='categorical_crossentropy')
        self.model = periModel

    def fit(self, data, answers, nb_epoch=3, batch_size=128):
        self.model.fit(data, answers, nb_epoch=nb_epoch, batch_size=batch_size, show_accuracy=True)

    def predict(self, data):
        return self.model.predict(data)

    def save(self, fname, overwrite=False):
        self.model.save_weights(fname, overwrite=overwrite)

    def load(self, fname):
        self.model.load_weights(fname)


class FoveaNet(object):
    model = None

    def __init__(self, input_shape=[3, 219, 219]):
        # Build model
        fovModel = Sequential()
        fovModel.add(Convolution2D(3, 15, 15, input_shape=input_shape, init='normal'))
        fovModel.add(Activation('relu'))
        fovModel.add(Dropout(0.2))
        fovModel.add(MaxPooling2D(pool_size=(2, 2)))
        fovModel.add(Convolution2D(3, 7, 7, input_shape=input_shape, init='normal'))
        fovModel.add(Activation('relu'))
        fovModel.add(Dropout(0.2))
        fovModel.add(Flatten())
        fovModel.add(Dense(output_dim=1))
        fovModel.add(Activation('softmax'))

    def fit(self, data, answers, nb_epoch=3, batch_size=128):
        self.model.fit(data, answers, nb_epoch=nb_epoch, batch_size=batch_size, show_accuracy=True)

    def predict(self, data):
        return self.model.predict(data)

    def save(self, fname, overwrite=False):
        self.model.save_weights(fname, overwrite=overwrite)

    def load(self, fname):
        self.model.load_weights(fname)


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
        beforeSect = np.zeros(layer.shape+objHalf*2, dtype=np.int8)
        size = 
        # error will happen if size*num < np_matrix.shape
        # currently not handled or needed
        beforeSect[objHalf:objHalf+layer.shape[0], objHalf:objHalf+layer.shape[1]] = layer
        for i, j in [range(numSectors[0]), range(numSectors[1])]:
            layer_sect.append(beforeSect[i:(i+1), j:(j+1)])
        sectors.append(layer_sect)
    return sectors


def main():
    # Fetch Data
    data = np.load('data/peripheryImages.npy')
    answers = np.load('data/peripheryIndexes.npy')
    answers = np_utils.to_categorical(answers, 16)

    periModel = PeripheryNet()
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
    # print(x)


    name = input("If you'd like to save the weights, please enter a savefile name now: ")
    if name:
        periModel.save(name)
    return

if __name__ == '__main__':
    main()