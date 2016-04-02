from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import numpy as np
from makeData import makeFoveImages, makePeriImages
from main import splitSectors
from random import choice


class PeripheryNet(object):
    model = None

    def __init__(self, input_shape=[1, 28, 28], sectors=2):
        # Build model
        periModel = Sequential()
        
    	periModel.add(Convolution2D(4, 7, 7, input_shape=input_shape, init='normal'))
        periModel.add(Activation('relu'))
    	periModel.add(Convolution2D(4, 5, 5))
    	periModel.add(Activation('relu'))
        
    	periModel.add(Flatten())

        periModel.add(Dense(output_dim=sectors))
        periModel.add(Activation('softmax'))

        sgd = SGD(lr=1e-6, momentum=0.9, nesterov=True)
        periModel.compile(optimizer=sgd, loss='categorical_crossentropy')
        self.model = periModel

    def fit(self, data, answers, nb_epoch=3, batch_size=128):
        history = self.model.fit(data, answers, nb_epoch=nb_epoch, batch_size=batch_size, show_accuracy=True)
        return history

    def fit_generator(self, generator=None, batch_size=128, samples_per_epoch=90000, nb_epoch=3, show_accuracy=True):
        if generator==None:
            generator = self.dataGen(numDistractors=20, batch_size=batch_size)
        history = self.model.fit_generator(generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, show_accuracy=show_accuracy)
        return history

    def dataGen(self, numDistractors=20, batch_size=128, border=40):
        while True:
            triLoc   = np.random.randint(border, 400-border, (batch_size ,numDistractors, 2))
            squaLoc  = np.random.randint(border, 400-border, (batch_size, 2))

            imgs = np.zeros( (batch_size, 1, 28, 28), dtype=np.uint8)
            ans = np.zeros((batch_size, 2), dtype=np.uint8)
            for i in range(batch_size):
                img, loc = makePeriImages(triLoc[i], squaLoc[i])
                sectors = splitSectors(img, objHalf=4)
                if np.random.random() > 0.5:
                    imgs[i] = sectors[loc]
                    ans[i][0]  = 1
                else:
                    newLoc = choice(range(loc)+range(loc+1, 16))
                    imgs[i]= sectors[newLoc]
                    ans[i][1]    = 1
            yield (imgs, ans)

    def predict(self, data):
        return self.model.predict(data)

    def save(self, fname, overwrite=False):
        self.model.save_weights(fname, overwrite=overwrite)

    def load(self, fname):
        self.model.load_weights(fname)


class FoveaNet(object):
    model = None

    def __init__(self, input_shape=[3, 240, 240]):
        # Build model
        fovModel = Sequential()

        fovModel.add(Convolution2D(3, 15, 15, input_shape=input_shape, init='normal'))
        fovModel.add(Activation('relu'))

        fovModel.add(MaxPooling2D(pool_size=(2, 2)))
        fovModel.add(Convolution2D(3, 7, 7, init='normal'))
        fovModel.add(Activation('relu'))

        fovModel.add(Flatten())

        fovModel.add(Dense(output_dim=2))
        fovModel.add(Activation('softmax'))

        sgd = SGD(lr=1e-6, momentum=0.9, nesterov=True)
        fovModel.compile(optimizer=sgd, loss='categorical_crossentropy')
        self.model = fovModel

    def fit(self, data, answers, nb_epoch=3, batch_size=128):
        history = self.model.fit(data, answers, nb_epoch=nb_epoch, batch_size=batch_size, show_accuracy=True)
        return history

    def fit_generator(self, generator=None, samples_per_epoch=128, nb_epoch=3, show_accuracy=True):
        if generator==None:
            generator = self.dataGen(numDistractors=20, batch_size=128)
        history = self.model.fit_generator(generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, show_accuracy=show_accuracy)
        return history

    def dataGen(self, numDistractors=20, batch_size=128, border=40):
        while True:
            triLoc = np.random.randint(border, 400-border, (batch_size, numDistractors, 2))
            squaLoc = np.random.randint(40, 400-border, (batch_size, 2))

            imgs = np.zeros((batch_size, 3, 140, 140), dtype=np.uint8)
            ans = np.zeros((batch_size, 2), dtype=np.uint8)
            for i in range(batch_size):
                img, loc = makeFoveImages(triLoc[i], squaLoc[i])
                sectors = splitSectors(img)
                if np.random.random() > 0.5:
                    imgs[i] = sectors[loc]
                    ans[i][0] = 1
                else:
                    newLoc = choice(range(loc)+range(loc+1, 16))            
                    imgs[i] = sectors[newLoc]
                    ans[i][1] = 1
            yield (imgs, ans)

    def predict(self, data):
        return self.model.predict(data)

    def save(self, fname, overwrite=False):
        self.model.save_weights(fname, overwrite=overwrite)

    def load(self, fname):
        self.model.load_weights(fname)
