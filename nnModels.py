from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping


class PeripheryNet(object):
    model = None

    def __init__(self, input_shape=[1, 80, 80], sectors=16):
        # Build model
        periModel = Sequential()
        periModel.add(Convolution2D(4, 5, 5, input_shape=input_shape, init='normal'))
        periModel.add(Activation('relu'))
        periModel.add(Dropout(0.1))
        periModel.add(Flatten())
        periModel.add(Dense(output_dim=sectors))
        periModel.add(Activation('softmax'))

        sgd = SGD(lr=1e-9, momentum=0.9, nesterov=True)
        periModel.compile(optimizer=sgd, loss='categorical_crossentropy')
        self.model = periModel

    def fit(self, data, answers, nb_epoch=3, batch_size=128):
        history = self.model.fit(data, answers, nb_epoch=nb_epoch, batch_size=batch_size, show_accuracy=True)
        return history

    def fit_generator(self, generator, samples_per_epoch=90000, nb_epoch=3, show_accuracy=True):
        history = self.model.fit_generator(generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, show_accuracy=show_accuracy)
        return history

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
        fovModel.add(Dropout(0.2))
        fovModel.add(MaxPooling2D(pool_size=(2, 2)))
        fovModel.add(Convolution2D(3, 7, 7, init='normal'))
        fovModel.add(Activation('relu'))
        fovModel.add(Dropout(0.2))
        fovModel.add(Flatten())
        fovModel.add(Dense(output_dim=1))
        fovModel.add(Activation('sigmoid'))
        fovModel.compile(loss='binary_crossentropy', optimizer='rmsprop')
        self.model = fovModel

    def fit(self, data, answers, nb_epoch=3, batch_size=128):
        self.model.fit(data, answers, nb_epoch=nb_epoch, batch_size=batch_size, show_accuracy=True)

    def fit_generator(self, generator, samples_per_epoch=128, nb_epoch=3, verbose=1,show_accuracy=True):
        self.model.fit_generator(generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, show_accuracy=show_accuracy)

    def predict(self, data):
        return self.model.predict(data)

    def save(self, fname, overwrite=False):
        self.model.save_weights(fname, overwrite=overwrite)

    def load(self, fname):
        self.model.load_weights(fname)