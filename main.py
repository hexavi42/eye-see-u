import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt

def main():
    #Build model

    periModel = Sequential()
    periModel.add(Convolution2D(2,10,10,input_shape=(1,80,80)))
    periModel.add(Activation('relu'))
    # periModel.add(MaxPooling2D(pool_size=(2,2)))
    periModel.add(Convolution2D(2,10,10))
    periModel.add(Activation('relu'))
    periModel.add(Flatten())
    periModel.add(Dense(output_dim=16))
    periModel.add(Activation('softmax'))

    sgd = SGD(lr=1e-7)
    periModel.compile(optimizer=sgd,loss='categorical_crossentropy')

    #Fetch Data
    data    = np.load('data/peripheryImages.npy')
    answers = np.load('data/peripheryIndexes.npy')
    answers = np_utils.to_categorical(answers,16)

    periModel.fit(data, answers, nb_epoch=48,batch_size=6)
    
    x = periModel.predict(data[:10])
    print x
    for i,j in enumerate(x):
        plt.subplot(211)
        plt.imshow(data[i,0,:,:],cmap='gray')
        plt.xlabel('We know the stimulus to be at position {}'.format(np.argmax(answers[i])))
        plt.subplot(212)
        plt.plot(j)
        plt.show()
    return

if __name__ == '__main__':
    main()