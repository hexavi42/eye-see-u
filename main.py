import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt

def main():
    #Build model

    periModel = Sequential()
    periModel.add(Convolution2D(4,3,3,input_shape=(1,80,80),init='normal'))
    periModel.add(Activation('relu'))
    periModel.add(MaxPooling2D(pool_size=(4,4)))
    periModel.add(Flatten())
    periModel.add(Dense(output_dim=16))
    periModel.add(Activation('softmax'))

    sgd = SGD(lr=1e-5, momentum=0.9, nesterov=True)
    periModel.compile(optimizer=sgd,loss='categorical_crossentropy')

    #Fetch Data
    data    = np.load('data/peripheryImages.npy')
    answers = np.load('data/peripheryIndexes.npy')
    answers = np_utils.to_categorical(answers,16)

    periModel.fit(data, answers, nb_epoch=6,batch_size=128)
    
    x = periModel.predict(data[:10])
    print x
    for i,j in enumerate(x):
        plt.subplot(211)
        plt.imshow(data[i,0,:,:])
        plt.xlabel('We know the stimulus to be at position {}'.format(np.argmax(answers[i])))
        plt.subplot(212)
        plt.plot(j)
        plt.show()
    return

if __name__ == '__main__':
    main()