import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt

def main():
    #Build model

    periModel = Sequential()
    periModel.add(Convolution2D(1,20,20,input_shape=(1,100,100)))
    periModel.add(Activation('tanh'))
    periModel.add(MaxPooling2D(pool_size=(10,10)))
    periModel.add(Flatten())
    periModel.add(Dense(output_dim=100))
    periModel.add(Activation('softmax'))

    periModel.compile(optimizer='sgd',loss='categorical_crossentropy')

    #Fetch Data
    data = np.load('data/peripheryImages.npy')
    answers=np.load('data/peripheryIndexes.npy')
    answers = np_utils.to_categorical(answers,100)

    periModel.fit(data, answers, nb_epoch=20,batch_size=64)
    
    x = periModel.predict(data)
    for i,j in enumerate(x):
        print np.argmax(answers[i])
        plt.plot(j)
        plt.show()
    return

if __name__ == '__main__':
    main()