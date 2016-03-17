import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten
from keras.optimizers import SGD
from pipeline.pipeline import TrainingImage
import matplotlib.pyplot as plt

def main():
    #Build model
    periModel = Sequential()
    
    firstHiddenLayer = Convolution2D(1,5,5,input_shape=(1,100,100))
    periModel.add(firstHiddenLayer)
    periModel.add(Activation('tanh'))
    
    poolingLayer = MaxPooling2D(pool_size=(10,10))
    periModel.add(poolingLayer)

    periModel.add(Flatten())

    outputLayer = Dense(output_dim=100)
    periModel.add(outputLayer)
    periModel.add(Activation('softmax'))

    periModel.compile(optimizer='sgd',loss='mse')

    #Fetch Data
    stimLocs = np.loadtxt('Gen/testValues.txt',delimiter=',')
    data = []
    answers = []

    for i in range(100):
        t = TrainingImage('Gen/Images/testImages{0}.png'.format(i),stimLocs[i][1:])
        data.append(t.peripheryImg)
        answers.append(t.periTrainVec)
    data = np.array(data)
    answers=np.array(answers)

    periModel.fit(data, answers, nb_epoch=50,batch_size=10)
    return

if __name__ == '__main__':
    main()