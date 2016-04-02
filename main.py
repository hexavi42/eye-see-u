import numpy as np
from helper import processForValidation
import nnModels
import argparse

# fix for python 2->3
try:
    input = raw_input
except NameError:
    pass

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
