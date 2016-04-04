# Different Search models to test time to find target and number
# of search iterations

import numpy as np
from numpy.random import randint
from random import sample
from helper import splitSectors

# All the models return the number of tries it took them to get the right
# answer, and the number of times they were sure they had the right answer.


def fullRandom(foveImg, periImg, index, foveModel=None):
    assert foveModel is not None, 'We need an initialized FoveaNet with loaded weights'
    tries = 0
    foveSure = 0
    loc = -1
    saccades = []
    foveImSect = splitSectors(foveImg, objHalf=20)
    while loc != index:
        tries += 1
        loc = randint(0, 16)
        saccades.append(loc)
        results = foveModel.predict(np.array([foveImSect[loc]]))
        a = np.argmax(results)
        if a == 0:
            foveSure += 1

    return tries, foveSure, saccades


def randomNoReplacement(foveImg, periImg, index, foveModel=None):
    assert foveModel is not None, 'We need an initialized FoveaNet with loaded weights'
    tries = 0
    foveSure = 0
    foveImSect = splitSectors(foveImg, objHalf=20)
    guesses = sample(range(16), 16)
    for guess in guesses:
        tries += 1
        a = np.argmax(foveModel.predict(np.array([foveImSect[guess]])))
        if a == guess:
            foveSure += 1
        if guess == index:
            break
    return tries, foveSure, guesses[:tries]


def linearSearch(foveImg, periImg, index, foveModel=None):
    assert foveModel is not None, 'We need an initialized FoveaNet with loaded weights'
    tries = 0
    foveSure = 0
    foveImSect = splitSectors(foveImg, objHalf=20)
    for sectInd in range(16):
        tries += 1
        a = np.argmax(foveModel.predict(np.array([foveImSect[sectInd]])))
        if a == sectInd:
            foveSure += 1
        if sectInd == index:
            break
    return tries, foveSure, range(16)[:tries]


def periNetSearch(foveImg, periImg, index, periModel=None):
    assert periModel is not None, 'We need an initialized PeripheryNet with loaded weights'

    periSectors = splitSectors(periImg, objHalf=4)
    predictions = periModel.predict(periSectors)

    attention = np.argsort(predictions[:, 0])[::-1]
    tries = 0
    for att in attention:
        tries += 1
        if att == index:
            break
    return tries, tries, attention[:tries]


def foveNetSearch(foveImg, periImg, index, foveModel=None):
    assert foveModel is not None, 'We need an initialized FoveaNet with loaded weights'

    foveSectors = splitSectors(foveImg, objHalf=20)
    predictions = foveModel.predict(foveSectors)

    attention = np.argsort(predictions[:, 0])[::-1]
    tries = 0
    for att in attention:
        tries += 1
        if att == index:
            break
    return tries, tries, attention[:tries]


def neuralNetSearch(foveImg, periImg, index, periModel=None, foveModel=None):
    assert periModel is not None, 'We need an initialized PeripheryNet with loaded weights'
    assert foveModel is not None, 'We need an initialized FoveaNet with loaded weights'

    periSectors = splitSectors(periImg, objHalf=4)
    foveSectors = splitSectors(foveImg, objHalf=20)

    predictions = periModel.predict(periSectors)
    attention = np.argsort(predictions[:, 0])[::-1]

    foveSure = 0
    tries = 0
    for att in attention:
        tries += 1
        a = np.argmax(foveModel.predict(foveSectors[att][np.newaxis,...]))
        if att == index and a == 0:
            foveSure += 1
            break
        elif a == 0:
            foveSure += 1
    return tries, foveSure, attention[:tries]  # If tries gets to 16 then the model failed to find the right answer.