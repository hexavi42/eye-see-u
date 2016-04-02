import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from nnModels import PeripheryNet

def visualizeKernels(model, wFname, layerNum):
    model.load(wFname)
    layer = model.model.layers[layerNum]
    kernels = layer.get_weights()[0]
    side = (len(kernels)**0.5 + 1) //1
    for i,k in enumerate(kernels):
        plt.subplot(side,side,1+i)
        plt.imshow(k[0], cmap=cm.binary, interpolation='nearest')
    plt.show()
    return
