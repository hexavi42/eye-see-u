# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import random as rand
import os


def makeCircle(r,centre,axis):
    """
    takes radius,r, centre (a container that contains x and y), 
    and axis (a matplotlib figure axis)
    makes a circle with radius r at centre and adds it to the axis
    """
    circ=plt.Circle(centre,r)
    axis.add_artist(circ)
    return

def makeSquare(length,centre,axis):
    centre = [c-0.025 for c in centre] #Make it the true center
    rec=plt.Rectangle(centre,length,length,color="r")
    axis.add_artist(rec)
    return

def randCoord():
    x=rand.random()
    y=rand.random()
    return [x,y]

def recordPos(filepath,imageNum,coords):
    #saving coords of square
    squarePos=open(filepath,"a")
    coordsString="{0},{1}".format(coords[0],coords[1])
    squarePos.write(str(imageNum)+","+coordsString+"\n")
    squarePos.close()

def makeImage(numCircles,numSquares,imageNum,filepath):
    fig=plt.figure(figsize=[8,8])
    axis=fig.gca()
    plt.xticks([])
    plt.yticks([])
    for i in range(numCircles):
        coords=randCoord()
        makeCircle(0.03,coords,axis)
    coords=randCoord()
    makeSquare(0.05,coords,axis)    
    plt.tight_layout(pad=0,h_pad=0,w_pad=0)
    plt.savefig(filepath+"/testImages"+str(imageNum))
    plt.close()
    return coords

if __name__ == "__main__":
    numCircles=20 #number of circles per image
    numSquares=1 #number of squares per image
    numImages=10 #number of images
    
    #removes previous files
    imageFolder="Images"
    for theFile in os.listdir(imageFolder):
        filePath=os.path.join(imageFolder,theFile)
        try:
            if os.path.isfile(filePath):
                os.unlink(filePath)
        except Exception as e:
            print(e)
    
    for i in range(numImages):
        recordPos(i,makeImage(numCircles,numSquares,i))