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

def makeSquare(l,centre,axis):
    rec=plt.Rectangle(centre,l,l,color="r")
    axis.add_artist(rec)
    return

def makeImage(numCircles,numSquares,imageNum):
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
    plt.savefig("Images/testImages"+str(imageNum))
    return coords

def randCoord():
    x=rand.random()
    y=rand.random()
    return [x,y]

def recordPos(imageNum,coords):
    #saving coords of square
    squarePos=open("testValues.txt","a")
    coordsString="{0},{1}".format(coords[0]+0.025,coords[1]+0.025)
    squarePos.write(str(imageNum)+","+coordsString+"\n")
    squarePos.close()
    
if __name__ == "__main__":
    numCircles=20 #number of circles per image
    numSquares=1 #number of squares per image
    numImages=5 #number of images
    
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
    




#fig=plt.figure(figsize=(8,8),dpi=100)
#circ=plt.Circle([0,0],2)
#fig.gca().add_artist(circ)#http://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot
#plt.axis([-10,10,-10,10])#sets axis lengths
#plt.xticks([])#labels axis
#plt.yticks([])
#plt.tight_layout()
#rec=plt.Rectangle([0,2],2,2)
#fig.gca().add_artist(rec)
#plt.show()
##plt.savefig("test1")
#plt.show()