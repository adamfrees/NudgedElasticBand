import numpy as np
from numpy.linalg import norm

class Image:
    """Insert description here"""
    def __init__(self,index,neighbors,position):
        self.index = index
        self.leftNeighbor = neighbors[0]
        self.rightNeighbor = neighbors[1]
        self.position = position

    def setPosition(self,newPosition):
        self.position = newPosition

def gradient(func,point,stepSize=1.e-3):
    xcomp = (func(point[0]+stepSize,point[1])-func(point[0]-stepSize,point[1]))/(2.*stepSize)
    ycomp = (func(point[0],point[1]+stepSize)-func(point[0],point[1]-stepSize))/(2.*stepSize)
    return np.array([xcomp,ycomp])

def initializeImages(numImages,startPoint,endPoint):
    """Given a starting and ending point, constructs a list of evenly spaced images."""
    images = []
    for counter,x in enumerate(np.linspace(startPoint[0],endPoint[0],numImages+2)):
        y = np.linspace(startPoint[1],endPoint[1],numImages+2)[counter]
        myIndex= counter
        if counter==0:
            leftIndex = None
        else:
            leftIndex = counter-1
        if counter==numImages+1:
            rightIndex = None
        else:
            rightIndex = counter+1
        images += [Image(myIndex,[leftIndex,rightIndex],np.array([x,y]))]
    return images

def getTangent(image,imageList):
    if image.leftNeighbor is None or image.rightNeighbor is None:
        return None
    else:
        t = imageList[image.rightNeighbor].position-imageList[image.leftNeighbor].position
        return t/norm(t,2)

def getParallelSpring(image,tangent,imageList,k):
    if image.leftNeighbor is None or image.rightNeighbor is None:
        return None
    else:
        amp = norm(imageList[image.rightNeighbor].position-image.position)-norm(imageList[image.leftNeighbor].position-image.position)
        return k*amp*tangent

def getPerpGrad(image,tangent,func):
    if tangent is None:
        return None
    grad = gradient(func,image.position)
    return grad - np.dot(grad,tangent)*tangent

def NEB(numImages,startPoint,endPoint,potential,k=1.,timeStep=1.e-2):
    currentImages = initializeImages(numImages,startPoint,endPoint)
    converged=False
    velocity = np.zeros(numImages*2)
    oldForce = np.zeros(numImages*2)
    while not converged:
        currentTangents = map(lambda x: getTangent(x,currentImages),currentImages)
        currentSprings = map(lambda x,y: getParallelSpring(x,y,currentImages,k),currentImages,currentTangents)
        currentGrads = map(lambda x,y: getPerpGrad(x,y,potential),currentImages,currentTangents)
        def subtractNotNone(x,y):
            if x is None or y is None:
                return None
            return x-y
        currentForces = map(subtractNotNone,currentSprings,currentGrads)
        force = np.array([x for x in currentForces if x is not None]).flatten()
        if np.dot(velocity,force)>0:
            velocity = np.dot(velocity,force)*force/(norm(force,2)**2)
        else:
            velocity = np.zeros(numImages*2)
        velocity+=(oldForce+force)/2.*timeStep
        for i,image in enumerate(currentImages):
            if i==0 or i==len(currentImages)-1:
                continue
            currentPosition = image.position
            currentPosition[0]+=velocity[2*(i-1)]*timeStep
            currentPosition[1]+=velocity[2*(i-1)+1]*timeStep
            image.setPosition(currentPosition)
        oldForce=force
        print norm(velocity,2)
        if norm(velocity,2)<1.e-6:
            converged=True
    return currentImages

#print map(lambda x: x.position,initializeImages(10,np.array([0.,0.]),np.array([5.,2.])))
