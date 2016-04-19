import numpy as np
from numpy.linalg import norm

class Image:
    """Image object is a single point along the elastic band, which is free to move."""
    def __init__(self,index,neighbors,position):
        """Image object remembers its index, its neighbors, and its position.

        Inputs:
        index - integer
        neighbors - array, tuple, or list of two integers.
        position - 2D array of floats

        Outputs:
        None
        """
        self.index = index
        self.leftNeighbor = neighbors[0]
        self.rightNeighbor = neighbors[1]
        self.position = position

    def setPosition(self,newPosition):
        """Call to change position of image.

        Inputs:
        newPosition - 2D array of floats

        Outputs:
        None
        """
        self.position = newPosition

def gradient(func,point,stepSize=1.e-3):
    """Calculates the gradient of function func at position point, using only 4 function calls.

    Inputs:
    func - function that takes in two floats and outputs one.
    point - 2D array of floats

    Outputs:
    2D array of floats
    """
    xcomp = (func(point[0]+stepSize,point[1])-func(point[0]-stepSize,point[1]))/(2.*stepSize)
    ycomp = (func(point[0],point[1]+stepSize)-func(point[0],point[1]-stepSize))/(2.*stepSize)
    return np.array([xcomp,ycomp])

def initializeImages(numImages,startPoint,endPoint):
    """Given a starting and ending point, constructs a list of evenly spaced images.

    Input:
    numImages - integer
    startPoint - 2D array of floats
    endPoint - 2D array of floats

    Output:
    list of Images
    """
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
    """Takes in an Image and the list of all images and outputs the tangent
    associated with the image given.

    Inputs:
    image - an Image object.
    imageList - a list of Image objects.

    Outputs:
    2D array of floats. (or None)
    """
    if image.leftNeighbor is None or image.rightNeighbor is None:
        return None
    else:
        t = imageList[image.rightNeighbor].position-imageList[image.leftNeighbor].position
        return t/norm(t,2)

def getParallelSpring(image,tangent,imageList,k):
    """Takes in an Image, that image's tangent and the list of all images (and a
    spring constant k) and outputs the spring force parallel to the tangent.

    Inputs:
    image - an Image object.
    tangent - a 2D array of floats
    imageList - a list of Image objects.
    k - a float

    Outputs:
    2D array of floats. (or None)
    """
    if image.leftNeighbor is None or image.rightNeighbor is None:
        return None
    else:
        amp = norm(imageList[image.rightNeighbor].position-image.position)-norm(imageList[image.leftNeighbor].position-image.position)
        return k*amp*tangent

def getPerpGrad(image,tangent,func):
    """Takes in an Image, that image's tangent and a potential function and
    outputs the component of the gradient of the potential (located at the
    image) that is perpendicular to the tangent.

    Inputs:
    image - an Image object.
    tangent - a 2D array of floats
    func - function that takes in two floats and outputs one.

    Outputs:
    2D array of floats. (or None)
    """
    if tangent is None:
        return None
    grad = gradient(func,image.position)
    return grad - np.dot(grad,tangent)*tangent

def NEB(numImages,startPoint,endPoint,potential,k=1.,timeStep=1.e-2):
    """The main function. This takes in all essential information about the
    problem and outputs a list of images that form a band which go along the MEP.

    Inputs:
    numImages - integer
    startPoint - 2D array of floats
    endPoint - 2D array of floats
    potential - function that takes in two floats and outputs one.
    k - a float
    timeStep - a float

    Outputs:
    A list of Image objects
    """
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
        if norm(velocity,2)<1.e-6:
            converged=True
    return currentImages
