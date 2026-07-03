import pdb
import numpy as np
from scipy.stats import mode, entropy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
np.random.seed(11)

class Node:
    
    # Initialize node with a data matrix and a label vector 
    #
    # inputs:
    #   X: (n,d) array: the data matrix
    #   y: (n,1) array: the label vector
    # outputs:
    #   None
    def __init__(self,X,y):

        self.X, self.y = X, y
        self.mode = mode(self.y)[0][0]

        self.featureDim, self.featureThreshold = None, None
        self.leftChild, self.rightChild = None, None

    # Compute entropy of a (n,1) array
    #
    # inputs:
    #   x: (n,1) array: the array for which to compute entropy
    # outputs:
    #   scalar: the entropy of x
    def H(self,x): return entropy(np.unique(x, return_counts=True)[1])

    # Train node by setting the splitting dimension (self.featureDim), the splitting threshold (self.featureThreshold), the left child node (self.leftChild) and the right child node (self.rightChild)
    #
    # inputs:
    #   maxDepth: scalar: the maximum tree depth
    #   depth: scalar: the current tree depth
    # outputs:
    #   None
    def train(self,maxDepth,depth=0):

        #initialize score matrix
        score = -1*np.ones_like(self.X)

        #compute scores
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):        
                leftIdx  = np.where(self.X[:,j] <  self.X[i,j])[0]
                rightIdx = np.where(self.X[:,j] >= self.X[i,j])[0]
                score[i,j] = self.H(self.y) - len(leftIdx)/self.y.shape[0]*self.H(self.y[leftIdx]) - len(rightIdx)/self.y.shape[0]*self.H(self.y[rightIdx])

        #find optimal splitting rule
        maxSample, maxFeature = np.unravel_index(np.argmax(score),score.shape)

        #assign optimal splitting rule and continue recursively
        if score[maxSample, maxFeature]>0 and depth<maxDepth:
            
            self.featureDim = maxFeature
            self.featureThreshold = self.X[maxSample,maxFeature]

            #train left child node
            leftIdx  = np.where(self.X[:,self.featureDim] <  self.featureThreshold)[0]
            XLeft = self.X[leftIdx]
            yLeft = self.y[leftIdx]
            self.leftChild = Node(XLeft,yLeft)
            self.leftChild.train(maxDepth=maxDepth,depth=depth+1)

            #train right child node
            rightIdx = np.where(self.X[:,self.featureDim] >= self.featureThreshold)[0]
            XRight = self.X[rightIdx]
            yRight = self.y[rightIdx]
            self.rightChild = Node(XRight,yRight)
            self.rightChild.train(maxDepth=maxDepth,depth=depth+1)

    # Make predictions for all samples in a data matrix
    #
    # inputs:
    #   X: (n,d) array: the data matrix
    # outputs:
    #   (n,1) array: the predictions
    def predict(self,X):

        y_hat = -1*np.ones((X.shape[0],1))
        if self.leftChild and self.rightChild:
            t = X[:,self.featureDim] < self.featureThreshold
            y_hat[t] = self.leftChild.predict(X[t])
            y_hat[~t] = self.rightChild.predict(X[~t])
            return y_hat
        else:
            return self.mode


if __name__ == '__main__':

    ############################
    ###### LOAD DATA ###########
    ############################        
    X = {'train': np.loadtxt('chargingData_X.csv',delimiter=',')[:400], 'test': np.loadtxt('chargingData_X.csv',delimiter=',')[400:]}
    y = {'train': np.loadtxt('chargingData_y.csv',delimiter=',')[:400].astype(np.int64)[:,None], 'test': np.loadtxt('chargingData_y.csv',delimiter=',')[400:].astype(np.int64)[:,None]}
    
    ############################
    ###### SETTINGS ############
    ############################
    MAXDEPTH = 15
    VALIDATION_FRACTION = 0.3

    ############################
    # CONSTRUCT VALIDATION SET #
    ############################
    n = X['train'].shape[0]
    p = np.random.permutation(n)

    nVal = np.floor(VALIDATION_FRACTION*n).astype(int)
    X['val'] = X['train'][p[:nVal]]
    y['val'] = y['train'][p[:nVal]]
    X['train'] = X['train'][p[nVal:]]
    y['train'] = y['train'][p[nVal:]]
    
    ############################
    ###### RUN VALIDATION ######
    ############################
    accuracy = {'train': np.zeros(MAXDEPTH), 'val': np.zeros(MAXDEPTH)}
    for maxDepth in range(MAXDEPTH):

        #train model
        root = Node(X['train'],y['train'])
        root.train(maxDepth=maxDepth)

        #compute accuracy
        accuracy['train'][maxDepth] = np.sum(root.predict(X['train']) == y['train'])/len(y['train'])
        accuracy['val'][maxDepth] = np.sum(root.predict(X['val']) == y['val'])/len(y['val'])
        print('Depth: ' + str(maxDepth) + ' | Accuracy: ' + str(np.round(accuracy['train'][maxDepth]*100,2)) + '% (train), ' + str(np.round(accuracy['val'][maxDepth]*100,2)) + '% (val)')

    optimalDepth = np.argmax(accuracy['val'])
    print('Optimal depth: ' + str(optimalDepth))