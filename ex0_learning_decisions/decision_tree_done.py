import pdb
import numpy as np
from scipy.stats import mode, entropy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
np.random.seed(42)

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
    data = {'X': np.loadtxt('chargingData_X.csv',delimiter=','), 'y': np.loadtxt('chargingData_y.csv',delimiter=',').astype(np.int64)[:,None]}

    ############################
    ###### DEFINE COLORS #######
    ############################
    lightBlue = [124/255,173/255,237/255]
    darkBlue = [29/255,61/255,117/255]
    lightRed = np.array([252,174,145])/255
    darkRed = np.array([251,106,74])/255
    lightGreen = np.array([186,228,179])/255
    darkGreen = np.array([116,196,118])/255
    colorRightWrong = np.array([darkBlue,darkBlue])
    colorPred0Pred1 = np.array([lightRed,lightGreen])

    ############################
    ###### DEFINE COLORS #######
    ############################
    marker0 = 'x'
    marker1 = 'o'

    ############################
    ###### SETTINGS ############
    ############################
    MAXDEPTH = 30
    plotData = True
    plotPredictions = True

    ###########################
    ### VISUALIZATION #########
    ###########################
    for maxDepth in range(MAXDEPTH):

        #train model
        root = Node(data['X'],data['y'])
        root.train(maxDepth=maxDepth)
        yhat = root.predict(data['X'])

        #compute accuracy
        accuracy = np.sum(yhat == data['y'])/len(data['y'])
        print('Depth: ' + str(maxDepth) + ' | Accuracy: ' + str(np.round(accuracy*100,2)) + '%')

        #configure axes
        ax = plt.subplot(111)
        ax.set_xlim([np.min(data['X'][:,0]),np.max(data['X'][:,0])])
        ax.set_ylim([np.min(data['X'][:,1]),np.max(data['X'][:,1])])
        ax.tick_params(axis='x', colors=darkBlue)
        ax.tick_params(axis='y', colors=darkBlue)
        ax.spines[['right', 'top','left','bottom']].set_visible(False)

        #draw prediction patches (background)
        queue = [(root,np.min(data['X'][:,0]),np.max(data['X'][:,0]),np.min(data['X'][:,1]),np.max(data['X'][:,1]))]
        while len(queue)>0:
            node = queue[0][0]
            left,right,bottom,top = queue[0][1],queue[0][2],queue[0][3],queue[0][4]
            t = node.featureThreshold

            if node.featureDim is not None:
                if node.featureDim == 0:
                    node.leftChild: queue.append((node.leftChild,left,t,bottom,top))
                    node.rightChild: queue.append((node.rightChild,t,right,bottom,top))
                else:
                    node.leftChild: queue.append((node.leftChild,left,right,bottom,t))
                    node.rightChild: queue.append((node.rightChild,left,right,t,top))
            else:
                if plotPredictions: ax.add_patch(patches.Rectangle((left, bottom), right-left, top-bottom, linewidth=2, linestyle = '-', edgecolor=None, facecolor=colorPred0Pred1[node.mode], alpha = 1.0))
            
            queue = queue[1:]

        #draw prediction markers (foreground)
        if plotData: 
            isWrong = np.abs(data['y']-yhat).astype(np.int64)
            ax.scatter(data['X'][data['y'][:,0]==0][:,0],data['X'][data['y'][:,0]==0][:,1], color=colorRightWrong[isWrong[data['y'][:,0]==0]],s=10,marker=marker0)
            ax.scatter(data['X'][data['y'][:,0]==1][:,0],data['X'][data['y'][:,0]==1][:,1], color=colorRightWrong[isWrong[data['y'][:,0]==1]],s=10,marker=marker1)
        
        #configure axes
        ax.set_xlabel('Longitude')
        ax.xaxis.label.set_color(darkBlue)
        ax.set_ylabel('Latitude')
        ax.yaxis.label.set_color(darkBlue)
        ax.set_title('Depth: ' + str(maxDepth) + ' | Accuracy: ' + str(np.round(accuracy*100,2)) + '%')
        ax.title.set_color(darkBlue)
        
        #save predictions as image
        plt.savefig('dt_depth_' + str(maxDepth) + '.png',dpi=1200)