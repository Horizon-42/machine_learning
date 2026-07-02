import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

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

    # Train node by setting the splitting dimension (self.featureDim), the splitting threshold (self.featureThreshold), the left child node (self.leftChild) and the right child node (self.rightChild)
    #
    # inputs:
    #   None
    # outputs:
    #   None
    def train(self):

        #initialize score matrix
        score = -1*np.ones_like(self.X)
        
        #compute scores
        
        #TODO 
        #compute score[i,j] (= classification accuracy when splitting at sample i and feature j) for all samples i and features j    
        row_n, col_n = self.X.shape
        for i in range(row_n):
            for j in range(col_n):
                sufficiant_mask = self.X[:, j] < self.X[i, j]
                # find the most common label
                y_hat = np.ones_like(self.y)
                y_hat[sufficiant_mask] = mode(self.y[sufficiant_mask])[0]
                y_hat[~sufficiant_mask] = mode(self.y[~sufficiant_mask])[0]
                score[i][j] = sum(y_hat!=self.y)
        #assign optimal splitting rule
        minSample, minFeature = np.unravel_index(np.argmin(score),score.shape)
        self.featureDim = minFeature
        self.featureThreshold = self.X[minSample,minFeature]

        #initialize child nodes
        
        #TODO
        #X_left = set to all samples for which the optimal splitting rule is "True"
        #y_left = set to all labels for which the optimal splitting rule is "True"
        left_mask = self.X[:, self.featureDim]<self.featureThreshold
        X_left = self.X[left_mask]
        y_left = self.y[left_mask]
        self.leftChild = Node(X_left,y_left)

        #TODO
        #X_right = set to all samples for which the optimal splitting rule is "False"
        #y_right = set to all labels for which the optimal splitting rule is "False"  
        right_mask = self.X[:, self.featureDim]>=self.featureThreshold
        X_right = self.X[right_mask]
        y_right = self.y[right_mask]  
        self.rightChild = Node(X_right,y_right)

    # Make predictions for all samples in a data matrix
    #
    # inputs:
    #   X: (n,d) array: the data matrix
    # outputs:
    #   y_hat: (n,1) array: the predictions
    def predict(self,X):

        if self.leftChild and self.rightChild:
            y_hat = -1*np.ones((X.shape[0],1))
            t = X[:,self.featureDim] < self.featureThreshold
            y_hat[t] = self.leftChild.mode
            y_hat[~t] = self.rightChild.mode
            return y_hat
        else:
            y_hat = self.mode*np.ones((X.shape[0],1))
            return y_hat



if __name__ == '__main__':

    #####################
    ###### CONFIG #######
    #####################

    TRAIN = True
    
    #####################
    ##### MODELLING #####
    #####################

    #load data
    data = {'X': np.loadtxt('sleepData_X.csv',delimiter=','), 'y': np.loadtxt('sleepData_y.csv',delimiter=',').astype(np.int64)[:,None]}
    root = Node(data['X'],data['y'])

    #make prediction with base rule
    yhat_base_rule = root.predict(data['X']) #(n,1) column vector with binary model predictions using the base rule
    #TODO: use yhat_base_rule and data['y'] to compute the model accuracy before training
    n = data['X'].shape[0]
    accuracy_base_rule = sum(yhat_base_rule==data['y'])/n
    if TRAIN:
        #train decision stump
        root.train()
        print('Training completed. Optimal splitting rule: feature ' + str(root.featureDim) + ' at threshold ' + str(np.round(root.featureThreshold,2)))

        #make prediction with decision stump
        yhat_decision_stump = root.predict(data['X']) #(n,1) column vector with binary model predictions using the decision stump
        #TODO: use yhat_decision_stump and data['y'] to compute the model accuracy after training
        accuracy_decision_stump = sum(yhat_decision_stump==data['y'])/n
        print('Accuracy (decision stump):',str(accuracy_decision_stump*100),'%')

    #####################
    ### VISUALIZATION ###
    #####################

    #define colors
    darkBlue = [29/255,61/255,117/255]
    lightBlue = [124/255,173/255,237/255]
    red = [226/255,33/255,70/255]
    green = [69/255,181/255,60/255]
    colorRightWrong = np.array([green,red])
    colorNormalElevated = np.array([darkBlue,lightBlue])

    #initialize figure
    ax = plt.subplot(111)
    ax.set_xlim([np.min(data['X'][:,0]),np.max(data['X'][:,0])])
    ax.set_ylim([np.min(data['X'][:,1]),np.max(data['X'][:,1])])
    ax.tick_params(axis='x', colors=darkBlue)
    ax.tick_params(axis='y', colors=darkBlue)
    ax.spines[['right', 'top','left','bottom']].set_visible(False)

    #plot data
    queue = [(root,np.min(data['X'][:,0]),np.max(data['X'][:,0]),np.min(data['X'][:,1]),np.max(data['X'][:,1]))]
    while len(queue)>0:
        node,left,right,bottom,top = queue[0][0],queue[0][1],queue[0][2],queue[0][3],queue[0][4]
        t = node.featureThreshold

        if node.featureDim is not None:
            if node.featureDim == 0:
                queue.append((node.leftChild,left,t,bottom,top))
                queue.append((node.rightChild,t,right,bottom,top))
            else:
                queue.append((node.leftChild,left,right,bottom,t))
                queue.append((node.rightChild,left,right,t,top))
        else:
            ax.add_patch(Rectangle((left, bottom), right-left, top-bottom, edgecolor='None', facecolor=colorNormalElevated[node.mode], alpha = 1.0))
        
        queue = queue[1:]

    ax.scatter(data['X'][data['y'][:,0]==0,0],data['X'][data['y'][:,0]==0,1], color=colorRightWrong[np.abs(data['y']-(yhat_decision_stump if TRAIN else yhat_base_rule)).astype(np.int64)][data['y'][:,0]==0],marker='o',s=15)
    ax.scatter(data['X'][data['y'][:,0]==1,0],data['X'][data['y'][:,0]==1,1], color=colorRightWrong[np.abs(data['y']-(yhat_decision_stump if TRAIN else yhat_base_rule)).astype(np.int64)][data['y'][:,0]==1],marker='x',s=15)

    #plot legend
    legend = [Line2D([0], [0], color=colorNormalElevated[0], lw=6, label='normal'),
            Line2D([0], [0], color=colorNormalElevated[1], lw=6, label='elevated'),
            Line2D([0], [0], marker='o', linewidth=0, label='correct', markeredgewidth=0, markerfacecolor=colorRightWrong[0], markersize=10),
            Line2D([0], [0], marker='o', linewidth=0, label='wrong', markeredgewidth=0, markerfacecolor=colorRightWrong[1], markersize=10)]
    ax.legend(handles=legend, loc='upper right')

    #show figure
    plt.show()