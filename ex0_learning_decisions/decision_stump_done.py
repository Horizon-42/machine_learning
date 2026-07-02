import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

np.random.seed(42)
class Node:
    
    def __init__(self,X,y):

        self.X, self.y = X, y
        self.mode = mode(self.y)[0][0]

        self.featureDim, self.featureThreshold = None, None
        self.leftChild, self.rightChild = None, None

    def train(self):

        #initialize score matrix
        score = -1*np.ones_like(self.X)
        
        #compute scores
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):        
                leftIdx  = np.where(self.X[:,j] <  self.X[i,j])[0]
                rightIdx = np.where(self.X[:,j] >= self.X[i,j])[0]
                score[i,j] = (np.sum(self.y[leftIdx] != mode(self.y[leftIdx])[0])) + (np.sum(self.y[rightIdx] != mode(self.y[rightIdx])[0]))

        #assign optimal splitting rule
        minSample, minFeature = np.unravel_index(np.argmin(score),score.shape)
        self.featureDim = minFeature
        self.featureThreshold = self.X[minSample,minFeature]

        #initialize child nodes
        leftIdx  = np.where(self.X[:,self.featureDim] <  self.featureThreshold)[0]
        X_left = self.X[leftIdx]
        y_left = self.y[leftIdx]
        self.leftChild = Node(X_left,y_left)

        rightIdx = np.where(self.X[:,self.featureDim] >= self.featureThreshold)[0]
        X_right = self.X[rightIdx]
        y_right = self.y[rightIdx]
        self.rightChild = Node(X_right,y_right)

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
    yhat_base_rule = root.predict(data['X'])
    accuracy_base_rule = np.sum(yhat_base_rule == data['y'])/len(data['y'])
    print('Accuracy (base rule):',str(accuracy_base_rule*100),'%')

    if TRAIN:
        #train decision stump
        root.train()
        print('Training completed. Optimal splitting rule: feature ' + str(root.featureDim) + ' at threshold ' + str(np.round(root.featureThreshold,2)))

        #make prediction with decision stump
        yhat_decision_stump = root.predict(data['X'])
        accuracy_decision_stump = np.sum(yhat_decision_stump == data['y'])/len(data['y'])
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