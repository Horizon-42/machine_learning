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

        #compute scores
        #
        #TODO 1
        #compute score[i,j] (= information gain when splitting at sample i and feature j) for all samples i and features j
        #
        # WHY THIS IS FAST:
        # The straightforward version tries every sample as a threshold and, for each try,
        # rebuilds both child label distributions from scratch (an np.unique per side).
        # That costs O(n^2 * log n) per feature. The code below computes the EXACT same
        # information-gain matrix in O(n * log n) per feature, using three optimizations:
        #
        #   OPTIMIZATION 1 - compute the parent entropy only once.
        #     H(parent) is identical for every candidate split in this node, so there is no
        #     reason to recompute it inside the loop (the old code recomputed it n*d times).
        #
        #   OPTIMIZATION 2 - sort each feature once, then sweep the threshold across it.
        #     Once a feature's values are sorted, sliding the threshold from one value to the
        #     next only ever moves samples from the RIGHT child into the LEFT child; nothing
        #     moves back. So the left-side class counts only grow as we sweep. A single
        #     cumulative sum (np.cumsum) therefore gives the left-side class counts at EVERY
        #     threshold at once - we never rebuild a split from scratch.
        #
        #   OPTIMIZATION 3 - get entropy straight from the class counts (no np.unique).
        #     Once we know how many samples of each class are on each side, entropy is just
        #     -sum(p * log(p)); we evaluate it for all thresholds in one vectorized step.
        n, d = self.X.shape
        y = self.y.ravel()                                  # labels as a flat (n,) array
        classes = np.unique(y)                              # the distinct class labels

        # entropy of many count-rows at once: `counts` has shape (rows, n_classes) -> returns (rows,).
        # Empty sides (row sum == 0) correctly contribute 0 entropy; the mask keeps log(0) out of the result.
        def entropy_rows(counts):
            totals = counts.sum(axis=1, keepdims=True)      # number of samples in each row
            with np.errstate(divide='ignore', invalid='ignore'):
                p = counts / totals                         # per-class probabilities
                terms = np.where(counts > 0, p * np.log(p), 0.0)
            return -terms.sum(axis=1)

        # OPTIMIZATION 1: parent entropy, computed a single time from the full label counts.
        total_counts = (y[:, None] == classes[None, :]).sum(axis=0).astype(float)   # (n_classes,)
        Hparent = entropy_rows(total_counts[None, :])[0]

        score = -1.0 * np.ones((n, d))                      # score[i,j] = info gain of splitting feature j at value X[i,j]
        for j in range(d):
            xj = self.X[:, j]
            order = np.argsort(xj, kind='mergesort')        # OPTIMIZATION 2: sort feature j once, O(n log n)
            xs = xj[order]                                  # feature values, ascending
            ys = y[order]                                   # labels aligned to the sorted values

            # Running (cumulative) class counts of the first p sorted samples, for p = 0..n.
            # prefix[p] = how many samples of each class lie strictly before sorted position p.
            onehot = (ys[:, None] == classes[None, :]).astype(float)                 # (n, n_classes)
            prefix = np.concatenate([np.zeros((1, classes.size)),
                                     np.cumsum(onehot, axis=0)])                     # (n+1, n_classes)
            total = prefix[-1]                              # class counts of the whole node

            # A threshold equal to value v sends every sample with value < v to the left child.
            # When several samples share value v they must all use the SAME left set: the samples
            # before v's FIRST occurrence. group_start[k] = that first-occurrence index for position k.
            is_new_value = np.ones(n, dtype=bool)
            is_new_value[1:] = xs[1:] != xs[:-1]            # True where a new distinct value begins
            group_start = np.maximum.accumulate(np.where(is_new_value, np.arange(n), 0))

            nL = group_start                                # samples in the left child, per threshold
            left_counts = prefix[group_start]               # left-side class counts  (OPTIMIZATION 3 input)
            right_counts = total[None, :] - left_counts     # right-side class counts
            wL = nL / n                                     # child weights = fraction of samples on each side
            wR = 1.0 - wL

            # OPTIMIZATION 3: information gain at every threshold of this feature, all at once.
            gain = Hparent - (wL * entropy_rows(left_counts) + wR * entropy_rows(right_counts))
            score[order, j] = gain                          # scatter the gains back to original sample order

        #find optimal splitting rule
        maxSample, maxFeature = np.unravel_index(np.argmax(score),score.shape)

        #assign optimal splitting rule and continue recursively
        if score[maxSample, maxFeature]>0 and depth<maxDepth:
            
            self.featureDim = maxFeature
            self.featureThreshold = self.X[maxSample,maxFeature]

            #TODO 2
            #XLeft = set to all samples for which the optimal splitting rule is "True"
            #yLeft = set to all labels for which the optimal splitting rule is "True"
            left_mask = self.X[:, self.featureDim] < self.featureThreshold
            XLeft = self.X[left_mask]
            yLeft = self.y[left_mask]
            self.leftChild = Node(XLeft,yLeft)

            #TODO 3
            #train left child node
            self.leftChild.train(maxDepth=maxDepth, depth=depth+1)

            #TODO 4
            #XRight = set to all samples for which the optimal splitting rule is "False"
            #yRight = set to all labels for which the optimal splitting rule is "False"   
            right_mask = self.X[:, self.featureDim] >= self.featureThreshold
            XRight = self.X[right_mask]
            yRight = self.y[right_mask]
            self.rightChild = Node(XRight,yRight)

            #TODO 5
            #train right child node
            self.rightChild.train(maxDepth=maxDepth, depth=depth+1)

    # Make predictions for all samples in a data matrix
    #
    # inputs:
    #   X: (n,d) array: the data matrix
    # outputs:
    #   (n,1) array: the predictions
    def predict(self,X):

        y_hat = -1*np.ones((X.shape[0],1))
        if self.leftChild and self.rightChild:
            #TODO 6
            #fill y_hat with predictions
            t = X[:, self.featureDim] < self.featureThreshold
            y_hat[t] = self.leftChild.predict(X[t])
            y_hat[~t] = self.rightChild.predict(X[~t])
            return y_hat
        else:
            return self.mode*np.ones((X.shape[0],1))


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

        #TODO 7
        print(yhat.shape)
        #accuracy = use yhat and data['y'] to compute the classification accuracy
        accuracy = sum(yhat==data['y'])/yhat.shape[0]
        print('Depth: ' + str(maxDepth) + ' | Accuracy: ' + str(np.round(accuracy*100,2)) + '%')

        #configure axes
        # OPTIMIZATION: start each frame on a FRESH figure. Without this, every iteration keeps
        # drawing on the SAME figure, so the artists (background patches + scatter points) pile up:
        # each frame renders slower than the last, memory keeps growing, and the saved images come
        # out with all depths overlaid on top of each other. plt.close(fig) below frees it again.
        fig = plt.figure()
        ax = fig.add_subplot(111)
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
                    if node.leftChild: queue.append((node.leftChild,left,t,bottom,top))
                    if node.rightChild: queue.append((node.rightChild,t,right,bottom,top))
                else:
                    if node.leftChild: queue.append((node.leftChild,left,right,bottom,t))
                    if node.rightChild: queue.append((node.rightChild,left,right,t,top))
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
        # OPTIMIZATION: dpi=1200 rendered a ~1.5 MB image and took ~2.7s PER FRAME (30 frames = ~80s).
        # 150 dpi is already crisp on screen and saves in ~0.06s, cutting almost all of the plotting time.
        plt.savefig('dt_depth_' + str(maxDepth) + '.png',dpi=150)
        plt.close(fig)   # OPTIMIZATION: release this frame's figure so figures/memory don't accumulate