import numpy as np
import matplotlib.pyplot as plt
import time
import pdb

# inputs:
#   degree: scalar; polynomial kernel degree
#   bias: boolean; True if kernel should include bias
# outputs:
#   function; polynomial kernel
#       inputs:
#           z1: (n,d)-array; n data points of dimension d
#           z2: (m,d)-array; m data points of dimension d
#       outputs:
#           (nxm)-array; kernel matrix
def makePolynomialKernel(degree,bias):
    #TODO 1: implement generator for polynomial kernel

# inputs:
#   sigma: scalar; Gaussian kernel standard deviation 
# outputs:
#   function; Gaussian kernel
#       inputs:
#           z1: (n,d)-array; n data points of dimension d
#           z2: (m,d)-array; m data points of dimension d
#       outputs:
#           (nxm)-array; kernel matrix
def makeGaussianKernel(sigma):
    #TODO 2: implement generator for Gaussian kernel

# inputs:
#   f: function; computes y_i for all x_i
#     inputs:
#       x: 1D sequence; x_i for all i=1...n
#     ouputs:
#       1D sequence; y_i for all i=1...n
#   n: scalar; number of data points
#   noise: scalar; standard deviation of Gaussian noise added to data points
#   lim: [a,b]; [lower,upper] limit of data points
# outputs:
#   dictionary with keys 'x' -> (n,1)-array ('features') and 'y' -> (nx1)-array ('labels')
def generateData(f,n,noise,lim):
    x = np.linspace(lim[0],lim[1],n)
    y = f(x) + np.random.normal(0,noise,n)
    return {'x':x[:,None],'y':y[:,None]}

# inputs:
#   train: dictionary; training data generated with 'generateData'
#   kernel: function; kernel function generated with 'makeGaussianKernel' or 'makePolynomialKernel'
#   gamma: scalar; learning rate
#   iter: scalar; training iterations
#   test: dictionary; test data generated with 'generateData' 
# ouputs:
#   scalar; runtime of one training iteration
def kernelRegression(train,kernel,gamma,iter,test=None):
    #TODO 3: initialize alpha (column vector)

    t1 = time.time()
    #TODO 4: compute kernel matrix K for data in train['x'] (square matrix)
    t2 = time.time()

    if test: fig = plt.figure()
    for i in range(iter):

        t3 = time.time()
        #TODO 5: update alpha (column vector)
        t4 = time.time()
        
        if test:
            fig.clear()
            plt.scatter(test['x'],test['y'],edgecolor='none', alpha=0.5)
            #TODO 6: predict yhat for data in test['x'] (column vector)
            plt.plot(test['x'],yhat,'r',linewidth=4,alpha=0.75)
            fig.canvas.draw()
            fig.canvas.flush_events()
    if test: plt.close(fig)

    return (t4-t3)+(t2-t1)

#setup
plt.ion()
np.random.seed(0)

#parameters
f = lambda x: x**3+4*x**2+x+4 #function from which to sample data points
n_train = 201 #number of training data points
n_test = 200 #number of test data points
noise = 2 #noise added to data points
sigma = 1.0 #standard deviation of Gaussian kernel
bias = True #whether to use a bias term in the polynomial kernel
degree = 0 #degree of polynomial kernel
gamma = 1e-4 #learning rate
iter = 1000 #number of iterations

kernel = makePolynomialKernel(degree=degree,bias=bias)
#kernel = makeGaussianKernel(sigma=sigma) 

#experiment 1
if True:    
    data_train = generateData(f,n_train,noise,[-4,2])
    data_test = generateData(f,n_test,noise,[-4,2])    
    kernelRegression(data_train,kernel,gamma,iter,data_test)
input()