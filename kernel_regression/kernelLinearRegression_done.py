import numpy as np
import matplotlib.pyplot as plt
import time


def makePolynomialKernel(degree,bias): return lambda z1,z2: ((1 if bias else 0) + np.sum(z1[None,:,:]*z2[:,None,:],2)).T**degree
def makeGaussianKernel(sigma): return lambda z1,z2: np.exp(-np.sum((z1[:,None,:]-z2[None,:,:])**2,2)/(2*sigma**2))

def generateData(f,n,noise,lim):
    x = np.linspace(lim[0],lim[1],n)
    y = f(x) + np.random.normal(0,noise,n)
    return {'x':x[:,None],'y':y[:,None]}

def kernelRegression(train,kernel,gamma,iter,test=None):
    alpha = np.zeros((train['x'].shape[0],1))
    
    t1 = time.time()
    K = kernel(train['x'],train['x'])
    t2 = time.time()

    if test: fig = plt.figure()
    for i in range(iter):

        t3 = time.time()
        alpha += gamma*(train['y']-np.dot(K,alpha))
        t4 = time.time()
        
        if test:
            fig.clear()
            plt.scatter(test['x'],test['y'],edgecolor='none', alpha=0.5)
            if i==0: input()
            yhat = np.dot(kernel(test['x'],train['x']),alpha)
            plt.plot(test['x'],yhat,'r',linewidth=4,alpha=0.75)
            fig.canvas.draw()
            fig.canvas.flush_events()

            print('Training error: ' + str(np.round(np.linalg.norm(np.dot(kernel(train['x'],train['x']),alpha)-train['y']),2)))
            print('Test error: ' + str(np.round(np.linalg.norm(np.dot(kernel(test['x'],train['x']),alpha)-test['y']),2)))

    if test: plt.close(fig)

    return (t4-t3)+(t2-t1)

#setup
plt.ion()
np.random.seed(1)

#parameters
f = lambda x: x**3+4*x**2+x+4
n_train = 201
n_test = 200
noise = 2
sigma = 0.01
bias = True
degree = 3
gamma = 1e-2 #deg0: 1e-4; deg1: 1e-4; deg2: 1e-4; deg3: 1e-5; deg4: 1e-6; sigma0.01: 1e-2; sigma0.1: 1e-3; sigma1.0: 1e-3
iter = 10000

#kernel = makePolynomialKernel(degree=degree,bias=bias)
kernel = makeGaussianKernel(sigma=sigma) 

#experiment
input()
if True:    
    data_train = generateData(f,n_train,noise,[-4,2])
    data_test = generateData(f,n_test,noise,[-4,2])
    kernelRegression(data_train,kernel,gamma,iter,data_test)
input()