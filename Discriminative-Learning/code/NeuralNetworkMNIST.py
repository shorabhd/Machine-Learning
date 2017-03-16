import numpy as np
import os
from pylab import *
import sklearn.metrics
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('mnist-original')
mnist.data.shape
mnist.target.shape
np.unique(mnist.target)

X, Y = mnist.data / 255., mnist.target
size = len(Y)

ind = [ k for k in range(size) if Y[k]==0 or Y[k]==1 or Y[k]==2]
X=X[ind,:]
Y=Y[ind]

m,n = np.shape(X)
        
Thetha = []
#Iterate for Each Unit
for k in range(0,5):
    
    #Create w with no. of Classes x no. of features
    #Create v for all Classes
    w =[]
    v=[]
    for k in range(0,3):
        w.append(np.random.rand(n))
        v.append(np.random.rand())
    v = np.asarray(v)
    
    niters = 1
    Alpha = 0.0001

    #Gradient Descent
    while(niters>0):
        
        GradientW = 0
        Z=[]
        for i in range(0,3):
            Z.append(np.divide(1,1 + np.exp(-np.dot(w[i],X.T))))
            Yhat = np.dot(Z[i],v[i])
            
            #Indicator
            ind =[]
            for j in range(0,m):
                if(Y[j]==i):
                    ind.append(1)
                else:
                    ind.append(0)
        
            GradientV = 0
            for j in range(0,m):
                GradientV += (Yhat[j] - ind[j])*Z[i][j]
                GradientW += (Yhat[j] - ind[j])*v[i]
                
            v[i] = v[i] - Alpha*GradientV
            #print(v[i])
            Thetha.append(v[i])
        
        for i in range(0,3):
            for j in range(0,m):
                GradientW += Z[i][j]*(1-Z[i][j])*X[j] 
            w[i] = w[i] - Alpha*GradientW
            #print(w[i])
        niters -= 1
    
    print(Thetha)
    memFunc =[]
    for i in range(3):
        memFunc.append(Thetha[i]*(np.divide(1,1 + np.exp(-np.dot(w[i],X.T)))).T)
    memFunc = np.asarray(memFunc)
    print(memFunc.shape)
    memFunc = np.reshape(memFunc, newshape=(3,len(Y)))
    yhat = []
    for i in memFunc.T:
        if i[:][0] > i[:][1] and i[:][0] > i[:][2]:
            yhat.append(0)
        elif i[:][1] > i[:][0] and i[:][1] > i[:][2]:
            yhat.append(1)
        else:
            yhat.append(2)
    yhat = np.asarray(yhat)
    print(yhat)
    z = sklearn.metrics.confusion_matrix(Y,yhat.T)
    print("Confusion Matrix:")
    print(z)
    
    #Measures
    p=[]
    r=[]
    fm=[]
    rows = np.sum(z, axis=0)
    cols = np.sum(z, axis=1)
    for i in range(0,3):
        p.append(np.divide(z[i][i],cols[i]))
        r.append(np.divide(z[i][i],rows[i]))
        fm.append(2*(np.divide(p[i]*r[i],p[i]+r[i])))
    a = np.divide(z.diagonal().sum(),z.sum())
    print("Precision is: ",p)
    print("Recall is:    ",r)
    print("F-Measure is: ",fm)
    print("Accuracy is:  ",a)
    print()
                    