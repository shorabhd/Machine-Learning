import numpy as np
import os
from pylab import *
import sklearn.metrics

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('mnist-original')
mnist.data.shape
mnist.target.shape
np.unique(mnist.target)

#print(mnist.data.shape)

X, Y = mnist.data / 255., mnist.target
size = len(Y)
## extract "3" digits and show their average"
ind = [ k for k in range(size) if Y[k]==0 or Y[k]==1 or Y[k]==2]
X=X[ind,:]
Y=Y[ind]

m,n = np.shape(X)
       
#Create Thetha with no. of features
Thetha =[]
for i in range(0,3):
    Thetha.append(np.random.rand(n))
Thetha = np.asarray(Thetha)        
niters = 100
Alpha = 0.0001
  
#Gradient Descent   
while(niters>0):
    for j in range(0,3):    
        
        #Indicator
        ind =[]
        for i in range(0,m):
            if(Y[i]==j):
                ind.append(1)
            else:
                ind.append(0)
        
        func=0
        Hypothesis = np.dot(X,Thetha[j].T)
        Gradient = Hypothesis - ind 
        func = np.dot(Gradient,X)
        Thetha[j] = Thetha[j] - Alpha*func
    
    niters -= 1
print(Thetha)

#Classification
memFunc =[]
for i in range(0,3):
    memFunc.append(np.dot(Thetha[i],X.T))
memFunc = np.asarray(memFunc)
#print(memFunc)
memFunc = np.reshape(memFunc, newshape=(3,len(Y)))
#print(memFunc.shape)
yhat = []
for i in memFunc.T:
    print(i[:][0],i[:][1],i[:][2])
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
with np.errstate(invalid='ignore'):
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