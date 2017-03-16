import numpy as np
import os
import sklearn.metrics
from pylab import *

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('mnist-original')
mnist.data.shape
mnist.target.shape
np.unique(mnist.target)

print(mnist.data.shape)

X, Y = mnist.data / 255., mnist.target
size = len(Y)
## extract "3" digits and show their average"
ind = [ k for k in range(size) if Y[k]==0 or Y[k]==1]
X=X[ind,:]
Y=Y[ind]

m,n = np.shape(X)
       
#Create Thetha with no. of features
Thetha = np.ones(n)
niters = 100
Alpha = 0.0001
Sigmoid =0
it=0

#Gradient Descent
while(it<niters):
    Gradient = 0
    for i in range(0,m):
        Sigmoid = np.divide(1,1 + np.exp(-np.dot(Thetha.T,X[i])))
        Gradient += np.subtract(Sigmoid,Y[i])*X[i]
    Thetha = Thetha - Alpha*Gradient
    it += 1
#print(Thetha)

#Classification
yhat = []
for i in range(0,len(X)):
    Sigmoid = np.divide(1,(1 + np.exp(-np.dot(Thetha.T,X[i]))))
    if(Sigmoid>0.5):
        yhat.append(1)
    else:
        yhat.append(0)
yhat = np.asarray(yhat)
z = sklearn.metrics.confusion_matrix(Y,yhat.T)
print("Confusion Matrix:")
print(z)

#Measures
p=[]
r=[]
fm=[]
rows = np.sum(z, axis=0)
cols = np.sum(z, axis=1)
for i in range(0,2):
    p.append(np.divide(z[i][i],cols[i]))
    r.append(np.divide(z[i][i],rows[i]))
    fm.append(2*(np.divide(p[i]*r[i],p[i]+r[i])))
a = np.divide(z.diagonal().sum(),z.sum())
print("Precision is: ",p)
print("Recall is:    ",r)
print("F-Measure is: ",fm)
print("Accuracy is:  ",a)
print()