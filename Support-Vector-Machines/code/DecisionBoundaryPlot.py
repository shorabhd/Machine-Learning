import matplotlib.pyplot as plt
from pylab import rand
import numpy as np
from cvxopt import solvers
from cvxopt.base import matrix
import sklearn.metrics
import matplotlib.pyplot as mt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel

"""Reference Link Mentioned in the Report"""
"""Generates a 2D linearly separable dataset with n samples, where the third element is the label"""
def generateData(n):
    xb = (rand(n)*2-1)/2-0.1
    yb = (rand(n)*2-1)/2+0.2
    xr = (rand(n)*2-1)/2+0.2
    yr = (rand(n)*2-1)/2-0.1
    inputs = []
    for i in range(len(xb)):
        inputs.append([xb[i],yb[i],1])
        inputs.append([xr[i],yr[i],-1])
    return inputs

Z = generateData(100)
Z = np.asarray(Z)
np.random.shuffle(Z)

#Split Matrix in X,Y
X = []
Y = []
X,B,Y =np.hsplit(Z,[2,2])
print(len(Z))

"""
For Polynomial Kernel
T = PolynomialFeatures(3)
X = T.fit_transform(X)
"""
"""
For Radial Kernel
X = rbf_kernel(X,X)
"""
X = X.astype(np.float)

P = np.dot(Y,Y.T)*np.dot(X,X.T)
m,n = np.shape(X)
q = np.zeros(shape=(m,1))
for i in range(len(q)): 
    q[i] = -1
G = np.zeros(shape=(m,m))
np.fill_diagonal(G,-1)
G2 = np.zeros(shape=(m,m))
np.fill_diagonal(G2,1)
G = np.concatenate((G,G2))
h = np.zeros(shape=(2*m,1))
for i in range(len(h)/2,len(h)): 
    h[i] = 1
A = Y.T
b=np.zeros(shape=(1,1))

sol = solvers.qp(matrix(P),matrix(q),matrix(G),matrix(h),matrix(A),matrix(b))
Alpha = np.ravel(sol['x']) 

"""Decision Boundary Plot Code"""
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

X2 = np.c_[xx.ravel(), yy.ravel()]

w=0
for i in range(0,len(X)):
    w = w + Alpha[i]*X[i]*Y[i]

SV_I = Alpha > 0.01
SV_A = Alpha[SV_I]
SV_X = X[SV_I]
SV_Y = Y[SV_I]

w0=0
for i in range(0,len(SV_A)):
    w0 = w0 + (SV_Y[i]-np.dot(w.T,SV_X[i]))
w0 = w0/len(SV_A)
#print(w0)
yhat=[]
for i in range(len(X2)):
    c = np.dot(w.T,X2[i])+w0
    if(c>0):
        yhat.append(1)
    else:
        yhat.append(-1)

yhat = np.asarray(yhat)
yhat = yhat.reshape(xx.shape)
plt.contourf(xx, yy, yhat, cmap=plt.cm.Paired)
plt.axis('off')
plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap=plt.cm.Paired)
plt.show()

"""
Change the Kernel for Radial 
clf = svm.SVC(kernel='rbf',gamma=0.9)
AND for Polynomial
clf = svm.SVC(kernel='poly',degree=3)
"""
clf = svm.SVC(kernel='linear')
clf.fit(X,Y.ravel())
yhat = clf.predict(X2)
yhat = yhat.reshape(xx.shape)
plt.contourf(xx, yy, yhat, cmap=plt.cm.Paired)
plt.axis('off')
plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap=plt.cm.Paired)
plt.show()
