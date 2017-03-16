from pylab import rand,plot,show,norm
import numpy as np
from sklearn.preprocessing import label_binarize
from cvxopt import solvers
from cvxopt.base import matrix
import sklearn.metrics
import matplotlib.pyplot as mt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel

#Read File
filename = "E:\MS\ML\Ass 4\data\iris.data"
Z = []
with open(filename, "r") as filestream:
    for line in filestream:
        current = line.split(",")
        Z.append([i for i in current[0:len(current)]])

#Take 2 Class and Shuffle Matrix for Cross Validation    
Z = np.delete(Z,np.s_[100::],axis=0) 
Z = np.asarray(Z)
np.random.shuffle(Z)

#Split Matrix in X,Y
X = []
Y = []
X,B,Y =np.hsplit(Z,[2,4])

y=[]
for i in Y:
    y.append(i.item(0).strip())
Y = y

Y = label_binarize(Y,['Iris-setosa','Iris-versicolor'],neg_label=-1)
Y = np.asarray(Y)
Y = Y.astype(np.float)
Y = np.reshape(Y, newshape=(len(Y),1))

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
    h[i] = 10000
A = Y.T
b=np.zeros(shape=(1,1))
sol = solvers.qp(matrix(P),matrix(q),matrix(G),matrix(h),matrix(A),matrix(b))
Alpha = np.ravel(sol['x']) 

w=0
for i in range(0,len(X)):
    w = w + Alpha[i]*X[i]*Y[i]

SV_I = Alpha > 0.001
SV_A = Alpha[SV_I]
SV_X = X[SV_I]
SV_Y = Y[SV_I]

w0=0
for i in range(0,len(SV_A)):
    w0 = w0 + (SV_Y[i]-np.dot(w.T,SV_X[i]))
w0 = w0/len(SV_A)

yhat=[]
for i in range(len(X)):
    c = np.dot(w.T,X[i])+w0
    if(c>0):
        yhat.append(1)
    else:
        yhat.append(-1)

yhat = np.asarray(yhat)
z = sklearn.metrics.confusion_matrix(Y,yhat.T)
print("Confusion Matrix:")
print(z)
a = sklearn.metrics.accuracy_score(Y,yhat.T)
print('Accuracy: ',a)
print('Precision: ',sklearn.metrics.precision_score(Y,yhat.T))
print('Recall: ',sklearn.metrics.recall_score(Y,yhat.T))
print('F-Measure: ',sklearn.metrics.f1_score(Y,yhat.T))

"""
Change the Kernel for Radial 
clf = svm.SVC(kernel='rbf',gamma=0.9)
AND for Polynomial
clf = svm.SVC(kernel='poly',degree=3)
"""
clf = svm.SVC(kernel='linear')
clf.fit(X,Y.ravel())
yhat = clf.predict(X)  
z = sklearn.metrics.confusion_matrix(Y.ravel(),yhat)
print("Confusion Matrix:")
print(z)
a = sklearn.metrics.accuracy_score(Y.ravel(),yhat)
print('Accuracy: ',a)
print('Precision: ',sklearn.metrics.precision_score(Y,yhat.T))
print('Recall: ',sklearn.metrics.recall_score(Y,yhat.T))
print('F-Measure: ',sklearn.metrics.f1_score(Y,yhat.T))


#Plot Graph
mt.subplot(211)
mt.title("Original Dataset")
mt.scatter(X[:,0],X[:,1], c=Y)
mt.subplot(212)
mt.title("Original Dataset with SV")
mt.scatter(X[:,0],X[:,1], c=Y)
mt.scatter(SV_X[:,0],SV_X[:,1],c=SV_Y,color='red')
mt.show()
