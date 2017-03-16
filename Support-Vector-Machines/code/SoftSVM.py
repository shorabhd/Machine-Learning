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

"""Reference Link Mentioned in the Report"""
def SepData(n):
    """
    generates a 2D linearly separable dataset with n samples, where the third element is the label
    """
    xb = (rand(n)*2-1)/2-0.5
    yb = (rand(n)*2-1)/2+0.5
    xr = (rand(n)*2-1)/2+0.5
    yr = (rand(n)*2-1)/2-0.5
    inputs = []
    for i in range(len(xb)):
        inputs.append([xb[i],yb[i],1])
        inputs.append([xr[i],yr[i],-1])
    return inputs

def NonSepData(n):
    """
    generates a 2D linearly separable dataset with n samples, where the third element is the label
    """
    xb = (rand(n)*2-1)/2-0.5
    yb = (rand(n)*2-1)/2+0.5
    xr = (rand(n)*2-1)/2-0.01
    yr = (rand(n)*2-1)/2+0.01
    inputs = []
    for i in range(len(xb)):
        inputs.append([xb[i],yb[i],1])
        inputs.append([xr[i],yr[i],-1])
    return inputs

Z = NonSepData(100)
Z = np.asarray(Z)
np.random.shuffle(Z)

#Split Matrix in X,Y
X = []
Y = []
X,B,Y =np.hsplit(Z,[2,2])

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
Y = Y.astype(np.float)

#10 Fold Cross Validation
var = [x*(len(X)/10) for x in range(0,11)]
for i in range(len(var)):
    if i!=10:

        print("Fold ",i+1)

        #Divide Training and Test Data
        Test = X[var[i]:var[i+1]],Y[var[i]:var[i+1]]
        Train = np.concatenate((X[var[i+1]:],X[:var[i]])),np.concatenate((Y[var[i+1]:],Y[:var[i]]))    
        
        P = np.dot(Train[1],Train[1].T)*np.dot(Train[0],Train[0].T)
        m,n = np.shape(Train[0])
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
        A = Train[1].T
        b=np.zeros(shape=(1,1))
        
        sol = solvers.qp(matrix(P),matrix(q),matrix(G),matrix(h),matrix(A),matrix(b))
        Alpha = np.ravel(sol['x']) 
        
        w=0
        for i in range(0,len(Train[0])):
            w = w + Alpha[i]*Train[0][i]*Train[1][i]
        
        SV_I = Alpha > 0.005
        SV_A = Alpha[SV_I]
        SV_X = X[SV_I]
        SV_Y = Y[SV_I]
        #print(SV_X)
        
        w0=0
        for i in range(0,len(SV_A)):
            w0 = w0 + (SV_Y[i]-np.dot(w.T,SV_X[i]))
        w0 = w0/len(SV_A)
        #print(w0)
        
        yhat=[]
        for i in range(len(Test[0])):
            c = np.dot(w.T,Test[0][i])+w0
            if(c>0):
                yhat.append(1)
            else:
                yhat.append(-1)
        
        #print(yhat)
        yhat = np.asarray(yhat)
        z = sklearn.metrics.confusion_matrix(Test[1],yhat.T)
        print("Confusion Matrix:")
        print(z)
        a = sklearn.metrics.accuracy_score(Test[1],yhat.T)
        print('Accuracy: ',a)
        print('Precision: ',sklearn.metrics.precision_score(Test[1],yhat.T))
        print('Recall: ',sklearn.metrics.recall_score(Test[1],yhat.T))
        print('F-Measure: ',sklearn.metrics.f1_score(Test[1],yhat.T))
        
        
        """
        Change the Kernel for Radial 
		clf = svm.SVC(kernel='rbf',gamma=0.9)
		AND for Polynomial
		clf = svm.SVC(kernel='poly',degree=3)
        """
        clf = svm.SVC(kernel='linear')
        clf.fit(X,Y.ravel())
        yhat = clf.predict(Test[0])  
        z = sklearn.metrics.confusion_matrix(Test[1].ravel(),yhat)
        print("Confusion Matrix:")
        print(z)
        a = sklearn.metrics.accuracy_score(Test[1].ravel(),yhat)
        print('Accuracy: ',a)
        print('Precision: ',sklearn.metrics.precision_score(Test[1],yhat.T))
        print('Recall: ',sklearn.metrics.recall_score(Test[1],yhat.T))
        print('F-Measure: ',sklearn.metrics.f1_score(Test[1],yhat.T))
        
        
        #Plot Graph
        mt.subplot(211)
        mt.title("Original Dataset")
        mt.scatter(X[:,0],X[:,1], c=Y)
        mt.subplot(212)
        mt.title("Original Dataset with SV")
        mt.scatter(X[:,0],X[:,1], c=Y)
        mt.scatter(SV_X[:,0],SV_X[:,1],c=SV_Y,color='red')
        mt.show()
    
