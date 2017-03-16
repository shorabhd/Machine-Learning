from scipy.spatial import distance
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import time;
import scipy

#Define Kernel Function
def kernelFunction(xi,x,Sigma):
    return scipy.exp(-(np.power(np.linalg.norm(xi-x),2))/(2*(np.power(Sigma,2))))

#Read File and combine all the features into X and Y.
X = []
Y = []
for line in open("E:\MS\ML\Ass 1\Datasets\mvar-set1.dat"):
    li=line.strip()
    if not li.startswith("#"):
        var = line.split()
        Y.append([float(var[-1])])
        X.append([float(i) for i in var[0:len(var)-1]])

#Convert the list to an array for future computations
X = np.asarray(X)
Y = np.asarray(Y)

Sigma = 1
P = PolynomialFeatures(1)

#10 Fold Cross Validation
var = [x*(len(X)/10) for x in range(0,11)]
for i in range(len(var)):
    if i!=10:

        #Divide Training and Test Data
        Test = X[var[i]:var[i+1]],Y[var[i]:var[i+1]]
        Train = np.concatenate((X[var[i+1]:],X[:var[i]])),np.concatenate((Y[var[i+1]:],Y[:var[i]]))

        Z = Train[0]

        start = time.time()

        #Calculate Gram Matrix
        EucDistance = distance.cdist(Z,Z,metric='euclidean')
        GramMatrix = scipy.exp(-EucDistance/(2*(Sigma**2)))

        #Calculate Alpha
        Alpha = np.linalg.solve(GramMatrix,Train[1])

        #Calculate y_Hat
        yhat = []
        for i in Test[0]:
            y = 0
            for j in range(len(Train[0])):
                y = y + Alpha[j].item(0) * kernelFunction(np.reshape(i,newshape=(1,len(i))),np.reshape(Train[0][j],newshape=(1,len(Train[0][j]))),Sigma=Sigma)
            yhat.append(y)
        print(yhat)
        yhat = np.reshape(yhat,newshape=(len(yhat),1))

        TestMSE=0
        TestMSE = TestMSE + np.divide(np.sum(np.power(np.subtract(yhat,Test[1]),2)),len(Test[1]))
        print("TestMSE",TestMSE)
        end = time.time()
        print("Time is: ",end-start)


