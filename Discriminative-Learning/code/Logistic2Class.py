import numpy as np
import sklearn.metrics
from sklearn.preprocessing.data import PolynomialFeatures
 
#Read File and combine all the features into X and Y.
filename = "E:\MS\ML\Ass 2\Datasets\iris.data"
Z = []
with open(filename, "r") as filestream:
    for line in filestream:
        current = line.split(",")
        Z.append([i for i in current[0:len(current)]])

#Shuffle Matrix for Cross Validation
Z = np.delete(Z,np.s_[100::],axis=0)
Z = np.asarray(Z)
np.random.shuffle(Z)

#Split Matrix in X,Y
X = []
Y = []
X,B,Y =np.hsplit(Z,[Z.shape[1]-1,Z.shape[1]-1])

#Non Linear
X = PolynomialFeatures(2).fit_transform(X)

#Get Float Data
X = X.astype(np.float)
classes = np.unique(Y)
for i in range(0,len(Y)):
    for j in range(0,2):
        if (Y[i].item(0) == classes[j]):
            Y[i]=j
Y = Y.astype(np.float)

#10-Fold Cross Validation 
var = [x*(len(X)/10) for x in range(0,11)]
for i in range(len(var)):
    if i!=10:
  
        print("Fold ",i+1)
          
        #Divide Training and Test Data
        Test = X[var[i]:var[i+1]],Y[var[i]:var[i+1]]
        Train = np.concatenate((X[var[i+1]:],X[:var[i]])),np.concatenate((Y[var[i+1]:],Y[:var[i]]))
       
        m,n = np.shape(Train[0])
       
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
                Sigmoid = np.divide(1,1 + np.exp(-np.dot(Thetha.T,Train[0][i])))
                Gradient += np.subtract(Sigmoid,Train[1][i])*Train[0][i]
            Thetha = Thetha - Alpha*Gradient
            it += 1
        #print(Thetha)
        
        #Classification
        yhat = []
        for i in range(0,len(Test[0])):
            Sigmoid = np.divide(1,(1 + np.exp(-np.dot(Thetha.T,Test[0][i]))))
            if(Sigmoid>0.5):
                yhat.append(1)
            else:
                yhat.append(0)
        yhat = np.asarray(yhat)
        z = sklearn.metrics.confusion_matrix(Test[1],yhat.T)
        print("Confusion Matrix:")
        print(z)
        
        #Measures
        p=[]
        r=[]
        fm=[]
        rows = np.sum(z, axis=0)
        cols = np.sum(z, axis=1)
        for i in range(0,len(classes)):
            p.append(np.divide(z[i][i],cols[i]))
            r.append(np.divide(z[i][i],rows[i]))
            fm.append(2*(np.divide(p[i]*r[i],p[i]+r[i])))
        a = np.divide(z.diagonal().sum(),z.sum())
        print("Precision is: ",p)
        print("Recall is:    ",r)
        print("F-Measure is: ",fm)
        print("Accuracy is:  ",a)
        print()