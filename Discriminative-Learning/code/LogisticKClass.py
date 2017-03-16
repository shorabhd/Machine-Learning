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
Z = np.asarray(Z)
np.random.shuffle(Z)

#Split Matrix in X,Y
X = []
Y = []
X,B,Y =np.hsplit(Z,[Z.shape[1]-1,Z.shape[1]-1])

#Non-Linear
X = PolynomialFeatures(1).fit_transform(X)

#Get Float Data
X = X.astype(np.float)
classes = np.unique(Y)
for i in range(0,len(classes)):
    classes[i] = classes[i].strip()
classes = np.unique(classes)

y = []
for i in range(0,len(Y)):
    for j in range(0,len(classes)):
        if (Y[i].item(0).strip() == classes[j]):
            y.append(j)
Y = np.asarray(y)
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
        Thetha =[]
        for i in range(0,len(classes)):
            Thetha.append(np.random.rand(n))
        Thetha = np.asarray(Thetha)        
        niters = 100
        Alpha = 0.0001
          
        #Gradient Descent   
        while(niters>0):
            for j in range(0,len(classes)):    
                
                #Indicator
                ind =[]
                for i in range(0,m):
                    if(Train[1][i]==j):
                        ind.append(1)
                    else:
                        ind.append(0)
                
                func=0
                Hypothesis = np.dot(Train[0],Thetha[j].T)
                Gradient = Hypothesis - ind 
                func = np.dot(Gradient,Train[0])
                Thetha[j] = Thetha[j] - Alpha*func
            
            niters -= 1
        #print(Thetha)
        
        #Classification
        memFunc =[]
        for i in range(len(classes)):
            memFunc.append(np.dot(Thetha[i],Test[0].T))
        memFunc = np.asarray(memFunc)
        memFunc = np.reshape(memFunc, newshape=(len(classes),len(Test[1])))
        yhat = []
        for i in memFunc.T:
            if i[:][0] > i[:][1] and i[:][0] > i[:][2]:
                yhat.append(0)
            elif i[:][1] > i[:][0] and i[:][1] > i[:][2]:
                yhat.append(1)
            else:
                yhat.append(2)
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
        with np.errstate(invalid='ignore'):
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