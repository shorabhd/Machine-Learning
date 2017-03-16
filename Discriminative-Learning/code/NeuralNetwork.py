import numpy as np
import sklearn.metrics
 
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


#Create no. of units
units = int((X.shape[1]+len(classes))/2)

#10-Fold Cross Validation 
var = [x*(len(X)/10) for x in range(0,11)]
for i in range(len(var)):
    if i!=10:
  
        print("Fold ",i+1)
          
        #Divide Training and Test Data
        Test = X[var[i]:var[i+1]],Y[var[i]:var[i+1]]
        Train = np.concatenate((X[var[i+1]:],X[:var[i]])),np.concatenate((Y[var[i+1]:],Y[:var[i]]))

        m,n = np.shape(Train[0])
        
        Thetha = []
        #Iterate for Each Unit
        for k in range(0,units):
            
            #Create w with no. of Classes x no. of features
            #Create v for all Classes
            w =[]
            v=[]
            for k in range(0,len(classes)):
                w.append(np.random.rand(n))
                v.append(np.random.rand())
            v = np.asarray(v)
            
            niters = 100
            Alpha = 0.0001

            #Gradient Descent
            while(niters>0):
                
                GradientW = 0
                Z=[]
                for i in range(0,len(classes)):
                    Z.append(np.divide(1,1 + np.exp(-np.dot(w[i],Train[0].T))))
                    Yhat = np.dot(Z[i],v[i])
                    
                    #Indicator
                    ind =[]
                    for j in range(0,m):
                        if(Train[1][j]==i):
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
                
                for i in range(0,len(classes)):
                    for j in range(0,m):
                        GradientW += Z[i][j]*(1-Z[i][j])*Train[0][j] 
                    w[i] = w[i] - Alpha*GradientW
                    #print(w[i])
                niters -= 1
            
            #print(Thetha)
            memFunc =[]
            for i in range(len(classes)):
                memFunc.append(Thetha[i]*(np.divide(1,1 + np.exp(-np.dot(w[i],Test[0].T)))).T)
            memFunc = np.asarray(memFunc)
            #print(memFunc.shape)
            memFunc = np.reshape(memFunc, newshape=(3,len(Test[0])))
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
           