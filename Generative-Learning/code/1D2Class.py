import numpy as np
import sklearn.metrics
 
#Read File
filename = "E:\MS\ML\Ass 2\Datasets\iris.data"

Z = []
with open(filename, "r") as filestream:
    for line in filestream:
        current = line.split(",")
        Z.append([i for i in current[3:len(current)]])
        
#Take 2 Class and Shuffle Matrix for Cross Validation
Z = np.delete(Z,np.s_[100::],axis=0) 
Z = np.asarray(Z)
np.random.shuffle(Z)

#Split Matrix in X,Y
X = []
Y = []
X,B,Y =np.hsplit(Z,[1,1])

#Get Float Data
X = X.astype(np.float)
y=[]
for i in Y:
    y.append(i.item(0).strip())
Y = y

#Get Class Name & Convert the list to an array for future computations
classes = np.unique(Y)
Y = np.asarray(Y)

#10-Fold Cross Validation
var = [x*(len(X)/10) for x in range(0,11)]
for i in range(len(var)):
    if i!=10:
 
        print("Fold ",i+1)
         
        #Divide Training and Test Data
        Test = X[var[i]:var[i+1]],Y[var[i]:var[i+1]]
        Train = np.concatenate((X[var[i+1]:],X[:var[i]])),np.concatenate((Y[var[i+1]:],Y[:var[i]]))
         
        count = []
        y=[]
        y=Train[1]
        y = y.tolist()
        for i in range(0,len(classes)):
            count.append(y.count(classes[i]))
        y = np.asarray(y) 
         
        memFunc = []
        for i in range(0,len(classes)):
            sum = 0
            Mu = []
            for j in range(len(Train[0])):
                if y[j]==classes[i]:
                    sum = sum + Train[0][j]
            Mu.append(np.divide(sum,count[i]))
            #print(Mu)
            mean = 0
            Sigma = []
            for j in range(len(Train[0])):
                if y[j]==classes[i]:
                   mean = mean + np.power(np.subtract(Train[0][j],Mu),2)
            Sigma.append(np.divide(mean,count[i]))
            #print(Sigma)
            sub = np.power(np.subtract(Test[0],Mu),2)
            memFunc.append(np.log(np.power((2*np.pi),0.5))-np.log(Sigma)-sub/(2*np.power(Sigma,2))+np.log(count[i]/len(Train[0])))
        #print(memFunc)
        memFunc = np.asarray(memFunc)
        memFunc = np.reshape(memFunc, newshape=(2,10))
        yhat = []
        for i in memFunc.T:
            if i[:][0] > i[:][1]:
                yhat.append(classes[0])
            else:
                yhat.append(classes[1])
        yhat = np.asarray(yhat)
        z = sklearn.metrics.confusion_matrix(Test[1],yhat.T)
        print("Confusion Matrix:")
        print(z)
        
        #Measures
        p = []
        r = []
        fm = []
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
        