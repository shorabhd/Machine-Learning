import numpy as np
import sklearn.metrics
 
#Read File and combine all the features into X and Y.
filename = "E:\MS\ML\Ass 2\Datasets\spambase.data"
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
X,B,Y =np.hsplit(Z,[54,57])

#Get Float Data
X = X.astype(np.float)
y=[]
for i in Y:
    y.append(i.item(0).strip())
Y = y
Y = np.asarray(Y)
Y = Y.astype(np.float)
 
#Convert the list to an array for future computations
classes = np.unique(Y)
 
#Change Values to 1 
m,n = np.shape(X) 
for j in range(0,n):
    for k in range(0,m):
        if X[k][j] != 0:
            X[k][j] = 1
 
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
            m,n = np.shape(Train[0])
            Alpha=[]
            for j in range(0,n):
                sum=0
                for k in range(0,m):
                    if y[k]==classes[i]:
                        sum = sum + Train[0][k][j]
                #print(sum)
                Alpha.append(np.divide(sum,count[i]))
            print(Alpha)
            mem = 0
            with np.errstate(invalid='ignore'):
                with np.errstate(divide='ignore'):
                    for j in range(0,n):
                        mem = mem + ((Test[0][:,j]*np.log(Alpha[j]))+((1-Test[0][:,j])*np.log(1-Alpha[j])))
                    mem = mem+np.log(count[i]/m)
                    memFunc.append(mem)
        memFunc = np.asarray(memFunc)
        #print(memFunc)
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