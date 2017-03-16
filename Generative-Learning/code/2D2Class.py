import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import sklearn.metrics
 
#Read File
filename = "E:\MS\ML\Ass 2\Datasets\iris.data"

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
X,B,Y =np.hsplit(Z,[4,4])

X = X.astype(np.float)

#Get Float Data
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
        
        mem = []
        memFunc = []
        for i in range(0,len(classes)):
            sum = 0
            Mu = []
            for j in range(len(Train[0])):
                if y[j]==classes[i]:
                    sum = sum + Train[0][j][:]
            Mu.append(np.divide(sum,count[i]))
            #print(Mu)
            m,n = np.shape(Train[0])
            mean = np.zeros(shape=(n,n))
            Sigma = []
            for j in range(len(Train[0])):
                if y[j]==classes[i]:
                    sub = np.subtract(Train[0][j].T,Mu)
                    dot = np.dot(sub.T,sub)
                    mean = mean + dot
            Sigma.append(np.divide(mean,count[i]))
            #print(Sigma)
            mem = []
            for j in range(0,len(Test[0])):
                sub = np.subtract(Test[0][j],Mu)
                dot = np.dot(np.dot(sub,np.linalg.inv(Sigma)),sub.T)
                mem.append(-np.log(np.power((2*np.pi),2))-np.log(np.linalg.det(Sigma))-0.5*dot+np.log(count[i]/len(Train[0])))
            memFunc.append(mem)
        #print(memFunc)
        memFunc = np.asarray(memFunc)
        memFunc = np.reshape(memFunc, newshape=(2,10))
        
        #Binarize Data Label
        ytrue=[]
        for i in Test[1]:
            if i == classes[0]:
                ytrue.append(0)
            else:
                ytrue.append(1)
        
        ypred=[]
        
        yhat = []
        for i in memFunc.T:
            if i[:][0] > i[:][1]:
                ypred.append(i[:][0] - i[:][1])
                yhat.append(classes[0])
            else:
                ypred.append(-i[:][0] + i[:][1])
                yhat.append(classes[1])
        yhat = np.asarray(yhat)
        z = sklearn.metrics.confusion_matrix(Test[1],yhat.T)
        print("Confusion Matrix:")
        print(z)
        
        #PR-Curve
        pr,re,th = precision_recall_curve(ytrue, ypred)
        plt.clf()
        plt.plot(re,pr,label='Precision-Recall curve')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall: AUC={0:0.2f}'.format(average_precision_score(ytrue, ypred)))
        plt.show()
        
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