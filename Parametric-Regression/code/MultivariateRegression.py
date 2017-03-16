import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import time

#Read File and combine all the features into X and Y.
filename = "E:\MS\ML\Ass 1\Datasets\mvar-set2.dat"
X = []
Y = []
for line in open(filename):
    li=line.strip()
    if not li.startswith("#"):
        var = line.split()
        Y.append([float(var[-1])])
        X.append([float(i) for i in var[0:len(var)-1]])

#Convert the list to an array for future computations
X = np.asarray(X)
Y = np.asarray(Y)

P = PolynomialFeatures(4)

#10 Fold Cross Validation
var = [x*(len(X)/10) for x in range(0,11)]
for i in range(len(var)):
    if i!=10:

        start = time.time()
        print("Fold ",i+1)

        #Divide Training and Test Data
        Test = X[var[i]:var[i+1]],Y[var[i]:var[i+1]]
        Train = np.concatenate((X[var[i+1]:],X[:var[i]])),np.concatenate((Y[var[i+1]:],Y[:var[i]]))

        #Create Z matrix with High Dimension
        Z = P.fit_transform(Train[0])

        #Calculate Thetha
        Thetha = np.dot(np.linalg.pinv(Z),Train[1])
        print("Thetha is: ",Thetha)
        m,n = X.shape

        #Test Error
        #Calculate y Hat for Test Data
        y = []
        for i in zip(Test[0]):
            sum = Thetha[0].item(0)
            for j in range(1,n+1):
                sum = sum + Thetha[j].item(0)*i[0][j-1]
            y.append(sum)
        print("Test Y_hat is")
        print(y)
        y = np.reshape(y,newshape=(len(y),1))
        TestMSE=0
        TestMSE = TestMSE + np.divide(np.sum((np.subtract(y,Test[1])**2)),len(Test[0]))
        print("TestMSE",TestMSE)

        #Training Error
        #Calculate y Hat for Training Data
        y1 = []
        for i in zip(Train[0]):
            sum1 = Thetha[0].item(0)
            for j in range(1,n+1):
                sum1 = sum1 + Thetha[j].item(0)*i[0][j-1]
            y1.append(sum1)
        print("Training Y_hat is")
        print(y1)
        y1 = np.reshape(y1,newshape=(len(y1),1))
        TrainMSE=0
        TrainMSE = TrainMSE + np.divide(np.sum(np.subtract(y1,Train[1])**2),len(Train[0]))
        print("TrainMSE",TrainMSE)
        end = time.time()
        print("Time is:",end-start)