import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plot

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

P = PolynomialFeatures(2)

#10 Fold Cross Validation
var = [x*(len(X)/10) for x in range(0,11)]
for i in range(len(var)):
    if i!=10:

        print("Fold ",i+1)

        #Divide Training and Test Data
        Test = X[var[i]:var[i+1]],Y[var[i]:var[i+1]]
        Train = np.concatenate((X[var[i+1]:],X[:var[i]])),np.concatenate((Y[var[i+1]:],Y[:var[i]]))

        #Create Z matrix with High Dimension
        Z = P.fit_transform(Train[0])

        m,n = np.shape(Z)

        #Create Thetha with no. of features
        Thetha = np.ones(n)
        niters = 100
        Alpha = 0.0001
        Cost=[]
        #Iteration Loop
        for x in range(niters):
            Loss = 0
            Gradient = np.zeros(len(Thetha))
            #No. of Rows Loop in Z matrix
            for i in range(len(Z)):
                Hypothesis = np.dot(Thetha.transpose(),Z[i])    #Calculate Hypothesis
                Gradient_tmp = (Hypothesis - Train[1][i])*Z[i]  #Calculate Gradient
                Gradient = Gradient + Gradient_tmp
                Loss = Loss + np.power((Hypothesis - Train[1][i]),2)    #Calculate Loss to compute Cost
            Thetha = Thetha - Gradient*Alpha    #Calculate Thetha
            Loss = np.divide(Loss,(2*len(Z)))   #Calculate Cost
            Cost.append(Loss)
        print("Thetha is: ",Thetha)
        plot.title("Cost Function")
        plot.plot([i for i in range(0,niters)],Cost,c='green',linewidth=2.5)
        plot.show()


        #Calculate y Hat for Training Data
        y1 = []
        for j in Z:
            y1.append(np.dot(np.array(Thetha).transpose(),np.array(j)))
        print("Training Y_hat is")
        print(y1)

        #Training Error
        y1 = np.reshape(y1,newshape=(len(y1),1))
        TrainMean = np.mean(Train[1])
        TrainMSE=0
        TrainMSE = TrainMSE + np.divide(np.sum(np.subtract(y1,Train[1])**2),len(Train[0]))
        print("TrainMSE",TrainMSE)


        #Calculate y Hat for Test Data
        ZTest = P.fit_transform(Test[0])
        y = []
        for i in ZTest:
            y.append(np.dot(np.array(Thetha).transpose(),np.array(i)))
        print("Test Y_hat is")
        print(y)

        #Test Error
        y = np.reshape(y,newshape=(len(y),1))
        TestMean = np.mean(Test[1])
        TestMSE=0
        TestMSE = TestMSE + np.divide(np.sum((np.subtract(y,Test[1])**2)),len(Test[0]))
        print("TestMSE",TestMSE)


