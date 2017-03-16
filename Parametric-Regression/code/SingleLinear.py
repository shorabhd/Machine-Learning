import numpy as np
import matplotlib.pyplot as mt
from sklearn.linear_model import LinearRegression

#Read File into X and Y.
X,Y = np.loadtxt("E:\MS\ML\Ass 1\Datasets\svar-set2.dat",unpack=True)

#10 Fold Cross Validation
var = [x*(200/10) for x in range(0,11)]
for i in range(len(var)):
    if i!=10:
        print("Fold ",i+1)
        #Divide Training and Test Data
        Test = X[var[i]:var[i+1]],Y[var[i]:var[i+1]]
        Train = np.append(X[var[i+1]:],X[:var[i]]),np.append(Y[var[i+1]:],Y[:var[i]])

        #Calculate A matrix
        DataA = [len(Train[0]),sum(Train[0])],[sum(Train[0]),sum(pow(Train[0],2))]
        A = np.matrix(DataA)

        #Calculate B matrix
        DataB = [sum(Train[1]),sum(Train[0]*Train[1])]
        B = np.matrix(DataB).transpose()

        #Calculate Thetha by solving A,B
        Thetha = np.linalg.solve(A,B)
        print("Thetha is: ",Thetha)

        #Calculate y Hat for Test Data
        y = []
        for i in Test[0]:
            y.append(Thetha[0].item(0) + Thetha[1].item(0)*i)
        print("Test Y_hat is")
        print(y)

        #Test Error
        TestMean = np.mean(Test[1])
        TestRSE=0
        TestMSE=0
        TestRSE = TestRSE + np.divide(np.sum(np.divide((np.subtract(y,Test[1])**2),(np.subtract(TestMean,Test[1])**2))),len(Test[0]))
        TestMSE = TestMSE + np.divide(np.sum((np.subtract(y,Test[1])**2)),len(Test[0]))
        print("TestRSE",TestRSE)
        print("TestMSE",TestMSE)


        #Calculate y Hat for Training Data
        y1 = []
        for i in Train[0]:
            y1.append(Thetha[0].item(0) + Thetha[1].item(0)*i)
        print("Training Y_hat is")
        print(y1)

        #Training Error
        TrainMean = np.mean(Train[1])
        TrainRSE=0
        TrainMSE=0
        TrainRSE = TrainRSE + np.divide(np.sum(np.divide((np.subtract(y1,Train[1])**2),(np.subtract(TrainMean,Train[1])**2))),len(Train[0]))
        TrainMSE = TrainMSE + np.divide(np.sum(np.subtract(y1,Train[1])**2),len(Train[0]))
        print("TrainRSE",TrainRSE)
        print("TrainMSE",TrainMSE)

        mt.scatter(Test[0],Test[1])
        mt.plot(Test[0],y)
        mt.show()

        #Fit to a Linear Model
        LP = LinearRegression()
        LP.fit(Train[0].reshape(len(Train[0]),1),Train[1].reshape(len(Train[1]),1))
        LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        print("Thetha with LR: ",LP.intercept_,LP.coef_)
        ylp = []
        for i in Test[0]:
            ylp.append(LP.intercept_.item(0) + LP.coef_.item(0)*i)
        print("LR Test Y_hat is")
        print(ylp)
        #Calculate y Hat for Test Data
        TestMean = np.mean(Test[1])
        TestRSE=0
        TestMSE=0
        TestRSE = TestRSE + np.divide(np.sum(np.divide((np.subtract(ylp,Test[1])**2),(np.subtract(TestMean,Test[1])**2))),len(Test[0]))
        TestMSE = TestMSE + np.divide(np.sum((np.subtract(ylp,Test[1])**2)),len(Test[0]))
        print("LR TestRSE",TestRSE)
        print("LR TestMSE",TestMSE)

        #Training Error
        #Calculate y Hat for Training Data
        ylp1 = []
        for i in Train[0]:
            ylp1.append(LP.intercept_.item(0) + LP.coef_.item(0)*i)
        print("LR Training Y_hat is")
        print(ylp1)
        TrainMean = np.mean(Train[1])
        TrainRSE=0
        TrainMSE=0
        TrainRSE = TrainRSE + np.divide(np.sum(np.divide((np.subtract(ylp1,Train[1])**2),(np.subtract(TrainMean,Train[1])**2))),len(Train[0]))
        TrainMSE = TrainMSE + np.divide(np.sum(np.subtract(ylp1,Train[1])**2),len(Train[0]))
        print("LR TrainRSE",TrainRSE)
        print("LR TrainMSE",TrainMSE)

