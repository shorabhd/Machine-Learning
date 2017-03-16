import numpy as np
import matplotlib.pyplot as mt
import numpy.polynomial.polynomial as poly

#Read File into X and Y.
X,Y = np.loadtxt("E:\MS\ML\Ass 1\Datasets\svar-set2.dat",unpack=True)

#Get the Polynomial Degree from the user
degree = int(input("Enter Degree of Polynomial "))

#10 Fold Cross Validation
var = [x*(len(X)/10) for x in range(0,11)]
for i in range(len(var)):
    if i!=10:
        print("Fold ",i+1)

        #Divide Training and Test Data
        Test = X[var[i]:var[i+1]],Y[var[i]:var[i+1]]
        Train = np.append(X[var[i+1]:],X[:var[i]]),np.append(Y[var[i+1]:],Y[:var[i]])

        #Add one additional column to X and create Z matrix
        X0 = np.ones((1,len(Train[0])))
        Z = np.vstack(X0)

        #Concate the Z matrix for different degree of polynomial
        for i in range(1,degree+1):
            Z = np.vstack((Z,Train[0]**i))
        Z = Z.transpose()

        #Calculate Thetha
        Thetha = np.dot(np.linalg.pinv(Z),Train[1])
        print(Thetha)

        #Test Error
        #Calculate y Hat for Test Data
        y = []
        for i in Test[0]:
            sum = Thetha[0].item(0)
            for j in range(1,degree+1):
                sum = sum + Thetha[j].item(0)*(i**j)
            y.append(sum)
        print("Test Y_hat is")
        print(y)
        TestMean = np.mean(Test[1])
        TestRSE=0
        TestMSE=0
        TestRSE = TestRSE + np.divide(np.sum(np.divide((np.subtract(y,Test[1])**2),(np.subtract(TestMean,Test[1])**2))),len(Test[0]))
        TestMSE = TestMSE + np.divide(np.sum((np.subtract(y,Test[1])**2)),len(Test[0]))
        print("TestRSE",TestRSE)
        print("TestMSE",TestMSE)

        #Training Error
        #Calculate y Hat for Training Data
        y1 = []
        for i in Train[0]:
            sum1 = Thetha[0].item(0)
            for j in range(1,degree+1):
                sum1 = sum1 + Thetha[j].item(0)*(i**j)
            y1.append(sum1)
        print("Training Y_hat is")
        print(y1)
        TrainMean = np.mean(Train[1])
        TrainRSE=0
        TrainMSE=0
        TrainRSE = TrainRSE + np.divide(np.sum(np.divide((np.subtract(y1,Train[1])**2),(np.subtract(TrainMean,Train[1])**2))),len(Train[0]))
        TrainMSE = TrainMSE + np.divide(np.sum(np.subtract(y1,Train[1])**2),len(Train[0]))
        print("TrainRSE",TrainRSE)
        print("TrainMSE",TrainMSE)

        #Plot Graph
        mt.scatter(Test[0],Test[1])
        plot = Test[0].argsort()
        mt.plot((Test[0])[plot],np.asarray(y)[plot])
        mt.show()

        #Fit to a Linear Model

        coefs = poly.polyfit(np.asarray(Train[0]),np.asarray(Train[1]),deg=degree)
        print(coefs)

        #Test Error
        #Calculate y Hat for Test Data
        ylp = []
        for i in Test[0]:
            sum = coefs[0].item(0)
            for j in range(1,degree+1):
                sum = sum + coefs[j].item(0)*(i**j)
            ylp.append(sum)
        print("Poly Test Y_hat is")
        print(ylp)
        TestMean = np.mean(Test[1])
        TestRSE=0
        TestMSE=0
        TestRSE = TestRSE + np.divide(np.sum(np.divide((np.subtract(ylp,Test[1])**2),(np.subtract(TestMean,Test[1])**2))),len(Test[0]))
        TestMSE = TestMSE + np.divide(np.sum((np.subtract(ylp,Test[1])**2)),len(Test[0]))
        print("PolyFit TestRSE",TestRSE)
        print("PolyFit TestMSE",TestMSE)

        #Training Error
        #Calculate y Hat for Training Data
        ylp1 = []
        for i in Train[0]:
            sum1 = coefs[0].item(0)
            for j in range(1,degree+1):
                sum1 = sum1 + coefs[j].item(0)*(i**j)
            ylp1.append(sum1)
        print("Poly Training Y_hat is")
        print(ylp1)
        TrainMean = np.mean(Train[1])
        TrainRSE=0
        TrainMSE=0
        TrainRSE = TrainRSE + np.divide(np.sum(np.divide((np.subtract(ylp1,Train[1])**2),(np.subtract(TrainMean,Train[1])**2))),len(Train[0]))
        TrainMSE = TrainMSE + np.divide(np.sum(np.subtract(ylp1,Train[1])**2),len(Train[0]))
        print("PolyFit TrainRSE",TrainRSE)
        print("PolyFit TrainMSE",TrainMSE)