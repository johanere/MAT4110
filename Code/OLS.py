from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_validate
import sklearn.linear_model as skl
from sklearn.model_selection import KFold


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def r2(z1, z2):
    r2 = r2_score(z1, z2, sample_weight=None, multioutput="uniform_average")
    return r2


def MSE(z1, z2):
    MSE = mean_squared_error(z1, z2, sample_weight=None, multioutput="uniform_average")
    return MSE


class Data:
    def __init__(self, p=5,eps=1):
        """Initiates class. Provide degree of polynomial, p"""
        self.p = p
        self.eps=eps

    def GenerateData_1(self):
        """Generates a random NxN-grid and computes targets [with noise] from Franke Function. Provide N and noise [true]/false"""
        n =30   # Total number of data points
        eps = self.eps
        start = -2
        stop = 2
        x = np.linspace(start,stop,n)
        self.x=x
        r = np.random.uniform(0, 1, n) * eps
        self.y = x*(np.cos(r+0.5*x**3)+np.sin(0.5*x**3))
        self.n=n

    def GenerateData_2(self):
        n = 30
        eps = self.eps
        start = -2
        stop = 2
        x = np.linspace(start,stop,n)
        self.x=x
        r = np.random.uniform(0, 1, n) * eps
        self.y = 4*x**5- 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10 + r
        self.n=n

    def CreateDesignMatrix(self):
        """  Sets up design matrix X. Splits X in training and test data"""
        x = self.x
        y = self.y
        X = np.ones((self.n, self.p))
        for i in range(1,self.p):
            X[:,i]=x**i
        self.X = X
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, self.y, test_size=0.2)

    def QRsolve(self):
        q,r=np.linalg.qr(self.X)
        qt=np.dot(q.T,self.y)
        p=self.p
        self.beta=np.zeros(self.p)
        self.beta[p-1]=qt[self.p-1]/r[p-1,p-1]
        for i in range(self.p-2,-1,-1):
            sum=0
            for j in range(i,p):
                sum+=r[i,j]*self.beta[j]
            self.beta[i]=(qt[i]-sum)/r[i,i]
        self.ypred=np.dot(self.X, self.beta)
        print(self.beta)

    def OLS(self):
        """Stores a beta vector using OLS on currently set training data"""
        X = self.X_train
        z = self.z_train
        self.beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)

    def Ridge(self, ridge_lambda=0):
        """Stores a beta vector using Ridge on currently set training data. Provide Lambda"""
        X = self.X_train
        z = self.z_train
        dim = np.shape(np.linalg.pinv(X.T.dot(X)))[0]
        self.beta = (
            np.linalg.pinv((X.T.dot(X)) + np.identity(dim) * ridge_lambda)
            .dot(X.T)
            .dot(z)
        )

    def Train_Predict(self):
        """Target prediction utilizing training data and current beta. Stores MSE and r2 scores"""
        self.z_train_predict = np.dot(self.X_train, self.beta)
        self.MSE_train = MSE(Data.z_train, self.z_train_predict)
        self.r2_train = 0
        if self.n > 4:
            self.r2_train = r2(Data.z_train, self.z_train_predict)

    def Test_Predict(self):
        """Target prediction utilizing test data and current beta. Stores MSE and r2 scores"""
        self.z_test_predict = np.dot(self.X_test, self.beta)
        self.MSE_test = MSE(self.z_test, self.z_test_predict)
        self.r2_test = 0
        if self.n > 4:
            self.r2_test = r2(self.z_test, self.z_test_predict)

    def Skl_OLS(self, k=4):
        clf = skl.LinearRegression().fit(self.X, self.y)
        self.ytilde = clf.predict(self.X)


    def Skl_Lasso(self, k=10):
        lasso = skl.Lasso().fit(self.X_train, self.z_train)
        ytilde = lasso.predict(self.X_train)
        self.MSE_SKL_Lasso = MSE(Data.z_train, ytilde)
        self.r2_SKL_Lasso = 0
        if self.n > 4:
            self.r2_SKL_Lasso = r2(Data.z_train, ytilde)
        cv_results = cross_validate(lasso, self.X, self.z, cv=k)
        print("Lasso score \n", cv_results["test_score"])

    def Kfold_Crossvalidation(self, method=0, llambda=0, k=10):
        "k fold cross validation. 0-OLS, 1-Ridge. Gets MSE from test predict"
        self.kscores = np.zeros((2, k))
        kfold = KFold(k, shuffle=True)
        j = 0
        for train_inds, test_inds in kfold.split(self.X):
            self.X_train, self.X_test = self.X[train_inds], self.X[test_inds]
            self.z_train, self.z_test = self.z[train_inds], self.z[test_inds]
            if method == 0:
                self.OLS()
            if method == 1:
                self.Ridge(llambda)
            self.Train_Predict()
            self.Test_Predict()
            self.kscores[0, j] = self.MSE_test
            self.kscores[1, j] = self.r2_test
            j += 1


np.random.seed(3)
data=Data(p=5,eps=0)
data.GenerateData_1()
data.CreateDesignMatrix()
data.QRsolve()
data.Skl_OLS()
plt.scatter(data.x,data.y)
plt.plot(data.x,data.ytilde)
plt.plot(data.x,data.ypred)
plt.show()
