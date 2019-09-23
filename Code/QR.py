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
    def __init__(self, p=5, k=10):
        self.p = p  # Degree of polynomial
        self.kfolds = k

    def GenerateData_1(self):
        N=30
        start = -2;
        stop = 2;
        x = np.linspace(start,stop,n);
        eps = 1;
        r = rand(1,n) * eps;
        print(r)

    def CreateDesignMatrix_X(self):
        """
    	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
    	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
    	"""
        x = self.x
        y = self.y
        p = self.p
        N = len(x)
        l = int((p + 1) * (p + 2) / 2)  # Number of elements in beta

        X = np.ones((N, l))
        for i in range(1, p + 1):
            q = int((i) * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = x ** (i - k) * y ** k
            self.X = X
            self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(
                X, self.z, test_size=0.2
            )

    def Train(self):
        X = self.X_train
        z = self.z_train
        self.beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)

    def Predict(self):
        self.z_train_predict = np.dot(self.X_train, self.beta)
        self.MSE_train = MSE(Data.z_train, self.z_train_predict)
        self.r2_train = 0
        if self.n > 4:
            self.r2_train = r2(Data.z_train, self.z_train_predict)

    def Test(self):
        self.z_test_predict = np.dot(self.X_test, self.beta)
        self.MSE_test = MSE(self.z_test, self.z_test_predict)
        self.r2_test = 0
        if self.n > 4:
            self.r2_test = r2(self.z_test, self.z_test_predict)

    def SklReg(self):
        clf = skl.LinearRegression().fit(self.X_train, self.z_train)
        self.ytilde = clf.predict(self.X_train)
        self.MSE_SKL = MSE(Data.z_train, Data.ytilde)
        self.r2_SKL = 0
        if self.n > 4:
            self.r2_SKL = r2(Data.z_train, Data.ytilde)
        cv_results = cross_validate(clf, self.X, self.z, cv=self.kfolds)
        print(cv_results["test_score"])

    def xval(self):
        k = self.kfolds
        self.kscores = np.zeros((2, k))
        kfold = KFold(k, shuffle=True)
        j = 0
        for train_inds, test_inds in kfold.split(self.X):
            self.X_train, self.X_test = self.X[train_inds], self.X[test_inds]
            self.z_train, self.z_test = self.z[train_inds], self.z[test_inds]
            self.Train()
            self.Predict()
            self.Test()
            self.kscores[0, j] = self.MSE_test
            self.kscores[1, j] = self.r2_test
            j += 1

    def QR(self):
        q, r = np.linalg.qr(self.X)
        print(r)
