import numpy as np
import matplotlib.pyplot as plt

class OLS:
    def __init__(self, m=3, n=6, set=0):
        self.m = m
        self.n = n
        self.set = set
        self.A=np.ones((n,m))

    def construct_matr(self):
        start = -2
        stop = 2
        x = np.linspace(start, stop, self.n)
        eps = 1
        r = np.random.rand(self.n) * eps
        if set == 0:
            y = x * (np.cos(r + 0.5 * x ** 3) + np.sin(0.5 * x ** 3))
        else:
            y = 4 * x ** 5 - 5 * x ** 4 - 20 * x ** 3 + 10 * x ** 2 + 40 * x + 10 + r
        for i in range(1, self.m):
            self.A[:, i] = x ** i
    def QR(self):


matr=OLS()
matr.construct_matr()
print(matr.A)



"""
A = construct_matr(n, m, 0)
q, r = np.linalg.qr(A)

print(q)
print(r)
"""
