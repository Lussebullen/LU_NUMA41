# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:02:48 2021

@author: mdu
"""

import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
#Q1 divided difference matrix for Newton basis
########################################################################################################################

def divdif(x,y):
    n = len(y)
    mat = np.zeros((n,n))
    mat[:,0]=y
    for j in range(1,n): #col
        for k in range(n-j): #row
            mat[k][j] = (mat[k+1][j-1]-mat[k][j-1])/(x[k+j]-x[k])
    return mat

X=[-1,0,2,3]
Y=[2,6,4,30]

print(divdif(X,Y))

########################################################################################################################

##Lagrange interpolation function.
def Linterp(x, f, a, b, n):
    p=0
    X = np.linspace(a,b,n)  #n evenly distributed interpolationpoints on [a,b]
    Y = f(X)
    for i in range(0,n):
        Ltemp = 1
        for k in range(0,n):
            if (k!=i):
                Ltemp = Ltemp*(x-X[k])/(X[i]-X[k])
        p+=Ltemp*Y[i]
    return p

def Ninterp(x,f,a,b,n):
    X = np.linspace(a, b, n)
    Y = f(X)
    A = divdif(X,Y)[0]
    p = A[0]
    for i in range(1,n):
        n=1
        for j in range(0,i):
            n=n*(x-X[j])
        p+=n*A[i]
    return p
##endpoints and interval
a = -1
b = 1
X1 = np.linspace(a,b,50)

def f1(x):
    return np.exp(-4*x**2)

def intf1(x,n):
    return Linterp(x,f1,a,b,n)

def g1(x):
    return 1/(1+25*x**2)

def intg1(x,n):
    return Linterp(x,g1,a,b,n)


#fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(X1,intf1(X1,5), "r", label="Int n=5")
plt.plot(X1,intf1(X1,12), "bx", label="Int n=12")
plt.plot(X1,f1(X1), "g", label="y=e^(-4x^2)")
plt.legend()
plt.subplot(2,1,2)
plt.plot(X1,intg1(X1,15), "r", label="Int n=15")
plt.plot(X1,intg1(X1,21), "bx", label="Int n=21")
plt.plot(X1,g1(X1), "g", label="y=1/(1+25x^2)")
plt.legend()
plt.show()
