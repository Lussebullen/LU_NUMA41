from sympy import *
from scipy.interpolate import lagrange
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

def Linterp(x, nod, j):
    n = len(nod)
    Ltemp = 1
    for k in range(0, n):
        if (k != j):
            Ltemp = Ltemp * (x - nod[k]) / (nod[j] - nod[k])
    return Ltemp

def weight(a,b, cjs):
    cjs=np.array(cjs)
    h, n = b-a, len(cjs)
    tau = [a]*n + np.array(cjs*h)
    bj=[]
    for j in range(0,n):
        def Lj(x):
            return Linterp(x, tau, j)
        bj+=[1/h*quad(Lj,a,b)[0]]
    return bj

def order(bjs,cjs):
    q=1
    for i in range(0,2*len(cjs)):
        sum=0
        n=0
        while(n<len(cjs)):
            sum += cjs[n]**(q-1)*bjs[n]
            n+=1
        if (abs(1/q-sum)<0.0005):
            q+=1
        else:
            return q-1


#cj = np.array([1/2])
#print(order(0,1,cj))

def f(x):
    return x**3+100*x**2-20*x+7


def gaus2(f, a,b):
    return (b-a)/2*(f((b-a)/2*1/np.sqrt(3)+(a+b)/2)+f(-(b-a)/2*1/np.sqrt(3)+(a+b)/2))

print(gaus2(f,0,1))
print(quad(f,0,1)[0])