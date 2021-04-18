import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

#regular simpsons rule
def simpson(f, a, b):
    return (b-a)/6*(f(a)+4*f((a+b)/2)+f(b))

def Q(f,a,b):
    m = (a+b)/2
    return simpson(f,a,m)+simpson(f,m,b)

#Lazy error term
err = 0

#adaptive simpson recursion, including storage of interval list (div)
def asrec(f, a, b, delta, div):
    m = (a+b)/2
    div.append(m)
    if (abs(Q(f,a,b)-simpson(f,a,b))<=(b-a)*delta):
        global err
        err = err + abs(Q(f,a,b)-simpson(f,a,b))/15
        return [Q(f,a,b)+(Q(f,a,b)-simpson(f,a,b))/15, div]
    return [asrec(f,a,m,delta, div)[0] + asrec(f,m,b,delta, div)[0], div]

#function to initialize adaptive simpson.
def adaptive_simpsons(f,a,b,eps):
    delta = 15*eps/(b-a)
    div = [a]
    res, div= asrec(f,a,b,delta, div)
    div.append(b)
    div=sorted(div)
    return res, div

#test function
def f(x):
    return np.sqrt(abs(x))

integral, X = adaptive_simpsons(f,-1,1,0.000005)
exact = 4/3
print("Exact error: ",exact-integral,"Estimated error: ", err)
print(integral, spi.quad(f,-1,1))
plt.plot(X,[f(x) for x in X], "r")
plt.plot(X,[0]*len(X),"b.")
plt.show()
