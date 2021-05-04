import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp

def ExEuler(fp, a, b, h, init):
    '''
    :param fp: f'(x,y) as an np.array
    :param a: interval start (for plot, as it doesnt necessarily fit with h)
    :param b: interval end --||--
    :param h: step size
    :param init: initial values as list
    :return: T, X, Y approximating the solution to ODE on [a,b]
    '''
    intlen = b-a
    n = int(np.ceil(intlen/h))
    X, Y = np.zeros(n), np.zeros(n)
    X[0],Y[0]=init[0], init[1]
    T = np.linspace(a,b,n)
    for i in range(n-1):
        eval = fp(X[i],Y[i])
        Xp, Yp = eval[0], eval[1]
        X[i+1], Y[i+1] = X[i]+Xp*h, Y[i]+Yp*h
    return T,X,Y

def ImEuler(fp, a, b, h, init): ##NOT DONE, doesnt work
    '''
    :param fp: f'(x,y) as an np.array
    :param a: interval start (for plot, as it doesnt necessarily fit with h)
    :param b: interval end --||--
    :param h: step size
    :param init: initial values as list
    :return: T, X, Y approximating the solution to ODE on [a,b]
    '''
    intlen = b-a
    n = int(np.ceil(intlen/h))
    X, Y = np.zeros(n), np.zeros(n)
    X[0],Y[0]=init[0], init[1]
    T = np.linspace(a,b,n)
    for i in range(n-1):
        #Root form of problem to solve
        f = lambda x: x-np.array([X[i],Y[i]])-h*fp(x[0],x[1])
        #Solve using root finding method
        nextit = sp.root(f,init).x
        X[i+1], Y[i+1] = nextit[0], nextit[1]
    return T,X,Y

def f(t):
    return 100/99*np.exp(-t)*np.array([1,0])+1/99*np.exp(-100*t)*np.array([-1,99])
#theoretical values:
T = np.linspace(0,1,100)
XY = np.array([list(f(t)) for t in T])
X, Y = XY[:,0:1], XY[:,1:2]
#plt.plot(T,X,"r")
#plt.plot(T,Y,"b")

def ode(y1,y2):
    return np.array([-y1+y2, -100*y2])
h=0.005
#Explicit euler values
Te,Xe,Ye = ExEuler(ode, 0,1,h,[1,1])

#Implicit euler values
Ti,Xi,Yi = ImEuler(ode, 0,1,h,[1,1])

plt.plot(Xi,Yi,"gx",label="implicit euler")
plt.plot(Xe,Ye,"b.",label="explicit euler")
plt.plot(X,Y,"r",label="theoretical")
plt.xlabel("Y1")
plt.ylabel("Y2")
plt.title(f"Euler methods for h={h}")
plt.legend()
plt.show()

########################################################
#Lotka-Volterra
########################################################

def LV(coeffs,x,y):
    '''
    :param coeffs: coefficients for LV function
    a -> prey growth rate
    b -> rate at which predators eat prey
    c -> predator increase rate
    d -> predator death rate
    :param x: prey
    :param y: predator
    :return: x',y' as an array
    '''
    a,b,c,d=coeffs
    return np.array([a*x-b*x*y, c*x*y-d*y])

#specific LV example function
def LVex(x,y):
    return LV([1.5,0.5,0.5,10],x,y)

#seemingly rabbits only wins if initial population of predators is 0
#but this would result in a degenerate case of the eq. system.

#T, X, Y = ExEuler(LVex,0,20,0.001,[10,5])

#plt.plot(T,X,"r",label="rabbits")
#plt.plot(T,Y,"b",label="foxes")
#plt.title("LV system")
#plt.xlabel("time (years)")
#plt.ylabel("population")
#plt.legend()
#plt.show()



