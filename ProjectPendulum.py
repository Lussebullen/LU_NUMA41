import numpy as np
import matplotlib; matplotlib.use("TkAgg") #show animation window
import matplotlib.pyplot as plt
import scipy.optimize as sp
from matplotlib import animation
import time

########################################################################################################################
#a) b) c) d)
########################################################################################################################
g,l=9.82,1

def Newton(F,guess, J):
    '''
    :param F: function to root
    :param guess: initial guess
    :param J: Jacobian of F
    :return: result from first iteration of Newtons method
    '''
    init = guess
    for i in range(1): #amount of Newton iterations
        dx = np.linalg.solve(J(init),-F(init))
        init+=dx
    return init


def ImEulerBuiltin(fp, a, b, h, init):
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
         f = lambda x: np.array([X[i],Y[i]])- x + h*fp(x[0],x[1])
         #Solve using root finding method
         nextit = sp.root(f,[X[i],Y[i]]).x
         X[i+1], Y[i+1] = nextit[0], nextit[1]
     return T,X,Y


def ImEuler(fp, a, b, h, init):         #Implicit Euler with Newton as root implementation
    '''
    :param fp: f'(x,y) as an np.array
    :param a: interval start (for plot, as it doesnt necessarily fit with h)
    :param b: interval end --||--
    :param h: step size
    :param init: initial values as list
    :return: T, X, Y approximating the solution to ODE on [a,b] using Implicit Euler with Newton root
    '''

    def Jacobian(vec):
        return np.array([[-1,h], [-g / l * h * np.cos(vec[0]), -1]])
    intlen = b - a
    n = int(np.ceil(intlen / h))
    X, Y = np.zeros(n), np.zeros(n)
    X[0], Y[0] = init[0], init[1]
    T = np.linspace(a, b, n)
    for i in range(n - 1):
        # Root form of problem to solve
        f = lambda x: np.array([X[i], Y[i]]) - x + h * fp(x[0], x[1])
        # Solve using root finding method
        nextit = Newton(f, [X[i],Y[i]], Jacobian)
        X[i + 1], Y[i + 1] = nextit[0], nextit[1]
    return T, X, Y

def trap(fp, a, b, h, init):
    '''
    :param fp: f'(x,y) as an np.array
    :param a: interval start (for plot, as it doesnt necessarily fit with h)
    :param b: interval end --||--
    :param h: step size
    :param init: initial values as list
    :return: T, X, Y approximating the solution to ODE on [a,b] using Trapezoidalrule and Newton root
    '''
    def Jacobian(vec):
        return np.array([[-1,h/2], [-g /(2*l) * h * np.cos(vec[0]), -1]])
    intlen = b - a
    n = int(np.ceil(intlen / h))
    X, Y = np.zeros(n), np.zeros(n)
    X[0], Y[0] = init[0], init[1]
    T = np.linspace(a, b, n)
    for i in range(n - 1):
        # Root form of problem to solve
        f = lambda x: np.array([X[i], Y[i]]) - x + h/2*(fp(x[0], x[1])+fp(X[i],Y[i]))
        # Solve using root finding method
        nextit = Newton(f, [X[i],Y[i]], Jacobian)
        X[i + 1], Y[i + 1] = nextit[0], nextit[1]
    return T, X, Y

#Note there is a slight difference between ImEuler and trap after many iterations, because one diverges faster than the other.

def ode(a,ap):
     return np.array([ap, -g/l*np.sin(a)])


#y0 = [np.pi/2, 0]
#T, X, Y = ImEuler(ode,0,5,0.001,y0)
#T1, X1, Y1 = ImEulerBuiltin(ode,0,5,0.001,y0)
#T2, X2, Y2 = trap(ode,0,5,0.001,y0)
#plt.plot(T,X,"r",label="a")
#plt.plot(T,Y,"b",label="a'")
#plt.xlabel("Time [s]")
#plt.legend()
#plt.show()

########################################################################################################################
#e) Obstacle version
########################################################################################################################

def Linterp(x, xarray, yarray):
    # Arrays of y and x values
    p = 0
    for i in range(len(xarray)):
        temp = yarray[i]
        for j in range(len(xarray)):
            if i != j:
                temp = temp * (x - xarray[j]) / (xarray[i] - xarray[j])

        p = p + temp
    return p

#Newton interpolation
def divdif(x,y):
    '''
    :param x: x coordinates
    :param y: y coordinates
    :return: Divided differences for set of coordinates
    '''
    n = len(y)
    mat = np.zeros((n,n))
    mat[:,0]=y
    for j in range(1,n): #col
        for k in range(n-j): #row
            mat[k][j] = (mat[k+1][j-1]-mat[k][j-1])/(x[k+j]-x[k])
    return mat

def Ninterp(x,X,Y):
    '''
    :param x: point for evaluation
    :param X: Interpolation x coordinates
    :param Y: Interpolation y coordinates
    :return: Newton interpolation of X,Y at point x
    '''
    A = divdif(X,Y)[0]
    p = A[0]
    for i in range(1,3):
        n=1
        for j in range(0,i):
            n=n*(x-X[j])
        p+=n*A[i]
    return p

def trapobN(fp, a, b, h, init,aob):
    '''
    :param fp: f'(x,y) as an np.array
    :param a: interval start (for plot, as it doesnt necessarily fit with h)
    :param b: interval end --||--
    :param h: step size
    :param init: initial values as list
    :param aob: angle of obstacle
    :return: T, X, Y approximating the solution to ODE on [a,b] using Trapezoidalrule, Newton root and interpolation.
    '''
    def Jacobian(vec):
        return np.array([[-1,h/2], [-g /(2*l) * h * np.cos(vec[0]), -1]])
    intlen = b - a
    n = int(np.ceil(intlen / h))
    X, Y = np.zeros(n), np.zeros(n)
    X[0], Y[0] = init[0], init[1]
    T = np.linspace(a, b, n)
    rt = []
    for i in range(n - 1):
        tob = []
        # Root form of problem to solve
        f = lambda x: np.array([X[i], Y[i]]) - x + h/2*(fp(x[0], x[1])+fp(X[i],Y[i]))
        # Solve using root finding method
        nextit = Newton(f, [X[i],Y[i]], Jacobian)
        X[i + 1], Y[i + 1] = nextit[0], nextit[1]
        if (i>2 and Y[i]<0): #second condition to ensure we only check for obstacle when heading for it.
            # i>2 rather than 1 to avoid root issue during initiation
            tp, xp, yp = T[i-2:i+1], X[i-2:i+1], Y[i-2:i+1]     #3 last consecutive values.
            #Newton interpolation at previous 3 points root form with obstacle angle
            apoly = lambda x: Ninterp(x, tp, xp) - aob
            #Check for root at obstacle, this time use builtin root to save some trouble
            rt1, rt2 = sp.root(apoly, tp[0]).x, sp.root(apoly, tp[2]).x
            #Check if a root is in interval
            if (tp[0]<=rt1 and rt1<=tp[2]):
                tob = tob + [rt1]
            if (tp[0]<=rt2 and rt2<=tp[2]):
                tob = tob + [rt2]
        if (len(tob)!=0):
            rt = rt + [min(tob)]
            T[i+1] = rt[-1]
            X[i+1] = apoly(rt[-1]) + aob
            #Angular velocity interpolation
            adotpoly = lambda x: Ninterp(x, tp, yp)
            Y[i+1]=-adotpoly(rt[-1])

    return T, X, Y, rt

def trapobL(fp, a, b, h, init,aob):
    '''
    :param fp: f'(x,y) as an np.array
    :param a: interval start (for plot, as it doesnt necessarily fit with h)
    :param b: interval end --||--
    :param h: step size
    :param init: initial values as list
    :param aob: angle of obstacle
    :return: T, X, Y approximating the solution to ODE on [a,b] using Trapezoidalrule, Newton root and Lagrange interp.
    '''
    def Jacobian(vec):
        return np.array([[-1,h/2], [-g /(2*l) * h * np.cos(vec[0]), -1]])
    intlen = b - a
    n = int(np.ceil(intlen / h))
    X, Y = np.zeros(n), np.zeros(n)
    X[0], Y[0] = init[0], init[1]
    T = np.linspace(a, b, n)
    rt = []
    for i in range(n - 1):
        tob = []
        # Root form of problem to solve
        f = lambda x: np.array([X[i], Y[i]]) - x + h/2*(fp(x[0], x[1])+fp(X[i],Y[i]))
        # Solve using root finding method
        nextit = Newton(f, [X[i],Y[i]], Jacobian)
        X[i + 1], Y[i + 1] = nextit[0], nextit[1]
        if (i>2 and Y[i]<0): #second condition to ensure we only check for obstacle when heading for it.
            # i>2 rather than 1 to avoid root issue during initiation
            tp, xp, yp = T[i-2:i+1], X[i-2:i+1], Y[i-2:i+1]     #3 last consecutive values.
            #Newton interpolation at previous 3 points root form with obstacle angle
            apoly = lambda x: Linterp(x, tp, xp) - aob
            #Check for root at obstacle, this time use builtin root to save some trouble
            rt1, rt2 = sp.root(apoly, tp[0]).x, sp.root(apoly, tp[2]).x
            #Check if a root is in interval
            if (tp[0]<=rt1 and rt1<=tp[2]):
                tob = tob + [rt1]
            if (tp[0]<=rt2 and rt2<=tp[2]):
                tob = tob + [rt2]
        if (len(tob)!=0):
            rt = rt + [min(tob)]
            T[i+1] = rt[-1]
            X[i+1] = apoly(rt[-1]) + aob
            #Angular velocity interpolation
            adotpoly = lambda x: Linterp(x, tp, yp)
            Y[i+1]=-adotpoly(rt[-1])

    return T, X, Y, rt


y0 = [np.pi/2, 0]
tsL = time.clock()
T, X, Y, obstacletime = trapobL(ode,0,5,0.001,y0,-1/6*np.pi)
tL = (time.clock() - tsL)

tsN = time.clock()
T, X, Y, obstacletime = trapobN(ode,0,5,0.001,y0,-1/6*np.pi)
tN = time.clock()-tsN

print("Time Lagrange:", tL)
print("Time Newton:", tN) #Newton was faster, better, stronger, harder

print(list(X))
print(list(Y))
plt.plot(T,X,"r")
plt.plot(T,Y,"b")
plt.show()
print(obstacletime)

#############
# Animation
#############

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-1.1, 1.1), ylim=(-1.2, 1))
point, = ax.plot([X[0]], [Y[0]], 'o')
plt.plot(0,0,"r.")
plt.plot(np.sin(-np.pi/6),-np.cos(-np.pi/6),"go")
# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.sin(X[i])
    y = -np.cos(X[i])
    point.set_data(x, y)
    return point,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate,frames=5000, interval=1, blit=True)

plt.show()









