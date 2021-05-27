import numpy as np
import matplotlib; matplotlib.use("TkAgg") #show animation window
import matplotlib.pyplot as plt
import scipy.optimize as sp
from matplotlib import animation

########################################################################################################################
#a) b) c) d) only replace sp.root with own newtons method
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

def ode(a,ap):
     return np.array([ap, -g/l*np.sin(a)])


y0 = [np.pi/2, 0]
T, X, Y = ImEuler(ode,0,5,0.001,y0)
T1, X1, Y1 = ImEulerBuiltin(ode,0,5,0.001,y0)
T2, X2, Y2 = trap(ode,0,5,0.001,y0)
print(list(X))
print(list(X1))
print(list(X2))
print(max(Y2-Y1), max(X1-X2))
plt.plot(T,X,"r",label="a")
plt.plot(T,Y,"b",label="a'")
plt.xlabel("Time [s]")
plt.legend()
plt.show()

########################################################################################################################
#e) poly interpolation
########################################################################################################################


# First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure()
# ax = plt.axes(xlim=(-1.1, 1.1), ylim=(-1.2, 1))
# point, = ax.plot([X[0]], [Y[0]], 'o')
# plt.plot(0,0,"r.")
# # initialization function: plot the background of each frame
# def init():
#     line.set_data([], [])
#     return line,
#
# # animation function.  This is called sequentially
# def animate(i):
#     x = np.sin(X[i])
#     y = -np.cos(X[i])
#     point.set_data(x, y)
#     return point,
#
# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate,frames=5000, interval=1, blit=True)
#
# plt.show()

########################################################################################################################
# Attempt with Newton
########################################################################################################################

# def Newt(F, init, J):
#     '''
#     :param F: function to root
#     :param init: initial guess
#     :param J: Jacobian of F
#     :return: root of F
#     '''
#     x2=init
#     for i in range(3):
#         Dx = np.linalg.solve(J(x2),-F(x2))
#         x2 = x2+Dx
#     return x2
#
#
# g, l, dt = 9.82, 1, 0.5
#
#
# def F(a,a0):
#     return a0-a1+dt/2*np.array([a0[1]+a[1],-g/l*(np.sin(a0[0]+np.sin(a[0])))])










