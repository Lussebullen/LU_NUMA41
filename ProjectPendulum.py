import numpy as np
import matplotlib; matplotlib.use("TkAgg") #show animation window
import matplotlib.pyplot as plt
import scipy.optimize as sp
from matplotlib import animation

########################################################################################################################
#a) b) c) d) only replace np.root with own newtons method
########################################################################################################################

def ImEuler(fp, a, b, h, init):
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


def ode(a,ap):
    g, l = 9.82, 1
    return np.array([ap, -g/l*np.sin(a)])


# y0 = [np.pi/2, 0]
# T, X, Y = ImEuler(ode,0,5,0.001,y0)
# plt.plot(T,X,"r",label="a")
# plt.plot(T,Y,"b",label="a'")
# plt.xlabel("Time [s]")
# plt.legend()
# plt.show()

########################################################################################################################
#e) poly interpolation
########################################################################################################################


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-1.1, 1.1), ylim=(-1.2, 1))
point, = ax.plot([X[0]], [Y[0]], 'o')
plt.plot(0,0,"r.")
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
anim = animation.FuncAnimation(fig, animate,
                               frames=5000, interval=1, blit=True)

plt.show()

