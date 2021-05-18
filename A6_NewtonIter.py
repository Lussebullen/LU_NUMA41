import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
########################################################################################################################
#Sketch
########################################################################################################################
def fcn1(x):
    y1=[0]*len(x)
    for i in range(len(x)):
        y1[i]=math.sqrt(4-x[i]*x[i])
    return y1

def fcn2(x):
    y2=[0]*len(x)
    for i in range(len(x)):
        y2[i]=math.sqrt(1-1/16*x[i]*x[i])
    return y2

x1=np.linspace(-2,2,1000)
x2=np.linspace(-4,4,1000)
y1=fcn1(x1)
y2=fcn2(x2)

fig, ax = plt.subplots()
ax.plot(x1, y1,color="C0")
ax.plot(x2, y2,color="C1")
ax.plot(x1, np.negative(y1),color="C0")
ax.plot(x2, np.negative(y2),color="C1")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

########################################################################################################################
# a posteriori
########################################################################################################################

def g(par):
    x, y = par
    return [math.sqrt(4-y**2),math.sqrt(1-x**2/16)]

def apost(L,x1,x2):
    dif = abs(np.array(x1) - np.array(x2))
    return L/(1-L)*max(dif)

x0, x1, eps, iter, err = [2,1], [], 10**(-4), 0, 0
L = 1.4/math.sqrt(4-1.4**2)

for i in range(20):
    x1=g(x0)
    err = apost(L,x0,x1)
    iter += 1
    if err>eps:
        x0=x1
    else:
        break
print("iterations:", iter, "final error:", err)
print("fixed point:", x1)



