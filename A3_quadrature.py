import numpy as np
import scipy.integrate as spi


def Q(f,a,b):
    return (b-a)/12*(f(a)+4*f(a+(b-a)/4)+2*f((a+b)/2)+4*f(a+3*(b-a)/4)+f(b))

def asrec(f, a, b, eps, sr):
    m = (a+b)/2

    if (abs(Q(f,a,b)-sr)<=15*eps):
        return Q(f,a,b)
    return asrec(f,a,m,eps,sr) + asrec(f,m,b,eps,sr)

def adaptive_simpsons(f,a,b,eps):
    sr = (b-a)/6*(f(a)+4*f((a+b)/2)+f(b))
    return asrec(f,a,b,eps,sr)

def f(x):
    return np.sqrt(abs(x))

print(adaptive_simpsons(f,-1,1,0.005))
print(spi.quad(f,-1,1))

