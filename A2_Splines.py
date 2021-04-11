import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

########################################################################################################################
# Problem 2:
########################################################################################################################

def cubspline(xint, yint):
    m = len(xint)-1
    # equally distributed interpolation points
    h= xint[1]-xint[0]
    #prep of matrix encoding recurison coefficients
    A=4*np.identity(m-1)
    for i in range(0,m-2):
        A[i+1][i]=1
        A[i][i+1]=1

    #prep of constant vector
    Y=6/h**2*np.array([yint[i+2]-2*yint[i+1]+yint[i] for i in range(0,m-1)])

    #sigmas, added initial conditions
    sig = np.linalg.solve(A,Y)
    sig = np.insert(sig,0,0,axis=0)
    sig = np.insert(sig, m, 0, axis=0)

    coeff = np.zeros((m,4))
    for i in range(0,m):
        coeff[i][0] = (sig[i+1]-sig[i])/(6*h)
        coeff[i][1] = sig[i]/2
        coeff[i][2] = (yint[i+1]-yint[i])/h-h*(2*sig[i]+sig[i+1])/6
        coeff[i][3] = yint[i]

    return coeff

def cubsplineval(coeff, xint, xval):
    if not isinstance(xval,type(np.array([1,1]))):
        xval = np.array([xval])
    Y = np.zeros(len(xval))
    for i in range(0,len(xval)):
        index = 0
        for k in range(0,len(xint)):
            if xval[i]> xint[k]:
                index = k
        #index = bs.bisect_left(xint, xval[i])
        if index == len(xint) - 1:
            index = index -1
        yval = np.array([(xval[i] - xint[index]) ** 3, (xval[i] - xint[index]) ** 2, (xval[i] - xint[index]), 1])
        Y[i] = np.dot(coeff[index],yval)
    return Y

def f1(x):
    return np.exp(-4*x**2)

def f2(x):
    return 1/(1+25*x**2)

#First plot f1:
X1 = np.linspace(-1,1,100)
X = np.linspace(-1,1,50)
Xint = np.linspace(-1,1,5)
Yint = f1(Xint)
plt.subplot(2,1,1)
plt.plot(Xint,Yint,"ro",label="Interpolation points, n=5")
coeff = cubspline(Xint,Yint)
plt.plot(X,cubsplineval(coeff,Xint,X),"b.",label="Cubic spline")
plt.plot(X1,f1(X1),"g",label="y=exp(-4x^2)")
cspline = si.CubicSpline(Xint,Yint,bc_type="natural")
plt.legend()
#second plot f1
plt.subplot(2,1,2)
Xint=np.linspace(-1,1,12)
Yint = f1(Xint)
plt.plot(Xint,Yint,"ro",label="Interpolation points, n=12")
coeff = cubspline(Xint,Yint)
plt.plot(X,cubsplineval(coeff,Xint,X),"b.",label="Cubic spline")
plt.plot(X1,f1(X1),"g",label="y=exp(-4x^2)")
cspline = si.CubicSpline(Xint,Yint,bc_type="natural")
plt.legend()
plt.show()

#First plot f2:
Xint = np.linspace(-1,1,15)
Yint = f2(Xint)
plt.subplot(2,1,1)
plt.plot(Xint,Yint,"ro",label="Interpolation points, n=15")
coeff = cubspline(Xint,Yint)
plt.plot(X,cubsplineval(coeff,Xint,X),"b.",label="Cubic spline")
plt.plot(X1,f2(X1),"g",label="y=1/(1+25x^2)")
cspline = si.CubicSpline(Xint,Yint,bc_type="natural")
plt.legend()
#second plot f2:
plt.subplot(2,1,2)
Xint=np.linspace(-1,1,21)
Yint = f2(Xint)
plt.plot(Xint,Yint,"ro",label="Interpolation points, n=21")
coeff = cubspline(Xint,Yint)
plt.plot(X,cubsplineval(coeff,Xint,X),"b.",label="Cubic spline")
plt.plot(X1,f2(X1),"g",label="y=y=1/(1+25x^2)")
cspline = si.CubicSpline(Xint,Yint,bc_type="natural")
plt.legend()
plt.show()


########################################################################################################################
# Problem 3:
########################################################################################################################

def s1002(S):
    """
    this function describes the wheel profile s1002
    according to the standard.
    S  independent variable in mm bewteen -69 and 60.
    wheel   wheel profile value
    (courtesy to Dr.H.Netter, DLR Oberpfaffenhofen)
                                             I
                                             I
                     IIIIIIIIIIIIIIIIIIIIIIII
                   II  D  C       B       A
                  I
       I         I   E
        I       I
     H   I     I   F
          IIIII

            G


    FUNCTIONS:
    ----------
    Section A:   F(S) =   AA - BA * S
    Section B:   F(S) =   AB - BB * S    + CB * S**2 - DB * S**3
                             + EB * S**4 - FB * S**5 + GB * S**6
                             - HB * S**7 + IB * S**8
    Section C:   F(S) = - AC - BC * S    - CC * S**2 - DC * S**3
                             - EC * S**4 - FC * S**5 - GC * S**6
                             - HC * S**7
    Section D:   F(S) = + AD - SQRT( BD**2 - ( S + CD )**2 )
    Section E:   F(S) = - AE - BE * S
    Section F:   F(S) =   AF + SQRT( BF**2 - ( S + CF )**2 )
    Section G:   F(S) =   AG + SQRT( BG**2 - ( S + CG )**2 )
    Section H:   F(S) =   AH + SQRT( BH**2 - ( S + CH )**2 )
    """
    #    Polynom coefficients:
    #     Section A:
    AA = 1.364323640
    BA = 0.066666667

    #     Section B:
    AB = 0.000000000
    BB = 3.358537058e-02
    CB = 1.565681624e-03
    DB = 2.810427944e-05
    EB = 5.844240864e-08
    FB = 1.562379023e-08
    GB = 5.309217349e-15
    HB = 5.957839843e-12
    IB = 2.646656573e-13
    #     Section C:
    AC = 4.320221063e+03
    BC = 1.038384026e+03
    CC = 1.065501873e+02
    DC = 6.051367875
    EC = 2.054332446e-01
    FC = 4.169739389e-03
    GC = 4.687195829e-05
    HC = 2.252755540e-07
    #     Section D:
    AD = 16.446
    BD = 13.
    CD = 26.210665
    #     Section E:
    AE = 93.576667419
    BE = 2.747477419
    #     Section F:
    AF = 8.834924130
    BF = 20.
    CF = 58.558326413
    #     Section G:
    AG = 16.
    BG = 12.
    CG = 55.
    #     Section H:
    AH = 9.519259302
    BH = 20.5
    CH = 49.5
    """
     Bounds
                       from                    to
    Section A:      Y = + 60               Y = + 32.15796
    Section B:      Y = + 32.15796         Y = - 26.
    Section C:      Y = - 26.              Y = - 35.
    Section D:      Y = - 35.              Y = - 38.426669071
    Section E:      Y = - 38.426669071     Y = - 39.764473993
    Section F:      Y = - 39.764473993     Y = - 49.662510381
    Section G:      Y = - 49.662510381     Y = - 62.764705882
    Section H:      Y = - 62.764705882     Y = - 70.
    """
    YS = [-70., -62.764705882, -49.662510381, -39.764473993, -38.426669071, -35., -26., 32.15796, 60.]
    if (S < YS[1]):
        #       Section H (Circular arc)
        radiant = BH ** 2 - (S + CH) ** 2
        sqroot = np.sqrt(radiant)
        wheel = AH + sqroot
    elif (S < YS[2]):
        #       Section G (Circular arc)
        radiant = BG ** 2 - (S + CG) ** 2
        sqroot = np.sqrt(radiant)
        wheel = AG + sqroot
    elif (S < YS[3]):
        #       Section F (Circular arc)
        radiant = BF ** 2 - (S + CF) ** 2
        sqroot = np.sqrt(radiant)
        wheel = AF + sqroot
    elif (S < YS[4]):
        #       Section E (LINEAR)
        wheel = -BE * S - AE
    elif (S < YS[5]):
        #       Section D (Circular arc)
        radiant = BD ** 2 - (S + CD) ** 2
        sqroot = np.sqrt(radiant)
        wheel = AD - sqroot
    elif (S < YS[6]):
        #       Section C
        wheel = - AC - BC * S - CC * S ** 2 - DC * S ** 3 - EC * S ** 4 - FC * S ** 5 - GC * S ** 6 - HC * S ** 7;
    elif (S < YS[7]):
        #       Section B
        wheel = AB - BB * S + CB * S ** 2 - DB * S ** 3 + EB * S ** 4 - FB * S ** 5 + GB * S ** 6 - HB * S ** 7 + IB * S ** 8;
    else:
        #       Section A (LINEAR)
        wheel = -BA * S + AA;
    return wheel


Xint = np.linspace(-69,60,30)
Yint = [s1002(i) for i in Xint]
coeff = cubspline(Xint,Yint)

X = np.linspace(-69,60,200)
Y = cubsplineval(coeff, Xint, X)
Y1 = [s1002(i) for i in X]

plt.plot(X,Y,"b.",label="Cubic spline")
plt.plot(Xint,Yint,"ro",label="Interpolation points")
plt.plot(X,Y1,"g",label="s1002(x)")
plt.legend()
plt.show()



