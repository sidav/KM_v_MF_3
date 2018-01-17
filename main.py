import numpy as np
from scipy.integrate import odeint, simps
from scipy.misc import derivative
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
from scipy.special import erf

def convert_angstrom_to_atomic_units(value):
    return value / 0.53


def convert_electronvolt_to_atomic_units(value):
    return value / 27.212

class Shooting_method:
    # initial data (atomic units)
    L = convert_angstrom_to_atomic_units(2.0)
    A = -L
    B = +L

    def __init__(self, fun_U, U0, ne, e2, count_e, n):
        self.U = fun_U
        self.U0 = U0
        self.ne = ne
        self.e2 = e2
        self.n = n

        self.X = np.linspace(self.A, self.B, self.n)  # forward
        self.XX = np.linspace(self.B, self.A, self.n)  # backwards
        self.r = (self.n - 1) * 3 // 4  # forward
        self.rr = self.n - self.r - 1
        self.e1 = self.U0 + 0.05
        self.count_e = count_e


        # function (13)
    def q(self, e, x):
        return 2.0 * (e - self.U(x))

    def system1(self, cond1, X):
        Y0, Y1 = cond1[0], cond1[1]
        dY0dX = Y1
        dY1dX = - self.q(self.eee, X) * Y0
        return [dY0dX, dY1dX]

    def system2(self, cond2, XX):
        Z0, Z1 = cond2[0], cond2[1]
        dZ0dX = Z1
        dZ1dX = - self.q(self.eee, XX) * Z0
        return [dZ0dX, dZ1dX]

    def average_value(self, psi, oper_value):
        value = []
        for ind in range(len(psi)):
            value.append(psi[ind] * oper_value[ind])

        fun = interp1d(self.X, value, kind='cubic')
        result = quad(fun, self.A, self.B)
        return result

    # calculation of f (eq. 18; difference of derivatives)
    def f_fun(self, e):
        self.eee = e
        """
        Cauchy problem ("forward")
        dPsi1(x)/dx = - q(e, x)*Psi(x);
        dPsi(x)/dx = Psi1(x);
        Psi(A) = 0.0
        Psi1(A)= 1.0
        """
        cond1 = [0.0, 1.0]
        sol1 = odeint(self.system1, cond1, self.X)
        self.Psi = sol1[:, 0]
        """
        Cauchy problem ("backwards")
        dPsi1(x)/dx = - q(e, x)*Psi(x);
        dPsi(x)/dx = Psi1(x);
        Psi(B) = 0.0
        Psi1(B)= 1.0
        """
        cond2 = [0.0, 1.0]
        sol2 = odeint(self.system2, cond2, self.XX)
        self.Fi = sol2[:, 0]
        # search of maximum value of Psi
        p1 = np.abs(self.Psi).max()
        p2 = np.abs(self.Psi).min()
        big = p1 if p1 > p2 else p2
        # scaling of Psi
        self.Psi[:] = self.Psi[:] / big
        # mathematical scaling of Fi for F[rr]=Psi[r]
        coef = self.Psi[self.r] / self.Fi[self.rr]
        self.Fi[:] = coef * self.Fi[:]
        # calculation of f(E) in node of sewing
        curve1 = interp1d(self.X, self.Psi, kind='cubic')
        curve2 = interp1d(self.XX, self.Fi, kind='cubic')
        der1 = derivative(curve1, self.X[self.r], dx=1.e-6)
        der2 = derivative(curve2, self.XX[self.rr], dx=1.e-6)
        f = der1 - der2
        return f

    def bisection_method(self, x1, x2, tol):
        while abs(x2 - x1) > tol:
            xr = (x1 + x2) / 2.0
            if self.f_fun(e=x2) * self.f_fun(e=xr) < 0.0:
                x1 = xr
            else:
                x2 = xr
            if self.f_fun(e=x1) * self.f_fun(e=xr) < 0.0:
                x2 = xr
            else:
                x1 = xr
        return (x1 + x2) / 2.0

    def get_energy(self):
        ee = np.linspace(self.e1, self.e2, self.ne)
        af = np.zeros(self.ne, dtype=float)
        porog = 5.0
        tol = 1.0e-7
        energy = []
        fun_psi = []
        ngr = 0
        for i in np.arange(self.ne):
            e = ee[i]
            af[i] = self.f_fun(e)
            if i > 0:
                Log1 = af[i] * af[i - 1] < 0.0
                Log2 = np.abs(af[i] - af[i - 1]) < porog
                if Log1 and Log2:
                    energy1 = ee[i - 1]
                    energy2 = ee[i]
                    eval = self.bisection_method(energy1, energy2, tol)
                    energy.append(eval)
                    coefPsi = self.average_value(self.Psi, self.Psi)
                    self.Psi[:] = self.Psi[:] / math.sqrt(coefPsi[0])
                    normPsi = self.average_value(self.Psi, self.Psi)
                    if (normPsi[0] - 1 > 0.000001):
                        print("Error! integrate Psi = ", normPsi[0])
                        return None
                    fun_psi.append(self.Psi)
                    ngr += 1
                    if ngr == self.count_e:
                        break
        return energy, fun_psi
#----------------- DON'T TOUCH ANYTHING ABOVE! -----------------------------


#------------------data initial---------------------------------------------
V0 = convert_electronvolt_to_atomic_units(20)
L = convert_angstrom_to_atomic_units(2.0)
W = 4.0
A = -L
B = +L
n = 257
X = np.linspace(A, B, n)  # forward

count_phi = 100
N1 = 5
N2 = 8
N3 = 15
phi_values = []
Temp = []


#--------------------   Func defs   ----------------------------------------

def fun_U(x): #potential function
    if (abs(x) < L):
        return float((-1 + (x + L) / (2 * L)) if abs(x) < L else W)
    else:
        return W

def mean(psi_m, oper_value, psi_n):
    value = []
    if oper_value is None:
        for ind in range(len(psi_m)):
            value.append(psi_m[ind] * psi_n[ind])
    else:
        for ind in range(len(psi_m)):
            value.append(psi_m[ind] * oper_value[ind] * psi_n[ind])

    result = simps(value, X)
    return result

def calcT(funF, X):
    localX = [X[0] - 4 * 1.e-6]
    localX.append(X[0] - 2 * 1.e-6)
    for x in X:
        localX.append(x)
    localX.append(X[len(X) - 1] + 2 * 1.e-6)
    localX.append(X[len(X) - 1] + 4 * 1.e-6)

    der = []
    for x in X:
        der.append(-1/2 * derivative(funF, x, dx=1.e-6, n=2, order=5))
    return der

#target of functions in point k
def get_phi_k_values(k):
    if k % 2 == 0:
        fun = lambda x: 1 / math.sqrt(L) * math.sin(math.pi * k * x / (2 * L))
    else:
        fun = lambda x: 1 / math.sqrt(L) * math.cos(math.pi * k * x / (2 * L))

    result = []
    for x in X:
        result.append(fun(x))

    return result

#functions

def get_phi_k(k):
    if k % 2 == 0:
        return lambda x: 1 / math.sqrt(L) * math.sin(math.pi * k * x / (2 * L))
    else:
        return lambda x: 1 / math.sqrt(L) * math.cos(math.pi * k * x / (2 * L))

def get_matrix_H(N):
    value_U = np.array([fun_U(X[i]) for i in np.arange(n)])
    matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            new_value = 0
            if i == j:
                new_value += math.pow(math.pi * (i + 1) / L, 2) / 8
            new_value += mean(phi_values[i], value_U, phi_values[j])
            matrix[i][j] = new_value
    # print_beautiful_matrix(matrix)
    return matrix

def get_psi(c):
    result = []
    for i in range(len(X)):
        value = 0
        for j in range(len(c)):
            value += c[j] * phi_values[j][i]
        result.append(value)

    coefPsi = mean(result, None, result)
    for i in range(len(result)):
        result[i] /= math.sqrt(coefPsi)
    return result

def plot(U, psi, psi1, psi2, psi3):
    offset = 0.2
    plt.axis([A-offset, B+offset, -1-offset, W-offset])
    plt.plot(X, U, 'g-', linewidth=5.0, label="U(x)")
    Zero = np.zeros(n, dtype=float)
    plt.plot(X, Zero, 'k-', linewidth=1.0)  # abscissa axis
    plt.plot(X, psi, color=(0.9, 0.0, 0.0), linewidth=10.0, label="$\psi$")
    plt.plot(X, psi1, color=(0.0, 0.9, 0.0), linewidth=7.0, label="$\psi$_1")
    plt.plot(X, psi2, color=(0.0, 0.0, 0.9), linewidth=3.5, label="$\psi$_2")
    plt.plot(X, psi3, color=(0.5, 0.5, 0.5), linewidth=1.0, label="$\psi$_3")
    plt.xlabel("X", fontsize=18, color="k")
    plt.ylabel("U(x), $\psi$(x)", fontsize=18, color="k")
    plt.grid(True)
    plt.legend(fontsize=16, shadow=True, fancybox=True, loc='upper right')
    plt.show()

###################################

def print_beautiful_matrix(matr): ### <-- DEL ME
    print("Printing matrix of size {:d}".format(len(matr)))
    for i in range(len(matr)):
        for j in range(len(matr[0])):
            print("{:12.5f}".format(matr[i][j]), end="")
        print("")

def normaliz_condition(vect):
    epsylon = 0.0000000000001
    for i in range(len(vect)):
        for j in range(len(vect)):
            curr_dot = np.dot(vect[i], vect[j])
            if i == j and abs(curr_dot - 1.0) > epsylon:
                print("ERROR EQ: {:d} product on itself is {:12.20f} ".format(i, curr_dot))
            elif i != j and abs(curr_dot) > epsylon:
                print("ERROR NEQ: {:d} and {:d} product is {:12.20f} ".format(i, j, curr_dot))

def comp_psi(curr_N):
    H = get_matrix_H(curr_N)  # calc matrix H for N1 count

    e, c = np.linalg.eig(H)

    # print_beautiful_matrix(c)
    normaliz_condition(c)
    # print(np.dot(c[0], c[2]))
    # print("above")

    E0 = e.min()
    for i in range(len(e)):
        if e[i] == E0:
            min_ind = i
            break

    coef_c = []
    for i in range(len(c)):
        coef_c.append(c[i][min_ind])
    if (coef_c[0] < 0):
        coef_c = np.dot(coef_c, -1)

    curr_psi = get_psi(coef_c)  # calc psi1
    return E0, curr_psi

####################################
# -------------  main  ----------- #

for i in range(count_phi):
    phi_values.append(get_phi_k_values(i + 1))

E0, psi1 = comp_psi(N1)
E02, psi2 = comp_psi(N2)
E03, psi3 = comp_psi(N3)

shooting_method_U = Shooting_method(fun_U, U0=-0.99999, ne=101, e2=15, count_e=1, n=n)
energy_U, psi_U = shooting_method_U.get_energy()
SM_energy = energy_U[0]

#---------write results--------------------
print("E0 (shooting_method) = {:12.8f}".format(SM_energy))
print("Using variational method: ")
print("E0 (for N = {:d} )      = {:12.8f}, diff {:12.10f}".format(N1, E0, E0 - SM_energy))
print("E0 (for N = {:d} )      = {:12.8f}, diff {:12.10f}".format(N2, E02, E02 - SM_energy))
print("E0 (for N = {:d} )     = {:12.8f}, diff {:12.10f}".format(N3, E03, E03 - SM_energy))

#----------plot graph----------------------
value_U = np.array([fun_U(X[i]) for i in np.arange(n)])
plot(value_U, psi_U[0], psi1, psi2, psi3)

