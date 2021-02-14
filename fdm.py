import numpy as np
from scipy.special import expit as H
import matplotlib.pyplot as plt
from matplotlib import cm


γ_GFR = 1.0  # rate of GFR reaching its steady state
γ_ERM = 1.0  # rate of ERM reaching its steady state
o_GFR = -2.56  # Basal inhibition of GFR
o_GFR_E2ER = -1.6 #-1.6  # GFR inhibition by E2ER
o_GFR_GFR = 6.8  # GFR self activation
o_GFR_ERM = 0.4  # GFR activation by ERM
o_ERM = -3.04  # Basal inhibition of EPM
o_ERM_E2ER = -1.6 #-1.6  # EPM inhibition by E2ER
o_ERM_ERM = 6.64 #6.64  # ERM self activation
o_ERM_GFR = 0.4  # ERM activation by GFR
o_E2ER = 5.0  # E2ER self activation
o_E2ER_E2 = 0.5  # E2ER activation by E2

def W_GFR(GFR, E2ER, ERM, o_GFR = o_GFR, o_GFR_E2ER = o_GFR_E2ER, o_GFR_GFR = o_GFR_GFR, o_GFR_ERM = o_GFR_ERM):
    return o_GFR + o_GFR_E2ER * E2ER + o_GFR_GFR * GFR + o_GFR_ERM * ERM


def W_ERM(GFR, E2ER, ERM, o_ERM = o_ERM, o_ERM_E2ER = o_ERM_E2ER,  o_ERM_ERM = o_ERM_ERM, o_ERM_GFR = o_ERM_GFR):
    return o_ERM + o_ERM_E2ER * E2ER + o_ERM_ERM * ERM + o_ERM_GFR * GFR


def W_E2ER(E2, o_E2ER = o_E2ER, o_E2ER_E2 = o_E2ER_E2):
    return o_E2ER + o_E2ER_E2 * E2

def dderivatives_grad(f, L, dr=1):
    grad_x = np.zeros(L)
    grad_y = np.zeros(L)
    for i in range(1, L[0]-1):
        for j in range(1, L[1]-1):
            grad_x[i, j] = (f[i + 1, j] - f[i, j]) / (2*dr)
            grad_y[i, j] = (f[i, j+1] - f[i, j]) / (2*dr)

    return grad_x, grad_y


def dderivatives_div(fx, fy, L, dr=1):
    div = np.zeros(L)
    for i in range(1, L[0] - 1):
        for j in range(1, L[1] - 1):
            div[i, j] = (fx[i + 1, j] - fx[i, j]) / (2 * dr) + (fy[i, j + 1] - fy[i, j]) / (2 * dr)

    return div

def dderivatives_lapl(f, L, dr=1):
    lapl = np.zeros(L)
    for i in range(1, L[0]-1):
        for j in range(1, L[1]-1):
            lapl[i, j] = (f[i + 1, j] + f[i - 1, j] - 4 * f[i, j] + f[i, j + 1] + f[i, j - 1])/(dr*dr)

    return lapl


def update_ghost_points(f, L, vx, vy, D=1, dr=1):

    for n in range(1, L[0]-1):
        f[0, n] = f[1, n] * (1 + dr * vx[1, n] / D)
        f[n, 0] = f[n, 1] * (1 + dr * vy[n, 1] / D)

        f[L[0]-1, n] = f[L[0]-2, n] * (1 + dr * vx[L[0]-2, n] / D)
        f[n, L[1]-1] = f[n, L[1]-2] * (1 + dr * vy[n, L[1]-2] / D)

    return f

NGRID = np.array([[-0.2, 1.2],
                [-0.2, 1.2]])



E2 = 10e-2
dt = 10e-5
dL = 0.5
nstep=20000
L = np.array([51, 51])
x1 = np.linspace(NGRID[0, 0], NGRID[0, 1], L[0])
x2 = np.linspace(NGRID[1, 0], NGRID[1, 1], L[1])
GFR, ERM = np.meshgrid(x1, x2)
p = np.zeros(L)
p[25,25] = 100.
#p = np.exp(-((GFR - 0.5) ** 2 + (ERM - 0.5) ** 2))

E2ER = H(W_E2ER(np.log10(E2)))
F_GFR = γ_GFR * (H(W_GFR(GFR, E2ER, ERM)) - GFR)
F_ERM = γ_ERM * (H(W_ERM(GFR, E2ER, ERM)) - ERM)

p = update_ghost_points(p, L, F_GFR, F_ERM)
F_div = np.gradient(F_GFR, dL, axis=0) + np.gradient(F_ERM, dL, axis=1)#dderivatives_div(F_GFR, F_ERM, L, dL)

for istep in range(nstep):

    grad_p = dderivatives_grad(p, L, dL)

    advection = grad_p[0]*F_GFR + grad_p[1]*F_ERM + p*F_div 
    diffusion = dderivatives_lapl(p, L, dL)

    p = p + dt * ( -advection + diffusion)

    p = update_ghost_points(p, L, F_GFR, F_ERM)

    if istep%1000==0:
        print(istep)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(GFR[1:49], ERM[1:49], -np.log(p[1:49]), cmap=cm.jet)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()



