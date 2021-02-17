import numpy as np
from scipy.special import expit as H
import matplotlib.pyplot as plt
from matplotlib import cm

γ_GFR = 1.  # rate of GFR reaching its steady state
γ_ERM = 1.  # rate of ERM reaching its steady state
o_GFR = -2.56  # Basal inhibition of GFR
o_GFR_E2ER = -1.6  # -1.6  # GFR inhibition by E2ER
o_GFR_GFR = 6.8  # GFR self activation
o_GFR_ERM = 0.4  # GFR activation by ERM
o_ERM = -3.04  # Basal inhibition of EPM
o_ERM_E2ER = -1.6  # -1.6  # EPM inhibition by E2ER
o_ERM_ERM = 6.64  # 6.64  # ERM self activation
o_ERM_GFR = 0.4  # ERM activation by GFR
o_E2ER = -5.0  # E2ER self activation
o_E2ER_E2 = 0.5  # E2ER activation by E2


def W_GFR(GFR, E2ER, ERM, o_GFR=o_GFR, o_GFR_E2ER=o_GFR_E2ER, o_GFR_GFR=o_GFR_GFR, o_GFR_ERM=o_GFR_ERM):
    return o_GFR + o_GFR_E2ER * E2ER + o_GFR_GFR * GFR + o_GFR_ERM * ERM


def W_ERM(GFR, E2ER, ERM, o_ERM=o_ERM, o_ERM_E2ER=o_ERM_E2ER, o_ERM_ERM=o_ERM_ERM, o_ERM_GFR=o_ERM_GFR):
    return o_ERM + o_ERM_E2ER * E2ER + o_ERM_ERM * ERM + o_ERM_GFR * GFR


def W_E2ER(E2, o_E2ER=o_E2ER, o_E2ER_E2=o_E2ER_E2):
    return o_E2ER + o_E2ER_E2 * E2


def dderivatives_grad(f, L, dr=1):
    grad_x = np.zeros(L)
    grad_y = np.zeros(L)
    for i in range(1, L[0] - 1):
        for j in range(1, L[1] - 1):
            if i==1:
                grad_x[i, j] = (f[i, j] - f[0, j]) / (dr)
            else:
                grad_x[i, j] = (f[i + 1, j] - f[i, j]) / ( dr)

            if j==1:
                grad_y[i, j] = (f[i, j] - f[i, 0]) / ( dr)
            else:
                grad_y[i, j] = (f[i, j + 1] - f[i, j]) / ( dr)

    return grad_x, grad_y


def dderivatives_div(fx, fy, L, dr=1):
    div = np.zeros(L)
    for i in range(1, L[0] - 1):
        for j in range(1, L[1] - 1):
            if i == 1 and j > 1:
                div[i, j] = (fx[i, j] - fx[0, j]) / ( dr) + (fy[i, j + 1] - fy[i, j]) / ( dr)
            elif i > 1 and j == 1:
                div[i, j] = (fx[i + 1, j] - fx[i, j]) / ( dr) + (fy[i, j] - fy[i, 0]) / ( dr)
            elif i == 1 and j == 1:
                div[i, j] = (fx[i, j] - fx[0, j]) / (dr) + (fy[i, j] - fy[i, 0]) / (dr)
            else:
                div[i, j] = (fx[i + 1, j] - fx[i, j]) / ( dr) + (fy[i, j + 1] - fy[i, j]) / ( dr)

    return div


def dderivatives_lapl(f, L, dr=1):
    lapl = np.zeros(L)
    for i in range(1, L[0] - 1):
        for j in range(1, L[1] - 1):
            lapl[i, j] = (f[i + 1, j] + f[i - 1, j] - 4 * f[i, j] + f[i, j + 1] + f[i, j - 1]) / (dr * dr)

    return lapl


def update_ghost_points(f, L, vx, vy, D=1, dr=1):
    #f[0, :] = f[1, :] * (1 + dr * vx[1, :] / D)
    #f[:, 0] = f[:, 1] * (1 + dr * vy[:, 1] / D)
    f[0, :] = f[1, :] * (D - vx[1, :]) * dr
    f[:, 0] = f[:, 1] * (D - vy[:, 1]) * dr
    f[-1, :] = f[-2, :] * (1 + dr * vx[-2, :] / D)
    f[:, -1] = f[:, -2] * (1 + dr * vy[:, -2] / D)



    return f


if __name__ == '__main__':
    import os

    NGRID = np.array([[-0.2, 1.2],
                      [-0.2, 1.2]])

    prefix_ = '001'
    os.system('mkdir '+str(prefix_))

    threshold = 10e-8
    E2 = 10e2
    dt = 10e-5
    dL = 1.0
    nstep = 40000
    L = np.array([53, 53])
    x1 = np.linspace(NGRID[0, 0], NGRID[0, 1], L[0])
    x2 = np.linspace(NGRID[1, 0], NGRID[1, 1], L[1])
    GFR, ERM = np.meshgrid(x1, x2)
    p = np.zeros(L)
    p[25, 25] = 1.0
    # p = np.exp(-((GFR - 0.5) ** 2 + (ERM - 0.5) ** 2))

    E2ER = H(W_E2ER(np.log10(E2)))

    F_GFR = γ_GFR * (H(W_GFR(GFR, E2ER, ERM)) - GFR)
    F_ERM = γ_ERM * (H(W_ERM(GFR, E2ER, ERM)) - ERM)
    #F_GFR = 0.1 * np.ones(GFR.shape)#
    #F_ERM = 0.1 * np.ones(GFR.shape)

    p = update_ghost_points(p, L, F_GFR, F_ERM)
    p_old = np.copy(p)

    F_div = dderivatives_div(F_GFR, F_ERM, L, dL)


    for istep in range(nstep):

        grad_p = dderivatives_grad(p, L, dL)

        advection = grad_p[0] * F_GFR + grad_p[1] * F_ERM + p * F_div
        diffusion = dderivatives_lapl(p, L, dL)

        p = p_old + dt * (-advection + diffusion)

        p_diff = np.max(np.abs(p_old[1:-2] - p[1:-2]))

        p = update_ghost_points(p, L, F_GFR, F_ERM)

        p_old = np.copy(p)

        if p_diff <= threshold:
            print(istep)
            np.savez(prefix_ + '/' + str(istep) + '.npz', GFR=GFR, ERM=ERM, prob=p)
            break

        if istep % 1000 == 0:
            print(istep, p[1:-2].sum())
            np.savez(prefix_ + '/' + str(istep) + '.npz', GFR=GFR, ERM=ERM, prob=p)
            #fig = plt.figure()
            #ax = plt.axes(projection='3d')
            #ax.set_xlabel('GFR')
            #ax.set_ylabel('ERM')
            #surf = ax.plot_surface(GFR[1:-2], ERM[1:-2], -np.log(p[1:-2]), cmap=cm.jet)
            #fig.colorbar(surf, shrink=0.5, aspect=5)
            #ax.view_init(30, 60)
            #plt.show()

