#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sunday, 07 Feb 2021
@author: moreira
"""

import numpy as np
from scipy.special import expit as H
from scipy.ndimage import convolve
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

kwargs = {'origin': 'lower',
          'interpolation': 'sinc',
          'cmap': 'plasma'}

γ_GFR = 1.0  # rate of GFR reaching its steady state
γ_ERM = 1.0  # rate of ERM reaching its steady state
o_GFR = -2.56  # Basal inhibition of GFR
o_GFR_E2ER = -1.6  # GFR inhibition by E2ER
o_GFR_GFR = 6.8  # GFR self activation
o_GFR_ERM = 0.4  # GFR activation by ERM
o_ERM = -3.04  # Basal inhibition of EPM
o_ERM_E2ER = -1.6  # EPM inhibition by E2ER
o_ERM_ERM = 6.64  # ERM self activation
o_ERM_GFR = 0.4  # ERM activation by GFR
o_E2ER = -5.0  # Basal inhibition E2ER
o_E2ER_E2 = 0.5  # E2ER activation by E2
#E2 = 1.4
#E2 = np.log10(E2)  # 2 (high), 0 (low), -2 (trace)
D = 0.005  # stochasticity


def W_GFR(GFR, E2ER, ERM, o_GFR = o_GFR, o_GFR_E2ER = o_GFR_E2ER, o_GFR_GFR = o_GFR_GFR, o_GFR_ERM = o_GFR_ERM):
    return o_GFR + o_GFR_E2ER * E2ER + o_GFR_GFR * GFR + o_GFR_ERM * ERM


def W_ERM(GFR, E2ER, ERM, o_ERM = o_ERM, o_ERM_E2ER = o_ERM_E2ER,  o_ERM_ERM = o_ERM_ERM, o_ERM_GFR = o_ERM_GFR):
    return o_ERM + o_ERM_E2ER * E2ER + o_ERM_ERM * ERM + o_ERM_GFR * GFR


def W_E2ER(E2, o_E2ER = o_E2ER, o_E2ER_E2 = o_E2ER_E2):
    return o_E2ER + o_E2ER_E2 * E2



def model(nstep, E2, T, D, dt):


    GFR = np.zeros(nstep)
    ERM = np.zeros(nstep)
    E2ER = np.zeros(nstep)
    E2ER[0] = H(W_E2ER(E2))



    for istep in range(1, nstep):
        E2t = np.sign(np.sin(-np.pi / T + istep * np.pi / T)) * E2

        GFR[istep] = GFR[istep-1] + dt * (γ_GFR * (H(W_GFR(GFR[istep-1], E2ER[istep-1], ERM[istep-1]))
                                                   - GFR[istep-1]) + D * (np.random.rand() - 0.5))
        ERM[istep] = ERM[istep-1] + dt * (γ_ERM * (H(W_ERM(GFR[istep-1], E2ER[istep-1], ERM[istep-1]))
                                                   - ERM[istep-1]) + D * (np.random.rand() - 0.5))
        E2ER[istep] = H(W_E2ER(E2t))

    fig, ax = plt.subplots()
    #ax = plt.axes(projection='3d')
    ax.plot(GFR[0:], label='GFR', color='black')
    ax.plot(ERM[0:], label='ERM', color='blue')
    #ax.plot(E2ER[0:], label='E2ER', color='red')
    ax.set_xlabel('Time')
    #ax.set_yscale('log')
    plt.ylim(0,1.1)
    plt.legend()
    plt.show()

    return GFR, ERM, E2ER

if __name__ == '__main__':


    import matplotlib.pyplot as plt
    from matplotlib import cm



    E2 = 10 #10e-17
    dt = 1.


    GFR, ERM, E2ER = model(100, E2, 10, 0.005, dt)

