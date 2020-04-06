
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#SIR model
def sir(y,t,alpha,beta):
    S,I,R = y
    dS = -alpha*S*I
    dI = alpha*S*I - beta*I
    dR = beta*I
    return [dS,dI,dR]

def get_sir(t,y0,alpha,beta):
    sol_sir = odeint(sir, y0, t, args=(alpha,beta))
    return sol_sir

def get_sir_confirmed(t,y0,alpha,beta):
    S,I,R = get_sir(t,y0,alpha,beta)
    return I

#SIR-X model
def sirx(y,t,alpha,beta,k0,k):
    S,I,R,X = y
    dS = -alpha*S*I         -k0*S
    dI = alpha*S*I - beta*I -k0*I-k*I
    dR = beta*X    + beta*I             +k0*S
    dX = (k0+k)*I  - beta*X
    return [dS,dI,dR,dX]

def get_sirx(t,y0,alpha,beta,k,k0):
    sol_sirx = odeint(sirx, y0, t, args=(alpha,beta,k,k0))
    return sol_sirx

def get_sirx_confirmed(t,y0,alpha,beta,k,k0):
    S,I,R,X = get_sirx(t,y0,alpha,beta,k,k0)
    return I

if __name__ == "__main__":
    beta  = 0.38
    r0 = 3.07
    k0 = 0.05
    k  = 0.05

    #initial conditions
    S0 = 1e6
    I0 = 1.0
    R0 = 0.0
    X0 = 0.0

    N = S0+I0+R0+X0
    alpha = r0*beta/N

    t = np.linspace(0, 100, 1000)
    sol_sir  = get_sir(t,alpha,beta)
    sol_sirx = get_sirx(t,alpha,beta,k,k0)

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.plot(t, sol_sir[:, 0], c='b', label='S')
    ax.plot(t, sol_sir[:, 1], c='g', label='I')
    ax.plot(t, sol_sir[:, 2], c='r', label='R')
    ax.set_yscale('log')
    ax.set_ylim(1,None)
    ax.grid()
    ax.legend(loc='best')

    ax = fig.add_subplot(2,1,2)
    ax.plot(t, sol_sirx[:, 0], c='b', label='S')
    ax.plot(t, sol_sirx[:, 1], c='g', label='I')
    ax.plot(t, sol_sirx[:, 2], c='r', label='R')
    ax.plot(t, sol_sirx[:, 3], c='y', label='X')
    ax.set_yscale('log')
    ax.grid()
    ax.legend(loc='best')
    ax.set_xlabel('t')
    ax.set_ylim(1,None)

    plt.show()
