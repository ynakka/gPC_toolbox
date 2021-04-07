#!/usr/bin/env python3

import numpy as np
from sympy import *
from scipy.special import comb
from scipy.linalg import sqrtm
from itertools import combinations
from gPC_toolbox_v0.UnivariateGaussHermiteQuadrature import UnivariateGaussHermiteQuadrature
from gPC_toolbox_v0.GaussHermitePC import GaussHermitePC
from gPC_toolbox_v0.numerical_integration import integrate_gauss


def initial_conditions_noinit(x,num_gpcpoly):
    num_states = x.shape[0]
    num_gpcstates = int(num_states*num_gpcpoly)
    Xinit = np.zeros(num_gpcstates)    
    # n_uncert-n_states,n_uncert
    for i in range(num_states):
        Xinit[int(i*num_gpcpoly)] = x[i]
    return np.round(np.array(Xinit,dtype=float),5)


def initial_conditions(x, x_mean,x_var, num_uncertainty,p):
    n_states = int(x.shape[0])
    n_uncert = int(num_uncertainty)

    # n_uncert-n_states,n_uncert
    xi_symbols = [symbols('xi'+str(i)) for i in range(1,n_uncert+1)]
    xi =  Matrix(xi_symbols)
    if n_uncert-n_states == 1:
        xi = Matrix([symbols('xi')])
    print(xi)   
    x_subs= zeros(n_states,1)
    print(x_subs)
    for i in range(n_states):
        x_subs[i,0] = x_mean[i] + np.sqrt(x_var[i])*xi[i+int(n_uncert-n_states)]
    print(x_subs)
    #rescaled Hermite polynomials
    Hp = GaussHermitePC(n_uncert,p)

    #print(Hp)
    # Kronecker Product

    Phi = Matrix(np.kron(np.identity(n_states),Hp.T))

    left = Phi.T*Phi
    right = Phi.T*x_subs
    
    n_nodesQuadrature = 20
    [weight,node] = UnivariateGaussHermiteQuadrature(n_nodesQuadrature)
    left_int = left
    right_int = right

    for xi_i in xi:
        left_int = integrate_gauss(weight,node,left_int, xi_i)
        right_int = integrate_gauss(weight,node,right_int, xi_i)
    

    X_init = np.asarray(left_int.inv()*right_int)

    return np.round(np.array(X_init,dtype=float),5)