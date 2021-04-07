#!/usr/bin/env python3

import numpy as np
from sympy import *
from scipy.special import comb
from scipy.linalg import sqrtm
from itertools import combinations
from gPC_toolbox_v0.UnivariateGaussHermiteQuadrature import UnivariateGaussHermiteQuadrature
from gPC_toolbox_v0.GaussHermitePC import GaussHermitePC

from gPC_toolbox_v0.numerical_integration import integrate_gauss


def l2_cost_function_noinit(Q,n_states,n_uncert,p):
    """
    The expectation cost function is of the form
    E(int_{0}^{t} x^Top Q x dt) = int_{0}^{t} X^Top Q_det X dt
    """
    xi_symbols = [symbols('xi'+str(i)) for i in range(1,n_uncert-n_states+1)]
    xi =  Matrix(xi_symbols)
    if n_uncert-n_states == 1:
        xi = Matrix([symbols('xi')])
    
    Hp = Matrix(GaussHermitePC(n_uncert-n_states,p))
    Phi = Matrix(np.kron(np.identity(n_states),Hp.T))

    Q_dummy =  Phi.T*Q*Phi
    n_nodesQuadrature = 20
    [weight,node] = UnivariateGaussHermiteQuadrature(n_nodesQuadrature)
    
    # computing the Expectation with respect to the uncertain parameters
    for xi_i in xi:
        Q_dummy = integrate_gauss(weight,node,Q_dummy, xi_i)
    #
    Q_det = Q_dummy
    #
    return np.round(np.array(Q_det,dtype=float),5)


def l2_cost_function(Q,n_states,n_uncert,p):
    
    """
    The expectation cost function is of the form
    E(int_{0}^{t} x^Top Q x dt) = int_{0}^{t} X^Top Q_det X dt
    
    """
    xi_symbols = [symbols('xi'+str(i)) for i in range(1,n_uncert+1)]
    xi =  Matrix(xi_symbols)
    if n_uncert-n_states == 1:
        xi = Matrix([symbols('xi')])
    Hp = Matrix(GaussHermitePC(n_uncert,p))
    Phi = Matrix(np.kron(np.identity(n_states),Hp.T))

    Q_dummy =  Phi.T*Q*Phi
    n_nodesQuadrature = 20
    [weight,node] = UnivariateGaussHermiteQuadrature(n_nodesQuadrature)
    
    # computing the Expectation with respect to the uncertain parameters
    for xi_i in xi:
        Q_dummy = integrate_gauss(weight,node,Q_dummy, xi_i)
    #
    Q_det = Q_dummy
    #
    return np.round(np.array(Q_det,dtype=float),5)