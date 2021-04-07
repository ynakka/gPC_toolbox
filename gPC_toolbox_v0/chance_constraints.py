#!/usr/bin/env python3

import numpy as np
from sympy import *
from scipy.special import comb
from scipy.linalg import sqrtm
from itertools import combinations


from gPC_toolbox_v0.UnivariateGaussHermiteQuadrature import UnivariateGaussHermiteQuadrature
from gPC_toolbox_v0.GaussHermitePC import GaussHermitePC
from cloudpickle import dump, load
from gPC_toolbox_v0.numerical_integration import integrate_gauss


def obstacle_variance_gpc_assignment(obstacle_var,num_states,num_uncert,num_gpcstates):
    
    obstacle_gpc = np.zeros((num_gpcstates))

    for j in range(num_states): # num_states
        obstacle_gpc[int(j*(num_gpcstates/num_states)+1)] = np.sqrt(obstacle_var[j]) 
    
    return obstacle_gpc

def linear_chance_constraint_matrices_noinit(n_states,n_uncert,p):
    
    xi_symbols = [symbols('xi'+str(i)) for i in range(1,n_uncert-n_states+1)]
    xi =  Matrix(xi_symbols)
    if n_uncert-n_states == 1:
        xi = Matrix([symbols('xi')])
    Hp = Matrix(GaussHermitePC(n_uncert-n_states,p))

    l = Hp.shape[0]
    
    M = zeros(1,l)
    M[0] = 1

    H_dummy = zeros(l-1,1)
    for i in range(1,l):
        H_dummy[i-1] = Hp[i]
    HH_dummy = H_dummy*H_dummy.T
    n_nodesQuadrature = 20
    [weight,node] = UnivariateGaussHermiteQuadrature(n_nodesQuadrature)
    
    # computing the Expectation with respect to the uncertain parameters
    for xi_i in xi:
        HH_dummy = integrate_gauss(weight,node,HH_dummy, xi_i)
    #
    EHH = sqrtm(np.array(HH_dummy,float))
    H = zeros(l,l)

    for i in range(1,l):
        for j in range(1,l):
            H[i,j] = EHH[i-1,j-1]

    N = np.kron(np.ones((n_states,n_states)),H)

    return M, N

def linear_chance_constraint_noinit(a,M,N,risk,num_gpcpoly,n_states,n_uncert,p):
    """
    Pr{a^\Top x + b \leq 0} \geq 1-eps
    Converts to SOCP
    """
    a_hat = np.kron(a.T,M)
    a_dummy = np.zeros((n_states,n_states))
    
    for ii in range(n_states):
        a_dummy[ii,ii] = a[ii,0]
    #print(a_dummy)
    U = np.kron(a_dummy,np.identity(num_gpcpoly))
    # Sigma_det = U*N*N.T*U.T
    Sigma_det = N.T*U.T

    return np.reshape(np.round(np.array(a_hat,dtype=float),5),num_gpcpoly*n_states), np.round(np.array(Sigma_det,dtype=float),5)



def quadratic_chance_constraint_noinit(A,n_states,n_uncert,p):
    """
    Pr{x^Top A x geq c} \leq eps
    Converts to SOCP
    """
    xi_symbols = [symbols('xi'+str(i)) for i in range(1,n_uncert-n_states+1)]
    xi =  Matrix(xi_symbols)
    if n_uncert-n_states == 1:
        xi = Matrix([symbols('xi')])
    Hp = Matrix(GaussHermitePC(n_uncert-n_states,p))

    l = Hp.shape[0]

    H_dummy = zeros(l,1)
    for i in range(1,l):
        H_dummy[i] = Hp[i]
    HH_dummy = H_dummy*H_dummy.T

    n_nodesQuadrature = 20
    [weight,node] = UnivariateGaussHermiteQuadrature(n_nodesQuadrature)
    
    # computing the Expectation with respect to the uncertain parameters
    for xi_i in xi:
        HH_dummy = integrate_gauss(weight,node,HH_dummy, xi_i)
    
    A_det = np.kron(A,HH_dummy)

    return np.round(np.array(A_det,dtype=float),5)



def linear_chance_constraint_matrices(n_states,n_uncert,p):
    
    xi_symbols = [symbols('xi'+str(i)) for i in range(1,n_uncert+1)]
    xi =  Matrix(xi_symbols)
    if n_uncert-n_states == 1:
        xi = Matrix([symbols('xi')])
    Hp = Matrix(GaussHermitePC(n_uncert,p))

    l = Hp.shape[0]
    
    M = zeros(1,l)
    M[0] = 1

    H_dummy = zeros(l-1,1)
    for i in range(1,l):
        H_dummy[i-1] = Hp[i]
    HH_dummy = H_dummy*H_dummy.T
    n_nodesQuadrature = 20
    [weight,node] = UnivariateGaussHermiteQuadrature(n_nodesQuadrature)
    
    # computing the Expectation with respect to the uncertain parameters
    for xi_i in xi:
        HH_dummy = integrate_gauss(weight,node,HH_dummy, xi_i)
    #

    EHH = sqrtm(np.array(HH_dummy,float))

    H = zeros(l,l)

    for i in range(1,l):
        for j in range(1,l):
            H[i,j] = EHH[i-1,j-1]

    N = np.kron(np.ones((n_states,n_states)),H)

    return M, N

def linear_chance_constraint(a,M,N,risk,num_gpcpoly,n_states,n_uncert,p):
    """
    Pr{a^\Top x + b \leq 0} \geq 1-eps
    Converts to SOCP
    """
    a_hat = np.kron(a.T,M)

    a_dummy = np.zeros((n_states,n_states))
    
    for ii in range(n_states):
        a_dummy[ii,ii] = a[ii,0]
    #print(a_dummy)
    U = np.kron(a_dummy,np.identity(num_gpcpoly))

    # Sigma_det = U*N*N.T*U.T
    Sigma_det = np.sqrt((1-risk)/risk)* N.T*U.T

    return np.reshape(np.round(np.array(a_hat,dtype=float),5),num_gpcpoly*n_states), np.round(np.array(Sigma_det,dtype=float),5)



def quadratic_chance_constraint(A,n_states,n_uncert,p):
    """
    Pr{x^Top A x geq c} \leq eps
    Converts to SOCP
    """
    xi_symbols = [symbols('xi'+str(i)) for i in range(1,n_uncert+1)]
    xi =  Matrix(xi_symbols)
    Hp = Matrix(GaussHermitePC(n_uncert,p))

    l = Hp.shape[0]

    H_dummy = zeros(l,1)
    for i in range(1,l):
        H_dummy[i] = Hp[i]
    HH_dummy = H_dummy*H_dummy.T

    n_nodesQuadrature = 20
    [weight,node] = UnivariateGaussHermiteQuadrature(n_nodesQuadrature)
    
    # computing the Expectation with respect to the uncertain parameters
    for xi_i in xi:
        HH_dummy = integrate_gauss(weight,node,HH_dummy, xi_i)
    
    A_det = np.kron(A,HH_dummy)

    return np.round(np.array(A_det,dtype=float),5)

    