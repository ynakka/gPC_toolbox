#!/usr/bin/env python3

import numpy as np
from sympy import *
from scipy.special import comb
from scipy.linalg import sqrtm
from itertools import combinations
from gPC_toolbox_v0.UnivariateGaussHermiteQuadrature import UnivariateGaussHermiteQuadrature
from gPC_toolbox_v0.GaussHermitePC import GaussHermitePC
from cloudpickle import dump, load

from gPC_toolbox_v0.density import gaussian_density
from gPC_toolbox_v0.numerical_integration import integrate_gauss



def covar_mat_noinit(n_states,n_uncert,p):
    
    # x and y are in gpc cordinates

    xi_symbols = [symbols('xi'+str(i)) for i in range(1,n_uncert-n_states+1)]
    xi =  Matrix(xi_symbols)
    if n_uncert-n_states ==1 :
        xi = Matrix([symbols('xi')])
    Hp = Matrix(GaussHermitePC(n_uncert-n_states,p))
    
    l = Hp.shape[0]
    Hp[0,0] = 0

    var = Hp*Hp.T 
    n_nodesQuadrature = 20
    [weight,node] = UnivariateGaussHermiteQuadrature(n_nodesQuadrature)
  
    for xi_i in xi:
        var =  integrate_gauss(weight,node,var, xi_i) #integrate(gaussian_density(xi_i)*var, (xi_i,-oo,oo))
    
    return np.round(np.array(var,dtype=float),5)

# x is an individual state

def meangpc_noinit(x,num_gpcpoly):
    
    # x is in gpc cordinates

    l = num_gpcpoly
    X = x.reshape((l,1))
    
    mean = X[0,0]
    return np.round(np.array(mean,dtype=float),5)



# x is full gpcstate

def meangpc_fullstate_noinit(x,n_states,num_gpcpoly):    
    # x is in gpc cordinates
    mean = np.zeros((n_states))
    for i in range(n_states):
        mean[i] = x[i*(num_gpcpoly)]
    
    return np.round(np.array(mean,dtype=float),5)

# x is an individual state 
def vargpc_noinit(x,n_states,n_uncert,p):
    
    xi_symbols = [symbols('xi'+str(i)) for i in range(1,n_uncert-n_states+1)]
    xi =  Matrix(xi_symbols)
    if n_uncert-n_states ==1 :
        xi = Matrix([symbols('xi')])
    Hp = Matrix(GaussHermitePC(n_uncert-n_states,p))

    l = Hp.shape[0]
    n_nodesQuadrature = 20
    [weight,node] = UnivariateGaussHermiteQuadrature(n_nodesQuadrature)

    X = x.reshape((l,1))
    X[0,0] = 0.0

    vec_dummy = zeros(1,1)
    vec_dummy[0,0] = X.T*Hp
    
    var = vec_dummy*vec_dummy 

    for xi_i in xi:
        var = integrate_gauss(weight,node,var, xi_i)
    
    return np.round(np.array(var,dtype=float),5)




#------------------------------------------------------------------------


## following functions are used for projecting back 

# return cross covariance matrix between x and y

def covar_mat(n_states,n_uncert,p):
    
    # x and y are in gpc cordinates

    xi_symbols = [symbols('xi'+str(i)) for i in range(1,n_uncert+1)]
    xi =  Matrix(xi_symbols)
    if n_uncert-n_states ==1 :
        xi = Matrix([symbols('xi')])
    Hp = Matrix(GaussHermitePC(n_uncert,p))

    l = Hp.shape[0]
    Hp[0,0] = 0
    # X = x.reshape((l,1))
    # X[0,0] = 0.0
    # #print(X.shape)
    # print(X)
    # #print(Hp.shape)
    # #print(Hp)
    
    # Y = y.reshape((l,1))
    # Y[0,0]= 0.0
    # print(Y)

    # vec_dummy = zeros(1,2)
    # vec_dummy[0,0] = X.T*Hp
    # vec_dummy[0,1] = Y.T*Hp
    #print(vec_dummy)

    var = Hp*Hp.T 
    for xi_i in xi:
        var =  integrate(gaussian_density(xi_i)*var, (xi_i,-oo,oo))
    
    return np.round(np.array(var,dtype=float),5)

# x is an individual state
def meangpc(x,num_gpc):
    mean = x[0]
    return np.round(np.array(mean,dtype=float),5)



# x is full gpcstate

def meangpc_fullstate(x,n_states,num_gpcpoly):
    # x is in gpc cordinates
    mean = np.zeros((n_states))
    for i in range(n_states):
        mean[i] = x[i*num_gpcpoly]

    return np.round(np.array(mean,dtype=float),5)

# x is an individual state 
def vargpc(x,n_states,n_uncert,p):
    
    xi_symbols = [symbols('xi'+str(i)) for i in range(1,n_uncert+1)]
    xi =  Matrix(xi_symbols)
    Hp = Matrix(GaussHermitePC(n_uncert,p))
    if n_uncert-n_states ==1 :
        xi = Matrix([symbols('xi')])
    l = Hp.shape[0]
    n_nodesQuadrature = 20
    [weight,node] = UnivariateGaussHermiteQuadrature(n_nodesQuadrature)

    X = x.reshape((l,1))
    X[0,0] = 0.0


    vec_dummy = zeros(1,1)
    vec_dummy[0,0] = X.T*Hp
    
    var = vec_dummy*vec_dummy 

    for xi_i in xi:
        var = integrate_gauss(weight,node,var, xi_i)
    
    return np.round(np.array(var,dtype=float),5)