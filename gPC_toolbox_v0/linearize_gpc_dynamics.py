#!/usr/bin/env python3

import numpy as np
from sympy import *
from scipy.special import comb
from scipy.linalg import sqrtm
from itertools import combinations
from gPC_toolbox_v0.UnivariateGaussHermiteQuadrature import UnivariateGaussHermiteQuadrature
from gPC_toolbox_v0.GaussHermitePC import GaussHermitePC


from sympy.printing.theanocode import theano_function

def linearize_gpc_dynamics_noinit_theano(dynamics,x,u,n_uncert,p,input_states):
    n_states = x.shape[0]

    Hp = GaussHermitePC(n_uncert-n_states,p)    
    l = Hp.shape[0]
    
    xc_vals = zeros(l,n_states)
    for i in range(0,n_states):
        xc_vals[:,i] = Matrix([symbols('x'+str(i)+str(j)) for j in range(0,l)])

    states = xc_vals.T.reshape(n_states*l,1)

    dynH = Matrix(dynamics)
    Aj = dynH.jacobian(states)
    Bj = dynH.jacobian(u)
    Cj = dynH - Aj*states - Bj*u
    
    A = theano_function(input_states,Aj,on_unused_input='ignore')
    B = theano_function(input_states,Bj,on_unused_input='ignore')
    C = theano_function(input_states,Cj,on_unused_input='ignore')

    return A, B, C #theano functions for linearization

def linearize_gpc_dynamics_noinit(dynamics, x, u, n_uncert, p,input_states):
    
    n_states = x.shape[0]

    Hp = GaussHermitePC(n_uncert-n_states,p)
    
    l = Hp.shape[0]
    
    xc_vals = zeros(l,n_states)
    for i in range(0,n_states):
        xc_vals[:,i] = Matrix([symbols('x'+str(i)+str(j)) for j in range(0,l)])

    states = xc_vals.T.reshape(n_states*l,1)

    dynH = Matrix(dynamics)
    Aj = dynH.jacobian(states)
    Bj = dynH.jacobian(u)

    Aa = Aj*states
    Bb = Bj*u

    Cj = dynH - Aa - Bb #- Aj*states - Bj*u

    A = lambdify((input_states),Aj,'numpy')

    B = lambdify((input_states),Bj,'numpy')

    C = lambdify((input_states),Cj,'numpy')

    # A , B , Z are function Handles
    return A, B, C



def linearize_gpc_dynamics(dynamics, x, u, n_uncert, p):
    
    n_states = x.shape[0]

    Hp = GaussHermitePC(n_uncert,p)
    
    l = Hp.shape[0]
    
    xc_vals = zeros(l,n_states)
    for i in range(0,n_states):
        xc_vals[:,i] = Matrix([symbols('x'+str(i)+str(j)) for j in range(0,l)])

    states = xc_vals.T.reshape(n_states*l,1)

    dynH = Matrix(dynamics)
    Aj = dynH.jacobian(states)
    Bj = dynH.jacobian(u)

    Cj = dynH - Aj*states - Bj*u

    A = lambdify((states,u),Aj,'numpy')

    B = lambdify((states,u),Bj,'numpy')

    C = lambdify((states,u),Cj,'numpy')

    # A , B , Z are function Handles
    return A, B, C
