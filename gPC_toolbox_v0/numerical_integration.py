#!/usr/bin/env python3

import numpy as np
from sympy import *
from scipy.special import comb
from scipy.linalg import sqrtm

def integrate_gauss(weight,node,funct,int_variable):
    """ 
    Integration using Gauss- Hermite Quadrature 
        \int_{-infty}^{infty} rho(xi)*f(xi) xi = \sum_{i} w_{i} f(n_{i})       
    """
    sum = np.zeros_like(funct)
    for i in range(len(weight)):
        sum += weight[i]*funct.subs({int_variable:node[i]})
    return sum

    