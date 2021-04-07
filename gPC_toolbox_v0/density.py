#!/usr/bin/env python3

import numpy as np
from sympy import *
from scipy.special import comb
from scipy.linalg import sqrtm
from itertools import combinations

def gaussian_density(xx):
    rho = exp(-(xx**2)/2)/sqrt(2*pi)
    return rho

