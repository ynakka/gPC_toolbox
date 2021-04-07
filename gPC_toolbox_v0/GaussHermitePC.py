#!/usr/bin/env python3
import numpy as np
from sympy import *
from scipy.special import comb
from itertools import combinations

def lambdify_gauss_hermite_pc(num_uncert,num_states,polynomial_degree):

    xi_symbols = [symbols('xi'+str(i)) for i in range(1,num_uncert-num_states+1)]
    xi = zeros(num_uncert-num_states,1) #xc_vals = zeros(l,n_states)
    for ii in range(num_uncert-num_states):
        xi[ii] =  xi_symbols[ii]
    #print(xi)
    Hp = GaussHermitePC(num_uncert-num_states,polynomial_degree)
    #print(Hp)
    Hvec = lambdify((xi),Hp,'numpy')

    return Hvec

def GaussHermitePC(n,p):
    if n==1:
        xi = symbols('xi')
        Hp = Matrix([((1/sqrt(2))**i)*hermite(i, xi/sqrt(2)) for i in range(p+1)])
        psi = Hp
        return psi
    else:
        xi = symbols('xi')
        Hp = Matrix([((1/sqrt(2))**i)*hermite(i, xi/sqrt(2)) for i in range(p+1)])
        xi_num = [symbols('xi'+str(i)) for i in range(1,n+1)]
        Hp_mv = zeros(p+1,n)
        for i in range(n):
            for j in range(p+1):
                Hp_mv[j,i] = Hp[j].subs([(xi,xi_num[i])])
        psi_size = int(comb(n+p,p))
        psi = zeros(psi_size,1)
        index = [np.zeros((1,n),dtype='float32')]
        for i in range(1,p+1):
            numi = np.array(list(combinations(list(range(1,n+i)),n-1)))
            num1 = np.zeros((numi.shape[0],1),dtype='float32')
            num2 = (n+i) + num1
            concat = np.hstack((num1,numi,num2))
            indexi = np.flipud(np.diff(concat,n=1,axis=1))-1
            index = index + indexi.tolist()
            if not np.allclose(np.sum(indexi,axis=1), i *np.ones((int(comb(n+i-1,n-1)),1))):
                print('The sum of each row has to be equal to p-th order')
                return
        index_mat = np.vstack(index)
        for i in range(1, psi_size+1):
            mult_s = 1
            for j in range(n):
                mult_s = mult_s * Hp_mv[int(index_mat[i-1][j]),j]
            psi[i-1] = mult_s
        return psi
    
if __name__ == "__main__":
    psi2 = GaussHermitePC(7,1)
    init_printing()
    print(psi2)
