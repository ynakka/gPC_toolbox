#!/usr/bin/env python3

import numpy as np
from sympy import *
from scipy.special import comb
from scipy.linalg import sqrtm


from itertools import combinations
from cloudpickle import dump, load

from gPC_toolbox.fast_project_sde_2_ode import make_deterministic_sym 

def main():
    ## -----------------------------------------------------------------
    ## -----------------------Simple Pendulum-----------------------------
    ## -----------------------------------------------------------------

    # Dynamics
    num_states = 2
    x = Matrix([symbols('x'+str(i)) for i in range(0,num_states)])
    u1 = symbols('u1')
    u2 = symbols('u2')
    u3 = symbols('u3')
    # Control Input 
    u = Matrix(3,1,[u1,u2,u3])

    # Uncertainty In Control same in x, y, z, F = F(1+theta)
    mu = np.array([0])
    sigma = np.array([0.001]) # this is variance

    theta = Matrix([symbols('theta'+str(i)) for i in range(0,2)])

    
    # f =  Matrix([[x[1]],[-9.8*sin(x[0])]])
    f =  Matrix([[x[1]],[-sin(x[0]) -0.8*x[1]]])
    
    g = Matrix([[np.sqrt(sigma[0])*theta[0]],[np.sqrt(sigma[0])*theta[1]]])

    print('theta[0]',theta[0])
    # f = Matrix([[x[1]]]).col_join(Matrix([[sx]]))

    n_uncert = 4
    polynomial_degree = 1
    # print(f+g)

    dyn_det, dyn_det_file, input_dyn = make_deterministic_sym(f,g,x,u,theta,n_uncert,polynomial_degree,init_uncertain=True)

    # print('dynamics:',dyn_det)
    print('Saving File.')

    name = 'gpc_pendulum'#+str(polynomial_degree) 
    with open(name, 'wb') as fdyn: 
        dump((dyn_det), fdyn)

    name = 'lambdified_pendulum_dynamics'+str(polynomial_degree)
    with open(name, 'wb') as fdyn_file: 
        dump((dyn_det_file), fdyn_file)

    name = 'lambdified_pendulum_dyn_input'#+str(polynomial_degree)
    with open(name, 'wb') as fdyn_file: 
        dump((input_dyn), fdyn_file)

    # linearized propagation 

    # print(f)

    kf_mean_lambdified = lambdify((x),f,'numpy')
    kf_var_lambdified = lambdify((x),f.jacobian(x),'numpy')

    name = 'kf_mean'
    with open(name, 'wb') as kf_file: 
         dump((kf_mean_lambdified), kf_file)
    
    name = 'kf_grad'
    with open(name, 'wb') as kf_var_file: 
         dump((kf_var_lambdified), kf_var_file)     
    # print(f.jacobian(x))


    return 1


if __name__ == "__main__":
    main()

    



    









