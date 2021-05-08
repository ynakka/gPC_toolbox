#!/usr/bin/env python3

import numpy as np
import math as mt
from sympy import *
from scipy.special import comb

from scipy.integrate import solve_ivp


from itertools import combinations
from cloudpickle import dump, load


import matplotlib
import matplotlib.pyplot as plt

from gPC_toolbox.GaussHermitePC import GaussHermitePC
from gPC_toolbox.initial_conditions import initial_conditions
# initial_conditions(x_mean,x_var,n_states,num_uncertainty,p)
from gPC_toolbox.project_gpc_2_statespace import meangpc_noinit
from gPC_toolbox.project_gpc_2_statespace import meangpc
from gPC_toolbox.project_gpc_2_statespace import vargpc_noinit
from gPC_toolbox.project_gpc_2_statespace import vargpc

# load kalman filter propgation matrices 

kf_mean_dummy = open('kf_mean', 'rb')
f_kf_mean = load(kf_mean_dummy)
kf_mean_dummy.close()

kf_grad_dummy = open('kf_grad', 'rb')
f_kf_grad = load(kf_grad_dummy)
kf_grad_dummy.close()

# functions for numerical integration
#   
def func_kf_mean(y):
    return np.array(f_kf_mean(*y).reshape(2) ,dtype=float) 

def func_kf_var(y):
    return np.array(f_kf_grad(*y) ,dtype=float) 

# gpc projection matrices 
gpc_dyn_dummy = open('lambdified_pendulum_dynamics1', 'rb')
gpc_dyn = load(gpc_dyn_dummy)
gpc_dyn_dummy.close()

def f_gpc_dyn(input_vec):
    return np.array(gpc_dyn(*input_vec)).reshape(10)

def main():
    
    fig,axs = plt.subplots(1,2,figsize=(6,3))

    t_init = 0 
    t_final = 10
    t_steps = 50
    tvec = np.linspace(t_init,t_final,t_steps)
    dt = tvec[1]-tvec[0]

    initial_condition_mean = np.array([(np.pi/8),0.0])
    initial_condition_var = np.array([0,0])

    num_gpcpoly = 5
    num_state = 2
    polynomial_degree = 1
    num_uncert = 4 

    num_gpcstates = int(num_gpcpoly*num_state) 

    ## ===============================gPC propgagtion============================= 
    #

    gpc_init = initial_conditions(initial_condition_mean,initial_condition_var,num_state,num_uncert,polynomial_degree)
    # initial_conditions(x_mean,x_var,n_states,num_uncertainty,p)

    print('gpc_init',gpc_init)    
    gpc_states = np.zeros((t_steps,num_gpcstates))
    gpc_states[0,:] = gpc_init.reshape(10) 


    for i in range(1,int(t_steps)):
        input_array = np.concatenate((gpc_states[i-1,:].reshape((num_gpcstates)),np.array([dt])))
        gpc_step = f_gpc_dyn(input_array)
        gpc_states[i,:] = gpc_states[i-1,:] + gpc_step

    theta_mean_gpc = np.zeros(t_steps)
    theta_var_gpc = np.zeros(t_steps)

    theta_dot_mean_gpc = np.zeros(t_steps)
    theta_dot_var_gpc = np.zeros(t_steps)

    # gpc_mean 
    
    for i in range(0,int(t_steps)):
        theta_mean_gpc[i] = meangpc(gpc_states[i,0:num_gpcpoly],num_gpcpoly) #meangpc_noinit(gpc_states[i,0:num_gpcpoly],num_gpcpoly)
        theta_dot_mean_gpc[i] = meangpc(gpc_states[i,num_gpcpoly:int(num_gpcpoly*num_state)],num_gpcpoly)

    # gpc_variance 
# vargpc(x,n_states,n_uncert,p)
    for i in range(0,int(t_steps)):
        theta_var_gpc[i] =  np.sqrt(vargpc(gpc_states[i,0:num_gpcpoly],num_state,num_uncert,polynomial_degree))
        theta_dot_var_gpc[i] =  np.sqrt(vargpc(gpc_states[i,num_gpcpoly:int(num_gpcpoly*num_state)],\
            num_state,num_uncert,polynomial_degree))


    axs[0].plot(tvec,theta_mean_gpc,color='blue')
    axs[0].fill_between(tvec,
        theta_mean_gpc-2*theta_var_gpc,
        theta_mean_gpc+2*theta_var_gpc,
        linewidth=1e-2,
        alpha=0.4,color='blue')

    axs[1].plot(tvec,theta_dot_mean_gpc,color='blue')
    axs[1].fill_between(tvec,
        theta_dot_mean_gpc-2*theta_dot_var_gpc,
        theta_dot_mean_gpc+2*theta_dot_var_gpc,
        linewidth=1e-2,
        alpha=0.4,color='blue',label=r'gPC-$2\sigma$')




    # -----------------------------------------xxxxxxxx-------------------------------------------------------
    

    # ------------------------ Monte Carlo Propagation-------------------------------------------------------

    num_trials = 1000
    t_init = 0 
    t_final = 10
    t_steps = 50
    tvec = np.linspace(t_init,t_final,t_steps)
    dt = tvec[1]-tvec[0]
    
    mc_sim = []

    # generate monte carlo data
    for trial in range(num_trials):
        print('trial: #',trial)
        data_mc  =np.zeros((t_steps,num_state))
        data_mc[0,:] = initial_condition_mean
        for t in range(t_steps-1):
            data_mc[t+1,:] = data_mc[t,:] + dt*func_kf_mean(data_mc[t,:].reshape(2)) + np.array([mt.sqrt(dt*0.001)*np.random.randn(),mt.sqrt(dt*0.001)*np.random.randn()])
        mc_sim.append(data_mc)    


    mc_mean = np.zeros(t_steps)
    mc_sd = np.zeros(t_steps)
    for jj in range(t_steps):
        mc_mean[jj] = 0 
        for ii in range(num_trials):
            mc_mean[jj] = mc_mean[jj] + mc_sim[ii][jj][0]
        mc_mean[jj] = mc_mean[jj]/num_trials
    # print(mc_mean)
    # sd unbiased 
    for jj in range(t_steps):
        mc_sd[jj] = 0 
        for ii in range(num_trials):
            mc_sd[jj] = mc_sd[jj] + (mc_sim[ii][jj][0]- mc_mean[jj])**2
        mc_sd[jj] = np.sqrt(mc_sd[jj]/(num_trials))

    for trial in range(num_trials):
        axs[0].scatter(tvec,mc_sim[trial][:,0].reshape((t_steps)),s=0.8,color='orangered',alpha=0.8)
    
    

    mc_mean = np.zeros(t_steps)
    mc_sd = np.zeros(t_steps)
    for jj in range(t_steps):
        mc_mean[jj] = 0 
        for ii in range(num_trials):
            mc_mean[jj] = mc_mean[jj] + mc_sim[ii][jj][1]
        mc_mean[jj] = mc_mean[jj]/num_trials
    # print(mc_mean)
    # sd unbiased 
    for jj in range(t_steps):
        mc_sd[jj] = 0 
        for ii in range(num_trials):
            mc_sd[jj] = mc_sd[jj] + (mc_sim[ii][jj][1]- mc_mean[jj])**2
        mc_sd[jj] = np.sqrt(mc_sd[jj]/(num_trials))

    axs[1].scatter(tvec,mc_sim[0][:,1].reshape((t_steps)),s=0.25,color='orangered',alpha=0.8,label='MC')
    for trial in range(1,num_trials):
        axs[1].scatter(tvec,mc_sim[trial][:,1].reshape((t_steps)),s=0.25,color='orangered',alpha=0.8)

    axs[1].legend(fontsize='10',loc='best')
    

    
    # ax.plot(tvec,mc_mean)
    # ax.fill_between(tvec,
    #     mc_mean-2*mc_sd,
    #     mc_mean+2*mc_sd,
    #     linewidth=1e-2,
    #     alpha=0.5)
    # plt.xlim((0,1))
    # plt.ylim((0,0.5))
    #------------------------------------------xxxxxxxxxxx-------------------------------------------------------

    # # ============================Kalman filter propagation=========================================== 
    t_init = 0 
    t_final = 10
    t_steps = 50
    tvec = np.linspace(t_init,t_final,t_steps)
    dt = tvec[1]-tvec[0]
    kf_mean = np.zeros((t_steps,2))
    kf_var = np.zeros((t_steps,2,2))
    # print(kf_var_sim[0])
    kf_mean[0,:] = initial_condition_mean
    Q_var = np.array([[0.001,0],[0,0.001]])*dt
    for ii in range(t_steps-1):
        kf_mean[ii+1,:] = kf_mean[ii,:] + dt*func_kf_mean(kf_mean[ii,:].reshape(2))
        jacobian_dummy = (func_kf_var(kf_mean[ii])*dt + np.eye(2))
        # print(jacobian_dummy.T)
        # print(type(jacobian_dummy))
        covar_prop = np.array(np.mat(jacobian_dummy)*np.mat(kf_var[ii])*np.mat(jacobian_dummy.T))
        # print(covar_prop)
        kf_var[ii+1] = covar_prop + Q_var
            # print(kf_var[ii])
    axs[0].plot(tvec,kf_mean[:,0],color='green')
    axs[0].fill_between(tvec,
        kf_mean[:,0]-2*np.sqrt(kf_var[:,0,0]),
        kf_mean[:,0]+2*np.sqrt(kf_var[:,0,0]),
        linewidth=1e-2,
        alpha=0.6,color='green')

    axs[1].plot(tvec,kf_mean[:,1],color='green')
    axs[1].fill_between(tvec,
        kf_mean[:,1]-2*np.sqrt(kf_var[:,1,1]),
        kf_mean[:,1]+2*np.sqrt(kf_var[:,1,1]),
        linewidth=1e-2,
        alpha=0.6,color='green',label=r'linear-$2\sigma$')
        # print(kf_var[ii])
    # axs[0].plot(tvec,kf_mean[:,0],color='green',label='linear')
    # axs[0].fill_between(tvec,
    #     kf_mean[:,0]-2*np.sqrt(kf_var[:,0,0]),
    #     kf_mean[:,0]+2*np.sqrt(kf_var[:,0,0]),
    #     linewidth=1e-2,
    #     alpha=0.2,color='green')

    # axs[1].plot(tvec,kf_mean[:,1],color='green',label='linear')
    # axs[1].fill_between(tvec,
    #     kf_mean[:,1]-2*np.sqrt(kf_var[:,1,1]),
    #     kf_mean[:,1]+2*np.sqrt(kf_var[:,1,1]),
    #     linewidth=1e-2,
    #     alpha=0.2,color='green',label=r'linear-$2\sigma$')


    #--------------------------------------------xxxxxxxxxxxxxxxxx----------------------------------------
    axs[0].set_ylabel(r'$\theta$')
    axs[0].set_xlabel('time (s)')
    #ax1.set_ylim((100,130))
    axs[1].set_ylabel(r'$\dot{\theta}$')
    axs[1].set_xlabel('time (s)')
    
    plt.legend(ncol=2,fontsize=7)

    plt.tight_layout()
    plt.savefig('gpc_mc_v1.pdf',dpi=300)
    plt.savefig('gpc_mc_v1.jpeg',dpi=300)
    # plt.show()


    
    return 1


if __name__ == "__main__":
    main()

    # fig, ax = plt.subplots()
    # for i in range(10):
    #     y0 = np.array([0.01])
    #     sol = solve_ivp(f_dyn0,[0,10],y0)
    #     ll = sol.t.shape[0]
    #     y = sol.y.reshape(ll)
    #     ax.plot(sol.t,y)

    # ax.grid()
    # plt.show()




        




