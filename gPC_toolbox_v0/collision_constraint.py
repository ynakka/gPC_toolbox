#!/usr/bin/env python3

import numpy as np
from sympy import *
from scipy.special import comb
from scipy.linalg import sqrtm

from gPC_toolbox_v0.project_gpc_2_statespace import meangpc_fullstate_noinit, meangpc_fullstate

# gives out linear equation (ax + b<=0)
def collision_constraint_3d_noinit(radius, obstacle_state, Xprev, num_states, num_uncert, polynomial_degree ):

    mean_states = meangpc_fullstate_noinit(Xprev,num_states,num_uncert,polynomial_degree)
    mean = mean_states.reshape((num_states,1))

    G = np.zeros((num_states,num_states)) # matrix to  pulling out position 
    G[0,0] = 1
    G[1,1] = 1
    G[2,2] = 1

    mean_position = (np.mat(G)*np.mat(mean)).reshape((num_states)) 
    #print(mean_position)

    collision_dist = obstacle_state.reshape((num_states)) - mean_position
    #print(obstacle_state)
    #print(mean_position)

    a = collision_dist.reshape((num_states,1))
    #print(a)
    b =  np.mat(-a.reshape((1,num_states)))*np.mat(obstacle_state.reshape((num_states,1))) + radius*np.linalg.norm(collision_dist,2)
    #print(b)

    return np.array(a,dtype=float), np.array(b,dtype=float) 


def collision_constraint_3d(radius, obstacle_state, Xprev, num_states, num_uncert, polynomial_degree ):

    mean_states = meangpc_fullstate(Xprev,num_states,num_uncert,polynomial_degree)
    mean = mean_states.reshape((num_states,1))

    G = np.zeros((num_states,num_states)) # matrix to  pulling out position 
    G[0,0] = 1
    G[1,1] = 1
    G[2,2] = 1
    mean_position = (np.mat(G)*np.mat(mean)).reshape((num_states)) 
    #print(mean_position)
    collision_dist = obstacle_state.reshape((num_states)) - mean_position
    #print(obstacle_state)
    #print(mean_position)
    a = collision_dist.reshape((num_states,1))
    #print(a)
    b =  np.mat(-a.reshape((1,num_states)))*np.mat(obstacle_state.reshape((num_states,1))) + radius*np.linalg.norm(collision_dist,2)
    #print(b)
    return np.array(a,dtype=float), np.array(b,dtype=float) 


def collision_constraint_2d_noinit(radius, obstacle_state, Xprev, num_states, num_gpcpoly ):
    mean_states = meangpc_fullstate_noinit(Xprev,num_states,num_gpcpoly)
    mean = mean_states.reshape((num_states,1))
    G = np.zeros((num_states,num_states)) # matrix to  pulling out position 
    G[0,0] = 1
    G[1,1] = 1
    mean_position = (np.mat(G)*np.mat(mean)).reshape((num_states)) 
    #print(mean_position)
    collision_dist = obstacle_state.reshape((num_states)) - mean_position
    #print(obstacle_state)
    #print(mean_position)
    a = collision_dist.reshape((num_states,1))
    #print(a)
    b =  np.mat(-a.reshape((1,num_states)))*np.mat(obstacle_state.reshape((num_states,1))) + radius*np.linalg.norm(collision_dist,2)
    #print(b)
    return np.array(a,dtype=float), np.array(b,dtype=float) 


def collision_constraint_2d(radius, obstacle_state, Xprev, num_states, num_uncert, polynomial_degree ):
    mean_states = meangpc_fullstate(Xprev,num_states,num_uncert,polynomial_degree)
    mean = mean_states.reshape((num_states,1))
    G = np.zeros((num_states,num_states)) # matrix to  pulling out position 
    G[0,0] = 1
    G[1,1] = 1
    mean_position = (np.mat(G)*np.mat(mean)).reshape((num_states)) 
    #print(mean_position)
    collision_dist = obstacle_state.reshape((num_states)) - mean_position
    #print(obstacle_state)
    #print(mean_position)
    a = collision_dist.reshape((num_states,1))
    #print(a)
    b =  np.mat(-a.reshape((1,num_states)))*np.mat(obstacle_state.reshape((num_states,1))) + radius*np.linalg.norm(collision_dist,2)
    #print(b)
    return np.array(a,dtype=float), np.array(b,dtype=float) 