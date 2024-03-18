import cvxpy as cp
import numpy as np

from utils import soft_bellman_operation
from utils import create_Pi
from utils import BlockingGridworld
import scipy
import matplotlib.pyplot as plt
import os
import time
from numpy.linalg import norm
from scipy.linalg import lapack


# Define the variables
if __name__ == '__main__':

    grid_size = 5
    wind = 0.1
    discount = 0.9 
    horizon =   13   
    start_state = 4 
    feature_dim = 2

    landmark_locations = [(0,24), (3,1), (3,19), (14,14)]
    p1,p2 = landmark_locations[0]
    horizon = 15
    theta = 5*np.ones((2,1))
    theta[1] += 5

    gw = BlockingGridworld(grid_size,wind,discount,horizon,start_state, feature_dim, p1,p2,theta)
    reward = gw.reward_v

    V,Q,pi = soft_bellman_operation(gw,reward)
    Gamma, Xi , n_unreach_states = gw.prune_Gamma_and_Xi(pi)
    F = gw.F_matrix()

    prob_ = np.linalg.lstsq(Gamma, Xi, rcond = -1)         
    recovered_reward = prob_[0][:gw.n_states*gw.n_actions]


    P = np.hstack((np.eye(F.shape[0]), np.zeros((F.shape[0], Gamma.shape[1] - F.shape[0])) ))
            
    A = np.hstack((Gamma, np.zeros((Gamma.shape[0], gw.feature_dim))))
    b = Xi
    C = np.hstack((-P,F))
    d = np.zeros((F.shape[0],1))

    x = cp.Variable((A.shape[1],1))

    # Define the objective function
    objective = cp.Minimize(cp.norm(A @ x - b))

    # Define the constraints
    constraints = [C @ x == d]

    # Formulate the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve(verbose=True)

    # Get the optimal value of x
    print(problem)
    optimal_x = x.value

    # Print the optimal solution
    # print("Optimal x:", optimal_x)
    print(Gamma.shape)
    print(optimal_x.shape)
    print("\nThe optimal value is", problem.value)
    print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)
    print("The norm of the equality residual is ", cp.norm(C @ x - d, p=2).value)
