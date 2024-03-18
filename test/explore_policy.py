import cvxpy as cp
import numpy as np
import os
import sys

parent_directory = os.path.abspath('..')
sys.path.append(parent_directory)

from utils import soft_bellman_operation
from utils import create_Pi
from utils import BlockingGridworld
import scipy
import matplotlib.pyplot as plt

import time
from numpy.linalg import norm
from scipy.linalg import lapack



# Define the variables
if __name__ == '__main__':

    grid_size = 5
    wind = 0.1
    discount = 0.9   
    start_state = 4 
    feature_dim = 2

    landmark_locations = [(0,24), (3,1), (3,19), (14,14)]
    p1,p2 = landmark_locations[-1]
    horizon = 15
    theta = 5*np.ones((2,1))
    theta[1] -= 5

    gw = BlockingGridworld(grid_size,wind,discount,horizon,start_state, feature_dim, p1,p2,theta)
    reward = gw.reward_v

    V,Q,pi = soft_bellman_operation(gw,reward)
    # print(pi.shape)
    print(f"pi[{horizon-5}] = ", pi[horizon-5].round(decimals = 3))

 
    