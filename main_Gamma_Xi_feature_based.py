from utils import soft_bellman_operation
from utils import create_Pi
from utils import BlockingGridworld
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os


cwd = os.getcwd()
image_folder_path = os.path.join(cwd, "plots","feature_based")

if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)

if __name__ == '__main__':
    grid_size = 5
    wind = 0.1
    discount = 0.9
    horizon =   13   
    start_state = 4   
    feature_dim = 2
    p1 = 20 # position of culvers burger joint
    p2 = 20 # position of charging station

    theta = 5*np.ones((2,1))


    gw = BlockingGridworld(grid_size,wind,discount,horizon,start_state, feature_dim, p1,p2,theta)

    # assert that the transition matrices are markovian
    for a in range(gw.n_actions):   
        assert np.linalg.norm(gw.transition_probability[:,a,:].sum(axis = 1)[:,None] -np.ones((25,1))) <= 1e-5
    

    reward = gw.reward_v

    F = gw.F_matrix()
    ones_ = np.ones((gw.n_states*gw.n_actions,1))

    # compute the soft optimal policy
    V,Q,pi = soft_bellman_operation(gw,reward)

    # assert that ran(1) \in \ran(F)
    # assert np.linalg.norm(F@v[0] - ones_) <= 1e-5


    Gamma = gw.construct_Gamma()
    Xi = gw.construct_Xi(pi)
    # Gamma, Xi , n_unreach_states = gw.prune_Gamma_and_Xi(pi)
    K = scipy.linalg.null_space(Gamma)
    projected_K = K[:gw.n_states*gw.n_actions,:]

    # print(np.round(projected_K[:gw.n_states,0], decimals=3 ))
    # print(np.round(projected_K[:gw.n_states,1], decimals=3 ))
    # print(np.round(projected_K[:gw.n_states,2], decimals=3 ))
    # print(np.round(projected_K[:gw.n_states,3], decimals=3 ))  

    print("The shape of projected K is: ", projected_K.shape)
    print("The rank of projected K is: ", np.linalg.matrix_rank(projected_K))
    print("Testing with Features ...")  
    # # construct K_eta     from equation (7) in the paper
    # K_eta = K[-gw.n_states:,:]
    # print(K_eta.shape)
    # # find the intersection as in equation (18)     
    # A = E@K_eta
    # B = F

    A = projected_K
    B = F
    print("Shape of B is: ", B.shape)

    # # find their intersection
    M = np.hstack((A,-B))

    V = scipy.linalg.null_space(M)
    print(V.shape)
    # # assert V.shape[1] == 1

    # V_p = V[:A.shape[1]]

    # intersection = A@V_p

    # # assert that the intersecttion is ran(1)
    # assert np.abs(min(intersection) - max(intersection)) <= 1e-6
    # print("Strongly Identifiable with Features.")


    # # print(gw.reachable_tube())