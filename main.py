from utils import soft_bellman_operation
from utils import create_Pi
# from utils import BlockingGridworld
from utils import cvpxy_LSE
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import time
from numpy.linalg import norm
from scipy.linalg import lapack
import cvxpy as cp
import argparse
from dynamics.GridWorlds import BasicGridWorld, StripBlockingGridWorld, WallBlockingGridWorld

cwd = os.getcwd()
image_folder_path = os.path.join(cwd, "plots","feature_based")

if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)


def main(BlockingGridworld):
    grid_size = 5
    wind = 0.1
    discount = 0.9
    horizon =   5   
    start_state = 20    
    feature_dim = 2
    p1 = 0  # position of burger joint
    p2 = 24 # position of charging station

    theta = 5*np.ones((2,1))
    feature_type = "dense"

    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)

    fig = plt.figure(figsize=(10,8))


    start_states = [0,4,7, 12, 15,20,24]
    color = ['blue', 'purple','green', 'magenta', 'yellow', 'cyan', 'black']
    overall_min = 1e6
    overall_max = -1e6
    
    for start_state,col in zip(start_states, color):
      
        t_step = []
        data = []
        total_unreach = ()

        for t in range(3,20):
            gw = BlockingGridworld(grid_size,wind,discount,t,start_state, feature_dim, p1,p2,theta,feature_type)
            reward = gw.reward_v
            V,Q,pi = soft_bellman_operation(gw,reward)
            Gamma, Xi , n_unreach_states = gw.prune_Gamma_and_Xi(pi)

            data.append( gw.compute_projected_kernel(Gamma))
            t_step.append(t)
            
        if min(data) < overall_min:
            overall_min = min(data)
        if max(data)> overall_max:
            overall_max = max(data)
        
        plt.plot(t_step, data, linestyle='--',  color= col, linewidth = 2.0, alpha=0.5, label = f"start state = {start_state}")    
        

    script = gw.__str__()

    K = scipy.linalg.null_space(Gamma)
    projected_K = K[:gw.n_states*gw.n_actions,:]

    plt.axhline(y = overall_min, color = 'black', linestyle=':',linewidth = 3.0, label = f'min dim = {overall_min}')
    plt.xlabel('Horizon')
    plt.ylabel('Dimension of $K_\eta$')
    plt.title(f'Plots for Figure 1({script}) of L4DC Paper')
    plt.grid(True)
    plt.gca().set_yticks(range(0, overall_max, 1), minor=True) 
    plt.legend()
    plt.grid()
    plt.show()    
    # plt.savefig(os.path.join(image_folder_path, 'fig_1_c.png'), dpi = 800)

def main_feature_based(BlockingGridworld, feature_type):
    
    grid_size = 5
    wind = 0.1
    discount = 0.9 
    horizon =   13   
    start_state = 4 
    feature_dim = 2

    theta = 5*np.ones((2,1))
    theta[1] += 5
    landmark_locations = [(0,24), (3,1), (3,19), (14,14)]
    color = ['blue', 'red','green', 'cyan']
    style = ["-","--",":","-."]
    overall_min = 1e6
    overall_max = -1e6

    # feature_type = "dense"
    
    for landmark_loc,col,st in zip(landmark_locations, color,style):
        # print(start_state)
        t_step = []
        data = []
        total_unreach = ()
        p1,p2 = landmark_loc

        for t in range(3,20):

            gw = BlockingGridworld(grid_size,wind,discount,t,start_state, feature_dim, p1,p2,theta, feature_type)
            reward = gw.reward_v
   
            V,Q,pi = soft_bellman_operation(gw,reward)
            Gamma, Xi , n_unreach_states = gw.prune_Gamma_and_Xi(pi)
          
    
            # TEST IF WE AN RECOVER A SOLUTION FOR:   Gamma@reward = Xi 
            #-------------------------------------------------------------------#
            sanity_check_1_res = np.linalg.lstsq(Gamma, Xi, rcond = -1)         
            recovered_reward = sanity_check_1_res[0][:gw.n_states*gw.n_actions] 
            assert np.linalg.norm(Xi - Gamma@sanity_check_1_res[0]) <= 1e-5     
            #-------------------------------------------------------------------#
            
           
            K = scipy.linalg.null_space(Gamma)
            projected_K = K[:gw.n_states*gw.n_actions,:]

            # TEST IF THE TRUE REWARD \in RECOVERED_REWARD + SPAN(projected_K)                        
            #-------------------------------------------------------------------------------------------#
            mat_ = recovered_reward + projected_K                                                     
            sanity_check_2_res = np.linalg.lstsq(mat_, reward.ravel(order='F')[:,None], rcond = -1)      
            assert np.linalg.norm(reward.ravel(order='F')[:,None] - mat_@sanity_check_2_res[0]) <= 1e-5  
            #-------------------------------------------------------------------------------------------#


            F = gw.F_matrix()

            # TEST IF THE RECOVERED REWARD \in SPAN(F)
            #-----------------------------------------------------------------------#
            sanity_check_3_res = np.linalg.lstsq(F, recovered_reward, rcond = -1)
            try:
                assert norm(F@sanity_check_3_res[0] - recovered_reward) <= 1e-5
            except AssertionError:
                pass
            #-----------------------------------------------------------------------#
          
            # recover a reward using equality constrained Lstsq
            reward_ , theta_  , prob = cvpxy_LSE(Gamma,Xi,gw , verbose = False)
            # print(prob.status)

            

            A = scipy.linalg.orth(projected_K)

            # assert that a vector of ones is in the projected kernel
            ones_ = np.ones_like(A[:,0][:None])
            resid = np.linalg.lstsq(A,ones_, rcond=-1)
            assert resid[1] <= 1e-5

            # A = projected_K

            B = F   
            

            # # find their intersection
            # M = np.hstack((A,-B))

            # V = scipy.linalg.null_space(M)
            # V_p = V[:A.shape[1]]

            # intersection = A@V_p
            # # basis_intersection = scipy.linalg.orth(intersection)

            l = scipy.linalg.null_space(np.hstack((scipy.linalg.null_space(A.T), scipy.linalg.null_space(B.T)  )).T)

            # print(l.shape)
            # if l.shape[1] == 1:
            #     print("P = ", landmark_loc)
            #     # print("min = ", min(l))
            #     # print("max = ", max(l))
            #     print("abs: ", np.abs(max(l) - min(l)))
                # time.sleep(1)
            data.append(l.shape[1]) 
            t_step.append(t)
            

        if min(data) < overall_min:
            overall_min = min(data)
        if max(data)> overall_max:
            overall_max = max(data)
        
        plt.plot(t_step, data, linestyle=st,  color= col, linewidth = 1.0, alpha=0.8, label = f"landmark location = {(p1,p2)}")    
        # plt.plot(t_step, n_unreach_states, linestyle=':',  color= col, linewidth = 1.0, alpha=0.5, label = f"start state = {start_state}")
    # print(F)
    plt.axhline(y = overall_min, color = 'black', linestyle=':',linewidth = 3.0, label = f'min dim = {overall_min}')
    plt.xlabel('Horizon')
    plt.ylabel('Dimension of Intersection')
    plt.title(f'Plots for Figure 2 of L4DC Paper -- Start = {start_state}')
    plt.gca().set_yticks(range(0, overall_max, 1), minor=True) 
    plt.legend()
    # plt.grid()
    plt.show()    
    # plt.savefig(os.path.join(image_folder_path, f'fig_2_{start_state}.png'), dpi = 800)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code for L4DC 2024 Paper: Learning True Objectives")
    parser.add_argument("-d" , choices= ['fig1a', 'fig1b', 'fig1c'], default= 'fig1c', help = "specifying the dynamics")
    parser.add_argument("--with_features" , action = 'store_true', help = "flag to use the feature based implementation")
    parser.add_argument("--dense" , action = 'store_true', help = "flag to use the dense features")
    parser.add_argument("--sparse" , action = 'store_false', help = "flag to use the sparse features")
    args = parser.parse_args()
 
    if args.d == "fig1a":
        print("Using the Dynamics of Figure 1.a ...")
        BlockingGridworld = BasicGridWorld
    elif args.d == "fig1b":
        print("Using the Dynamics of Figure 1.b ...")
        BlockingGridworld = StripBlockingGridWorld
    else:
        print("Using the Dynamics of Figure 1.c ...")
        BlockingGridworld = WallBlockingGridWorld

    feature_type = "dense" if args.dense else "sparse"

    if args.with_features:
        if args.dense:
            print("Running the main function with Dense Features ...")
            
        else:
            print("Running the main function with Sparse Features ...")

        main_feature_based(BlockingGridworld, feature_type)

    else:
        print("Running the main function without Features ...")
        main(BlockingGridworld)

