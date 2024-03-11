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
    start_state = 24  
    feature_dim = 2
    # p1 = 20 # position of culvers burger joint
    # p2 = 20 # position of charging station

    theta = 5*np.ones((2,1))

    landmark_locations = [(0,24), (1,3), (4,20), (14,14)]
    color = ['blue', 'red','green', 'magenta']
    overall_min = 1e6
    overall_max = -1e6
    
    for landmark_loc,col in zip(landmark_locations, color):
        # print(start_state)
        t_step = []
        data = []
        total_unreach = ()
        p1,p2 = landmark_loc

        for t in range(3,20):
            gw = BlockingGridworld(grid_size,wind,discount,t,start_state, feature_dim, p1,p2,theta)
            reward = gw.reward_v

            V,Q,pi = soft_bellman_operation(gw,reward)
            Gamma, Xi , n_unreach_states = gw.prune_Gamma_and_Xi(pi)

            K = scipy.linalg.null_space(Gamma)
            projected_K = K[:gw.n_states*gw.n_actions,:]
            # print("RANK IS: ", np.linalg.matrix_rank(projected_K))
            F = gw.F_matrix()
            # print("shape of Gamma is: ", Gamma.shape)
            # quick sanity check
            # v = np.linalg.pinv(Gamma) @ Xi    
            # assert np.linalg.norm(Gamma@v - Xi) <= 1e-5
            A = projected_K
            B = F
            # print("Shape of B is: ", B.shape)

            # # find their intersection
            M = np.hstack((A,-B))

            V = scipy.linalg.null_space(M)
            V_p = V[:A.shape[1]]

            intersection = A@V_p
            # print(np.linalg.matrix_rank(intersection))
            data.append( np.linalg.matrix_rank(intersection))
            t_step.append(t)
            # print(t)
        
        # # print("min is: ", min(data))    
        if min(data) < overall_min:
            overall_min = min(data)
        if max(data)> overall_max:
            overall_max = max(data)
        
        plt.plot(t_step, data, linestyle='--',  color= col, linewidth = 2.0, alpha=0.5, label = f"landmark location = {(p1,p2)}")    
        # plt.plot(t_step, n_unreach_states, linestyle=':',  color= col, linewidth = 1.0, alpha=0.5, label = f"start state = {start_state}")

    plt.axhline(y = overall_min, color = 'black', linestyle=':',linewidth = 3.0, label = f'min dim = {overall_min}')
    plt.xlabel('Horizon')
    plt.ylabel('Dimension of Intersection')
    plt.title(f'Plots for Figure 2 of L4DC Paper -- Start = {start_state}')
    # plt.grid(True)
    plt.gca().set_yticks(range(0, overall_max, 1), minor=True) 
    plt.legend()
    plt.grid()
    # plt.show()    
    plt.savefig(os.path.join(image_folder_path, f'fig_2_{start_state}.png'), dpi = 800)    