from utils import soft_bellman_operation
from utils import create_Pi
from utils import BlockingGridworld
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os


cwd = os.getcwd()
image_folder_path = os.path.join(cwd, "plots")

if __name__ == '__main__':
    grid_size = 5
    wind = 0.1
    discount = 0.9
    horizon =   5   
    start_state = 20    
    feature_dim = 2
    p1 = 0  # position of culvers burger joint
    p2 = 24 # position of charging station

    theta = 5*np.ones((2,1))


    gw = BlockingGridworld(grid_size,wind,discount,horizon,start_state, feature_dim, p1,p2,theta)

    # assert that the transition matrices are markovian
    for a in range(gw.n_actions):   
        assert np.linalg.norm(gw.transition_probability[:,a,:].sum(axis = 1)[:,None] -np.ones((25,1))) <= 1e-5


    reward = gw.reward_v
    F = gw.F_matrix()
    ones_ = np.ones((gw.n_states*gw.n_actions,1))

    # assert that ran(1) \in \ran(F)
    # assert np.linalg.norm(F@v[0] - ones_) <= 1e-5


    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)

    fig = plt.figure(figsize=(10,8))

    # generate plots for reachability vs indifiability
    start_states = [0,4,7, 12, 15,20,24]
    color = ['blue', 'purple','green', 'magenta', 'yellow', 'cyan', 'black']
    overall_min = 1e6
    overall_max = -1e6
    
    for start_state,col in zip(start_states, color):
        # print(start_state)
        t_step = []
        data = []
        total_unreach = ()

        for t in range(3,20):
            gw = BlockingGridworld(grid_size,wind,discount,t,start_state, feature_dim, p1,p2,theta)
            V,Q,pi = soft_bellman_operation(gw,reward)
            Gamma, Xi , n_unreach_states = gw.prune_Gamma_and_Xi(pi)

            # print("shape of Gamma is: ", Gamma.shape)
            # quick sanity check
            # v = np.linalg.pinv(Gamma) @ Xi    
            # assert np.linalg.norm(Gamma@v - Xi) <= 1e-5

            data.append( gw.compute_projected_kernel(Gamma))
            t_step.append(t)
            # print(t)
        
        # print("min is: ", min(data))    
        if min(data) < overall_min:
            overall_min = min(data)
        if max(data)> overall_max:
            overall_max = max(data)
        
        plt.plot(t_step, data, linestyle='--',  color= col, linewidth = 2.0, alpha=0.5, label = f"start state = {start_state}")    
        # plt.plot(t_step, n_unreach_states, linestyle=':',  color= col, linewidth = 1.0, alpha=0.5, label = f"start state = {start_state}")

    # plt.plot(x, y, color='blue', alpha=0.5)
    # plt.set_xticks(np.arange(0,5),np.arange(4,10))
    K = scipy.linalg.null_space(Gamma)
    projected_K = K[:gw.n_states*gw.n_actions,:]
    print(np.round(projected_K[:gw.n_states,0], decimals=3 ))
    print(np.round(projected_K[:gw.n_states,1], decimals=3 ))
    print(np.round(projected_K[:gw.n_states,2], decimals=3 ))
    print(np.round(projected_K[:gw.n_states,3], decimals=3 ))
    plt.axhline(y = overall_min, color = 'black', linestyle=':',linewidth = 3.0, label = f'min dim = {overall_min}')
    plt.xlabel('Horizon')
    plt.ylabel('Dimension of $K_\eta$')
    plt.title('Plots for Figure 1(c) of L4DC Paper')
    # plt.grid(True)
    plt.gca().set_yticks(range(0, overall_max, 1), minor=True) 
    plt.legend()
    plt.grid()
    # plt.show()    
    plt.savefig(os.path.join(image_folder_path, 'fig_1_c.png'), dpi = 800)    


   