from utils import soft_bellman_operation
from utils import create_Pi
from utils import BlockingGridworld
import numpy as np
import scipy

if __name__ == '__main__':
    grid_size = 5
    wind = 0.1
    discount = 0.9
    horizon = 10
    start_state = 15 
    feature_dim = 2
    p1 = 24  # position of culvers burger joint
    p2 = 24 # position of charging station

    theta = 5*np.ones((2,1))


    gw = BlockingGridworld(grid_size,wind,discount,horizon,start_state, feature_dim, p1,p2,theta)

    # assert that the transition matrices are markovian
    for a in range(gw.n_actions):
        assert np.linalg.norm(gw.transition_probability[:,a,:].sum(axis = 1)[:,None] -np.ones((25,1))) <= 1e-5


    reward = gw.reward_v
    F = gw.F_matrix()
    ones_ = np.ones((gw.n_states*gw.n_actions,1))

    # reward[-1][-1] = 1
    V,Q,pi = soft_bellman_operation(gw,reward)

    v = np.linalg.lstsq(F,ones_, rcond = None)

    # assert that ran(1) \in \ran(F)
    # assert np.linalg.norm(F@v[0] - ones_) <= 1e-5


    # Construct E and P
    I = np.eye(gw.n_states)
    E = I
    P = gw.transition_probability[:,0,:]

    for a in range(1, gw.n_actions):
        E = np.vstack((E,I))
        P = np.vstack((P, gw.transition_probability[:,a,:]))

    # Construct Psi and Pi  
    gamma = gw.discount
    psi_rows = (horizon - 1)*gw.n_states*gw.n_actions
    psi_cols = horizon*gw.n_states

    Pi = create_Pi(pi)

    Psi = np.zeros((psi_rows, psi_cols))
    Reach, UnReach = gw.get_reachable_tube()

    # for t in range(5):
    #     print(f"Reach[{t}] = ", Reach[str(t)] )
    #     print(f"UnReach[{t}] = ", UnReach[str(t)] )

    for i in range(horizon-1):
        Psi[i*gw.n_states*gw.n_actions:(i+1)*gw.n_states*gw.n_actions, i*gw.n_states:(i+1)*gw.n_states] = E
        Psi[i*gw.n_states*gw.n_actions:(i+1)*gw.n_states*gw.n_actions, (i+1)*gw.n_states:(i+2)*gw.n_states] = -discount*P


    # print(Psi.shape)
    # # vector from remark (1) that is in kernel of Psi, sanity check for Psi
    # v = np.ones((horizon*grid_size**2,1))
    # for i in range(horizon):
    #     v[i*grid_size**2:(i+1)*grid_size**2] = discount**(horizon-1 -i)
        
    # assert np.linalg.norm(Psi@v) <= 1e-5

    # print("The rank of P is  : ", np.linalg.matrix_rank(P))
    # print("The rank of Psi is: ", np.linalg.matrix_rank(Psi))

    # find the  solution
    v = np.linalg.pinv(Psi) @ Pi

    # assert that v is in the span of Psi

    assert np.linalg.norm(Psi@v - Pi) <= 1e-5

    K = scipy.linalg.null_space(Psi)
    # print("The solution of mu (V_{T-1}) is: ", v[-grid_size**2:])
    print("The dim of kernel of Psi is: ", K.shape[1])

    print(K.shape)
    if K.shape[1] > 1:
        print("Not Strongly Identifiable in the Original Space.")
    if scipy.linalg.null_space(Psi).shape[1] <= 1:
        print("The min of kernel projection is:", min(scipy.linalg.null_space(Psi)[-grid_size**2:]))
        print("The max of kernel projection is:", max(scipy.linalg.null_space(Psi)[-grid_size**2:]))

    print("Testing with Features ...")
    # construct K_eta from equation (7) in the paper
    K_eta = K[-gw.n_states:,:]
    print(K_eta.shape)
    # find the intersection as in equation (18)     
    A = E@K_eta
    B = F
    
    # find their intersection
    M = np.hstack((A,-B))

    V = scipy.linalg.null_space(M)
    print(V.shape)
    # assert V.shape[1] == 1

    V_p = V[:A.shape[1]]

    intersection = A@V_p

    # assert that the intersecttion is ran(1)
    assert np.abs(min(intersection) - max(intersection)) <= 1e-6
    print("Strongly Identifiable with Features.")


    # print(gw.reachable_tube())