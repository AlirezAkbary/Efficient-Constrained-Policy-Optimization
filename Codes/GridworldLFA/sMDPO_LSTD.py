from tabularCMDPEnv import TabularCMDP
from policy import *
from helper_utlities import *
from LFAUtils.feature import TileCodingFeatures, TabularTileCoding, CartPoleTileCoding
from LFAUtils.LSTD import LSTD
from LFAUtils.sampler import *
import os
import argparse
import time
from datetime import timedelta
import copy
from primal import FMAPGPrimal
from scipy.stats import entropy

# MDPO policy gradient method
# surrogate: J(\pi(\theta_t)) + <\pi(\theta) - \pi(\theta_t), \nabla_{\pi} J(\pi(\theta_t))> -1/(\eta) * D_{\phi}(\pi(\theta), \pi(\theta_t))
# bregman divergence: D_{\phi} (a, b) = \phi(a) - \phi(b) - <\nabla \phi(b), a - b>
# policy representation \pi : softmax = z^{\pi}(a, s)
# mirror map \phi(z) : logsumexp  = \Sigma_{s} d^{\pi_t}(s) \log \Simga_{a} exp(z^{\pi}(a, s))



class LFAsMDPOPrimal(FMAPGPrimal):

    def __init__(self, env, feature, num_cons, eta, alpha, m, init_theta, gamma):
        super().__init__(env, num_cons, eta, alpha, m)

        self.gamma = gamma
        
        # feature object
        self.feature_obj = feature

        # representation is softmax and parametrization Tabular -> init_theta is |S*A| matrix with logits
        self.init_theta = copy.deepcopy(init_theta)

        # current theta
        self.current_theta = copy.deepcopy(init_theta)

        self.features = self.construct_feature(self.feature_obj.get_feature_size())


    def get_objective():
        pass

    def update_policy_param(self, ppi_t, dpi_t, Advpi_t):
        omega = copy.deepcopy(self.current_theta)
        updated_omega = self.update_inner_weight(omega, ppi_t, dpi_t, Advpi_t)
        self.current_theta = copy.deepcopy(updated_omega)

    def update_inner_weight(self, omega, ppi_t, dpi_t, Advpi_t):

        for i in range(self.m):
            update_direction = np.einsum('sa, sad->sad', dpi_t.reshape(-1, 1) * ppi_t * (Advpi_t + 1/self.eta), self.grad_prob_wrt_theta(omega)).sum(1).sum(0)
            omega = omega + self.alpha * update_direction
        return omega
        

    # given logits S*A, returns probability matrix S*A
    def get_policy_prob(self, x):
        e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
        out = e_x / e_x.sum(1).reshape(-1, 1)
        return out
    
    # given theta and features returns logits
    def get_logit(self, theta):
 
        logits = np.einsum('d, sad->sa', theta, self.features)
        return logits
    
    # returns feature tensor (s, a, d)
    def construct_feature(self, d):
        num_s, num_a = self.env.S, self.env.A
        features = np.zeros((num_s, num_a, d))
        for s in range(num_s):
            for a in range(num_a):
                tmp_feature = np.zeros(d)
                tmp_feature[self.feature_obj.get_feature([s], a)] = 1
                features[s, a, :] = tmp_feature
        return features

    def grad_prob_wrt_theta(self, omega):

        # grad_prob_wrt_theta (for state x and action b) = \phi(b, x) - \sigma_{a}p(a|x)\phi(a, x)

        pi = self.get_policy_prob(self.get_logit(omega))

        # average feature |S*d|
        avg_feature = np.einsum('sad,sa->sd',self.features, pi)
        
        # |S*A*d|
        grad_prob_wrt_theta = self.features - avg_feature[:, np.newaxis, :]
        
        # print(grad_prob_wrt_theta /grad_prob_wrt_theta_sum[:, :, np.newaxis])
        return grad_prob_wrt_theta
        

class Dual:

    def __init__(self, num_constraints, eta, lower_limit_lambd, upper_limit_lambd):
        
        self.lower_limit_lambd = lower_limit_lambd
        
        self.upper_limit_lambd = upper_limit_lambd
        
        self.num_constraints = num_constraints

        self.curr_lambd = self.get_projected_lambd(np.ones(num_constraints))  # lambda intialised to 1
        
        self.initial_lambd = self.curr_lambd
        
        self.eta = eta
        

    def update(self, grad):
        self.curr_lambd = self.get_projected_lambd(self.curr_lambd - self.eta * grad)

    def get_projected_lambd(self, lambd):
        return np.maximum(self.lower_limit_lambd, np.minimum(self.upper_limit_lambd, lambd))


# given a primal object, return probability representation of its policy
def get_policy_for_all_states(primal, S, A, tc_args):
    # helper function to get policy for all states
    policy_prob = np.zeros((S, A))
    for s in range(S):
        new_s_rep = tc_args.get_state_representation(s)
        policy_prob[s] = primal.policy_prob(new_s_rep)
    return policy_prob


def run_agent(cmdp, optimal_performance, num_iterations, eta, alpha_policy, m, ub_lambd, initial_theta,
                 initial_policy, alpha_lambd, nsamples, feature, tc_args, update_dual_variable=True):
    """
    Parameter:
        initial policy : Policy class
    """            
    # For storing optimality gap and constriant violation
    og_list, cv_list, avg_og_list, avg_cv_list = [], [], [], []

    # For storing primal and duals
    # Why we need avg_lambda list ?
    lambd_list, avg_lambd_list, policy_list = [], [], []
    
    # For storing Q_l = Q_r + \gamma Q_c
    Q_l_list = []
    Q_l_hat_list = []

    A_l_list = []
    A_l_hat_list = []

    #J_r = E[V_r(\pho)]
    #J_c = E[V_c(\pho)]
    J_r_list, J_c_list, J_r_hat_list, J_c_hat_list = [], [], [], []
    
    # Storing Q function estimation error 
    diff_in_Q_l, diff_in_Q_r, diff_in_Q_c = [], [], []

    # Policy class is used to obtain true state (action) value functions
    curr_policy = initial_policy
    
    # set lower limit and upper limit for lambda
    ll_lambd, ul_lambd = get_lower_upper_limit_lambd(cmdp.num_constraints, ub_lambd, update_dual_variable)


    # initialize primal object used to update policy parameters
    primal = LFAsMDPOPrimal(cmdp, feature, cmdp.num_constraints, eta, alpha_policy, m, initial_theta, cmdp.gamma)

    # initiliaze dual object to update lambda
    # [NOTE]: if mdp is passed then it creates a dummy dual object
    dual = Dual(cmdp.num_constraints, alpha_lambd, ll_lambd, ul_lambd)

    # TD Q estimator
    #q_estimator = TD(cmdp, nsamples, cmdp.num_constraints, update_dual_variable)

    data_sampler = Sampling(cmdp, nsamples, tc_args, update_dual_variable)

    # initialize Q function estimator; uses collected data from sampler
    q_estimator = LSTD(feature.get_feature_size(), feature, cmdp, data_sampler)

    for t in range(num_iterations):

        # Primal phase
        print("Iteration:", t)

        current_theta = primal.current_theta
        current_policy_prob = primal.get_policy_prob(primal.get_logit(current_theta))
        curr_policy.set_policy_prob(current_policy_prob)

        # set true state(action) value function (reward)
        Q_r = curr_policy.get_Q_function(cmdp.R)
        V_r = np.einsum('sa,sa->s', curr_policy.policy_prob, Q_r)
        A_r = Q_r - V_r.reshape(-1, 1)
        J_r = np.dot(cmdp.p0, V_r)
        J_r_list.append(J_r)
        # We will add \gamma * Q_c to it in the dual phase later
        Q_l = copy.deepcopy(Q_r)
        A_l = copy.deepcopy(A_r)

        # runs LSTD object to get weights for Q estimation
        q_estimator.run_solver(current_policy_prob)

        # set estimated Q function for reward and constrainsts under current policy
        q_hat = np.asarray([q_estimator.get_estimated_Q(tc_args.get_state_representation(s)) for s in
                            range(cmdp.S)])  # dim [S X A X num_constraints+1]

        # set estimated state(action) value function (reward)
        Q_r_hat = q_hat[:, :, 0]
        V_r_hat = np.einsum('sa,sa->s', curr_policy.policy_prob, Q_r_hat)
        A_r_hat = Q_r_hat - V_r_hat.reshape(-1, 1)
        J_r_hat = np.dot(cmdp.p0, V_r_hat)
        J_r_hat_list.append(J_r_hat)
        # We will add \gamma * Q_c_hat to it in dual pahse later
        Q_l_hat = copy.deepcopy(Q_r_hat)
        A_l_hat = copy.deepcopy(A_r_hat)

        # store Q_r estimtion error
        diff_in_Q_r.append(q_estimator.get_error_norm(Q_r, Q_r_hat))

        # Dual phase
        curr_lambd = dual.curr_lambd
        lambd_list.append(curr_lambd)


        J_c, J_c_hat = np.zeros(cmdp.num_constraints), np.zeros(cmdp.num_constraints)
        error_in_cost_constraint = 0


        for c in range(cmdp.num_constraints):
            # set true state (action) value function (constraint)
            Q_c = curr_policy.get_Q_function(cmdp.G[c])
            V_c = np.einsum('sa,sa->s', curr_policy.policy_prob, Q_c)
            A_c = Q_c - V_c.reshape(-1, 1)
            J_c[c] = np.dot(cmdp.p0, V_c)
            # Q_l = Q_r + \gamma * Q_c
            Q_l += curr_lambd[c] * Q_c
            A_l += curr_lambd[c] * A_c


            Q_c_hat = q_hat[:, :, c + 1]
            V_c_hat = np.einsum('sa,sa->s', curr_policy.policy_prob, Q_c_hat)
            A_c_hat = Q_c_hat - V_c_hat.reshape(-1, 1)
            J_c_hat[c] = np.dot(cmdp.p0, V_c_hat)
            # Q_l = Q_r + \gamma * Q_c
            Q_l_hat += curr_lambd[c] * Q_c_hat
            A_l_hat += curr_lambd[c] * A_c_hat

            # Overall Q_c estiamtion error
            error_in_cost_constraint += q_estimator.get_error_norm(Q_c, Q_c_hat)

        # store Q_c estimation error
        diff_in_Q_c.append(error_in_cost_constraint / cmdp.num_constraints)

        # store J_c and J_c_hat
        J_c_list.append(J_c)
        J_c_hat_list.append(J_c_hat)


        # store Q_l 
        Q_l_list.append(Q_l)
        # print("Q_l", Q_l)
        # print("Q_l_hat", Q_l_hat)
        Q_l_hat_list.append(Q_l_hat)
        diff_in_Q_l.append(q_estimator.get_error_norm(Q_l, Q_l_hat))
        

        # store A_l
        A_l_list.append(A_l)
        A_l_hat_list.append(A_l_hat)

        

        # update the dual variable lambda by passing the gradient of regret of lambda
        # need to be updated
        grad_lambd = (J_c_hat - cmdp.b)
        dual.update(grad_lambd)

        # cv with true J_c
        cv = cmdp.b - J_c
        
    
        # updating the primal policy
        primal.update_policy_param(current_policy_prob, cmdp.p0, A_l_hat)

        # og with true J_r
        og = optimal_performance - J_r

        policy_list.append(curr_policy.policy_prob)

        # storing optimal gap and constraint violation
        if t % 1 == 0:
            og_list.append(og)
            cv_list.append(cv) 
            if not t:
                avg_cv_list.append(cv)
                avg_og_list.append(og)
            else:
                l = len(avg_cv_list)
                avg_cv_list.append((l*avg_cv_list[-1]+cv) / (l+1))
                avg_og_list.append((l*avg_og_list[-1]+og) / (l+1))
            print(og)
            print(cv)
    
    return og_list, cv_list, avg_og_list, avg_cv_list


if __name__ == '__main__':
    start_main_t = time.time()
    try_locally = True  # parameter to try code locally
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', help="iterations", type=int, default=1500)
    # cmdp: 1 -> creates env with constraints and solve for Coin Betting on both policy and lambda
    # cmdp:0 -> creates a mdp and solve for Coin Betting on policy
    parser.add_argument('--cmdp', help="create a cmdp:1 or mdp:0", type=int, default=1)

    # multiple_constraints: 0 -> Adds only 1 constraint for cmdp
    # multiple_constraints: 1 -> There are 3 constraints for cmdp
    parser.add_argument('--multiple_constraints', help="multiple constraints: 0, 1", type=int, default=0)

    # full_average:1 -> Stores result with average policy from iteration 0 to t
    # full_average:0 -> Would use Moving average with window size selected from next parameter
    parser.add_argument('--full_average', help="Full average: 0, 1", type=int, default=1)

    # stores result with a moving average window over policy for past k iterations
    parser.add_argument('--moving_avg_window', help="window size", type=int, default=200)

    # Run is used to keep track of runs with different policy initialization.
    parser.add_argument('--run', help="run number", type=int, default=1)

    # alpha_lambd: parameter used for updating the lambda variable for cmdp
    parser.add_argument('--alpha_lambd', help="alpha COCOB", type=float, default=1)

    # iht table size
    parser.add_argument('--iht_size', help="iht size", type=float, default=2000)

    # iht number of tiles
    parser.add_argument('--num_tiles', help="num of tiles", type=int, default=1)

    # iht number of tiles
    parser.add_argument('--tiling_size', help="dim of grid", type=int, default=5)

    # num of samples for estimating Q value function
    parser.add_argument('--num_samples', help="num of samples", type=int, default=300)

    parser.add_argument('--gamma', help="discount factor", type=float, default=0.9)

    parser.add_argument('--b_thresh', help="threshold for single constraint", type=float, default=1.5)

    # number of inner loops
    parser.add_argument('--m', help="inner loop number", type=int, default=10)

    # 1/{\eta} is the parameter for divergence term
    parser.add_argument('--eta', help="divergence ", type=float, default=1)

    # step size to update omega 
    parser.add_argument('--alpha_policy', help="policylearning ", type=float, default=1)

    args = parser.parse_args()
    if try_locally:
        save_dir_loc = "./"
    else:
        save_dir_loc = "/network/scratch/j/jainarus/CMDPData/"
    outer_file_name = "iter" + str(args.num_iterations) + "_alphalamb" + str(args.alpha_lambd) + "_iht_size" + str(
        int(args.iht_size)) + "_ntiles" + str(int(args.num_tiles)) + "_tileDim" + str(args.tiling_size) + \
                      "_num_samples" + str(args.num_samples) 

    # fold_name = os.path.join(save_dir_loc, "Results/Tabular/CB/ModelFree/LFA/TileCodingFeatures")
    # output_dir_name = os.path.join(fold_name, outer_file_name)
    # inner_file_name = "R" + str(args.run)
    # output_dir = os.path.join(output_dir_name, inner_file_name)
    args.num_iterations = int(args.num_iterations)
    args.iht_size = int(args.iht_size)
    #load_data_dir = output_dir
    args.multiple_constraints = bool(args.multiple_constraints)
    args.full_average = bool(args.full_average)
    args.run = int(args.run)

    current_dir = os.path.join(os.getcwd(), "Results/sMDPO_LSTDSampling/")
    param_dir = "R" + str(args.run) + "_iter" + str(args.num_iterations) + "_alphalam" + str(args.alpha_lambd)\
        + "_Qnsamples" + str(args.num_samples)  +\
        "_m" + str(args.m) + "_eta" + str(args.eta) + "_alphapol" + str(args.alpha_policy) \
            + "_tilings_size" + str(args.tiling_size) + "_iht_size" + str(args.iht_size)


    output_dir = os.path.join(current_dir, param_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # creates a cmdp/mdp based on arguments
    if not args.cmdp:
        # creates a mdp without constraints
        cmdp_env = TabularCMDP(add_constraints=False)
        lambd_star_upper_bound = 0
        _, optimal_perf, _ = get_optimal_pol_and_perf(cmdp_env, constraint=False)
    else:
        # creates a cmdp
        cmdp_env = TabularCMDP(add_constraints=True, multiple_constraints=args.multiple_constraints)
        # cmdp_env.change_mdp(args.gamma, [args.b_thresh])
        lambd_star_upper_bound = get_lambd_upper_bound(args.multiple_constraints, cmdp_env.gamma, cmdp_env.b)
        _, optimal_perf, _ = get_optimal_pol_and_perf(cmdp_env, constraint=True)
    tabular_tc = TabularTileCoding(args.iht_size, args.num_tiles, args.tiling_size)
    tc_feature = TileCodingFeatures(cmdp_env.A, tabular_tc.get_tile_coding_args())
    cartpole_feature = CartPoleTileCoding()
    
    #print_tc_features(tc_feature, cmdp_env, tabular_tc, output_dir)

    seed_val = 0
    rng = np.random.RandomState(seed_val + args.run)

    initial_policy = Policy(cmdp_env, rng)
    initial_theta = np.random.normal(0, 1, args.iht_size)

    agent_params = {'cmdp': cmdp_env,
                        'optimal_performance': optimal_perf,
                        'num_iterations': args.num_iterations,
                        'eta': args.eta,
                        'alpha_policy': args.alpha_policy,
                        'm': args.m,
                        'ub_lambd': np.asarray(lambd_star_upper_bound),
                        'initial_theta': initial_theta,
                        'initial_policy': initial_policy,
                        'alpha_lambd': args.alpha_lambd,
                        'nsamples': args.num_samples,
                        'feature': tc_feature,
                        'tc_args': tabular_tc
                        }
    og, cv, avg_og, avg_cv = run_agent(**agent_params)
    np.save(output_dir+"/og.npy", og)
    np.save(output_dir+"/cv.npy", cv)
    np.save(output_dir+"/avgog.npy", avg_og)
    np.save(output_dir+"/avgcv.npy", avg_cv)
    end_main_t = time.time()
    time_to_finish = timedelta(seconds=end_main_t - start_main_t)
    time_log(time_to_finish, args.num_iterations, output_dir)
