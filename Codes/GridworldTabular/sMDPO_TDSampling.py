from tabularCMDPEnv import TabularCMDP
from policy import *
from helper_utlities import *
from GridworldTabular.TD import TD
import numpy as np
import os
import argparse
from primal import FMAPGPrimal
import copy
from scipy.stats import entropy

# Parametrization \theta is Tabular; a |S*A| matrix storing logits
# Representation \pi is softmax; 
class TabularsMDPOPrimal(FMAPGPrimal):
    def __init__(self, env, num_cons, eta, alpha, m, init_theta, gamma):
        super().__init__(env, num_cons, eta, alpha, m)

        # representation is softmax and parametrization Tabular -> init_theta is |S*A| matrix with logits
        self.init_theta = copy.deepcopy(init_theta)

        # current theta
        self.current_theta = copy.deepcopy(init_theta)

        self.gamma = gamma


    def get_objective(self, omega, dpi_t, Advpi_t):
        ppi = self.get_policy_prob(omega)
        ppi_t = self.get_policy_prob(self.current_theta)
        obj = (dpi_t * (ppi_t * Advpi_t * log_divide(ppi, ppi_t)).sum(1)).sum()
        reg =  (dpi_t * entropy(ppi_t, ppi, axis=1)).sum()
        return obj - (1/self.eta) * reg
    
    def update_policy_param(self, ppi_t, dpi_t, Advpi_t):
        omega = copy.deepcopy(self.current_theta)
        updated_omega = self.update_inner_weight(omega, ppi_t, dpi_t, Advpi_t)
        self.current_theta = copy.deepcopy(updated_omega)

    def update_inner_weight(self, omega, ppi_t, dpi_t, Advpi_t):
        for i in range(self.m):
            ppi = self.get_policy_prob(omega)
            update_direction = dpi_t.reshape(-1, 1) * ppi_t * Advpi_t - 1/self.eta * dpi_t.reshape(-1, 1) * (ppi - ppi_t)
            #armijo_alpha = self.armijo(omega, update_direction, ppi_t, dpi_t,Advpi_t, self.alpha, 0.5, 1.1)
            omega = omega + self.alpha * update_direction
        return omega
    
    # given logits matrix S*A, returns the probability matrix S*A
    def get_policy_prob(self, x):
        e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
        out = e_x / e_x.sum(1).reshape(-1, 1)
        return out

    def armijo(self, omega, update_direction, ppi_t, dpi_t, Advpi_t, alpha_min, armijo_const, decay_factor):
        current_alpha = alpha_min
        print("first", self.get_objective(omega+current_alpha*update_direction, dpi_t, Advpi_t))
        print("second", self.get_objective(omega, dpi_t, Advpi_t))
        print("gradient", np.linalg.norm(update_direction))
        while self.get_objective(omega+current_alpha*update_direction, dpi_t, Advpi_t) < \
        self.get_objective(omega, dpi_t, Advpi_t) + armijo_const * current_alpha*(np.linalg.norm(update_direction)**2):
            current_alpha *= decay_factor
        print("Armijo:", current_alpha)
        return current_alpha

class Dual:
    """
    Paper: Training Deep Networks without Learning Rates Through Coin Betting,
     Orabona [2016] (https://arxiv.org/abs/1705.07795)
    Creates a CB algorithm to update the dual variable \lambda using COCOB-backprop, Algorithm 2, Pg 6
    """

    def __init__(self, num_constraints, eta, lower_limit_lambd, upper_limit_lambd):
        
        self.lower_limit_lambd = lower_limit_lambd
        
        self.upper_limit_lambd = upper_limit_lambd
        
        self.num_constraints = num_constraints

        self.curr_lambd = self.get_projected_lambd(np.ones(num_constraints))  # lambda intialised to 1
        
        self.initial_lambd = self.curr_lambd
        
        self.eta = eta
        
        # self.alpha_lambd = alpha
        # self.G_lambd = np.zeros(num_constraints)  # G_t = G_{t-1} + |g_t|, where G_0 = L
        # self.L_lambd = np.zeros(num_constraints)  # L_t = max(L_{t−1}, |g_t|)
        # self.reward_lambd = np.zeros(num_constraints)  # Reward_t = Reward_{t-1} + (w_t - w_1)g_t, where R_0 = 0
        # self.theta_lambd = np.zeros(num_constraints)  # theta_{t} = theta_{t-1} + g_t, where theta_0 = 0
        # self.w_lambd = np.ones(num_constraints)  # w_t = w_1 + \beta(L + Reward_t), where w_1 = projected(1)

    def update(self, grad):
        """
        Implements COCOB Algorithm 2 for updating lambda variable a d-dim vector
        Parameter:
            g: negative value of un-normalized gradient of the regret for dual variable
            Here,
            R(\lambda^*, T) = \sum_t <\lambda_t - \lambda^*, (\hat V_c - b)>,
            where V_c is cost value function and b is threshold for cost constraint.
        """
        # self.L_lambd = np.maximum(self.L_lambd, np.abs(g))
        # self.G_lambd += np.abs(g)
        # self.reward_lambd = np.maximum(self.reward_lambd + (self.w_lambd - self.initial_lambd) * g, 0)
        # self.theta_lambd += g
        # beta = self.theta_lambd / (
        #         self.L_lambd * np.maximum(self.G_lambd + self.L_lambd, self.alpha_lambd * self.L_lambd))
        # self.curr_lambd = self.get_projected_lambd(self.initial_lambd + beta * (self.L_lambd + self.reward_lambd))

        self.curr_lambd = self.get_projected_lambd(self.curr_lambd - self.eta * grad)

    def get_projected_lambd(self, lambd):
        """
        Parameter:
            lambd
        Returns:
            projected value of lambd
        Here, lambd \in [lower_limit_lambd, upper_limit_lambd]
        """
        return np.maximum(self.lower_limit_lambd, np.minimum(self.upper_limit_lambd, lambd))


def get_max_grad_policy(gamma, ul_lambd):
    # To normalize the gradient of policy, value of max of grad of policy
    # Note: Assumption here is that 0<= reward, cost < =1.
    # If assumption change then multiply with max reward/cost
    gamma = 0.9  # fixing for now [CHANGE LATER]
    return (1 + np.sum(ul_lambd)) / (1 - gamma)


def run_agent(cmdp, optimal_performance, num_iterations, eta, alpha_policy, m, ub_lambd, initial_theta,
                 initial_policy, alpha_lambd, nsamples, update_dual_variable=True):
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
    primal = TabularsMDPOPrimal(cmdp, cmdp.num_constraints, eta, alpha_policy, m, initial_theta, cmdp.gamma)

    # initiliaze dual object to update lambda
    # [NOTE]: if mdp is passed then it creates a dummy dual object
    dual = Dual(cmdp.num_constraints, alpha_lambd, ll_lambd, ul_lambd)

    # TD Q estimator
    q_estimator = TD(cmdp, nsamples, cmdp.num_constraints, update_dual_variable)

    for t in range(num_iterations):

        # Primal phase
        print("Iteration:", t)

        current_theta = primal.current_theta
        curernt_policy_prob = primal.get_policy_prob(current_theta)
        curr_policy.set_policy_prob(curernt_policy_prob)

        # set true state(action) value function (reward)
        Q_r = curr_policy.get_Q_function(cmdp.R)
        V_r = np.einsum('sa,sa->s', curr_policy.policy_prob, Q_r)
        A_r = Q_r - V_r.reshape(-1, 1)
        J_r = np.dot(cmdp.p0, V_r)
        J_r_list.append(J_r)
        # We will add \gamma * Q_c to it in the dual phase later
        Q_l = copy.deepcopy(Q_r)
        A_l = copy.deepcopy(A_r)


        q_estimator.set_policy_prob(curr_policy.policy_prob)
        q_hat = q_estimator.get_Q() 

        # set estimated state(action) value function (reward)
        Q_r_hat = q_hat[0]
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


            Q_c_hat = q_hat[c + 1]
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
        primal.update_policy_param(curernt_policy_prob, cmdp.p0, A_l_hat)

        # og with true J_r
        og = optimal_performance - J_r

        policy_list.append(curr_policy.policy_prob)

        # storing optimal gap and constraint violation
        #if t % 10 == 0:
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

    parser = argparse.ArgumentParser()
    # number of iterations
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

    # alpha_lambd: step size to update dual
    parser.add_argument('--alpha_lambd', help="alpha COCOB", type=float, default=0.1)
    # -------
    # num of samples for estimating Q value function
    parser.add_argument('--num_samples', help="num of samples", type=int, default=1000)

    # changing MDP by selecting discount factor
    parser.add_argument('--gamma', help="discount factor", type=float, default=0.9)

    # threshold for the constraint 
    parser.add_argument('--b_thresh', help="threshold for single constraint", type=float, default=1.5)

    # number of inner loops
    parser.add_argument('--m', help="inner loop number", type=int, default=10)

    # 1/{\eta} is the parameter for divergence term
    parser.add_argument('--eta', help="divergence ", type=float, default=0.1)

    # step size to update omega 
    parser.add_argument('--alpha_policy', help="policylearning ", type=float, default=1)


    args = parser.parse_args()
    
    args.multiple_constraints = bool(args.multiple_constraints)
    args.full_average = bool(args.full_average)
    args.num_iterations = int(args.num_iterations)
    args.run = int(args.run)

    current_dir = os.path.join(os.getcwd(), "Results/sMDPO_TDSampling/")
    param_dir = "R" + str(args.run) + "_iter" + str(args.num_iterations) + "_alphalam" + str(args.alpha_lambd)\
        + "_Qnsamples" + str(args.num_samples)  +\
        "_m" + str(args.m) + "_eta" + str(args.eta) + "_alphapol" + str(args.alpha_policy)


    output_dir = os.path.join(current_dir, param_dir)

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
        cmdp_env.change_mdp(args.gamma, [args.b_thresh])
        lambd_star_upper_bound = get_lambd_upper_bound(args.multiple_constraints, cmdp_env.gamma, cmdp_env.b)
        optimal_policy, optimal_perf, _ = get_optimal_pol_and_perf(cmdp_env, constraint=True)

    seed_val = 0
    rng = np.random.RandomState(seed_val + args.run)
    initial_policy = Policy(cmdp_env, rng)
    initial_theta = np.zeros((cmdp_env.S, cmdp_env.A))
    run_agent_params = {'cmdp': cmdp_env,
                        'optimal_performance': optimal_perf,
                        'num_iterations': args.num_iterations,
                        'eta': args.eta,
                        'alpha_policy': args.alpha_policy,
                        'm': args.m,
                        'ub_lambd': np.asarray(lambd_star_upper_bound),
                        'initial_theta': initial_theta,
                        'initial_policy': initial_policy,
                        'alpha_lambd': args.alpha_lambd,
                        'nsamples': args.num_samples
                        }
    og, cv, avg_og, avg_cv = run_agent(**run_agent_params)
    np.save(output_dir+"/og.npy", og)
    np.save(output_dir+"/cv.npy", cv)
    np.save(output_dir+"/avgog.npy", avg_og)
    np.save(output_dir+"/avgcv.npy", avg_cv)


