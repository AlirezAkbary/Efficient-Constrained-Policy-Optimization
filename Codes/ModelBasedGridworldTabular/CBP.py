from tabularCMDPEnv import TabularCMDP
from policy import *
from helper_utlities import get_lambd_upper_bound, get_lower_upper_limit_lambd, get_optimal_pol_and_perf, save_data, \
    update_store_info
import os
import argparse

"""
Coin Betting implementation for parameter update of simplex (policy) and d-dim vector (lambda, dual variable)
"""


class CB_Primal:
    """
    Paper: Coin Betting and Parameter-Free Online Learning, Orabona and Pal [2016] (https://arxiv.org/abs/1602.04128)
    Creates a CB algorithm to update the simplex policy using Learning with Expert Advice (LEA) based on KT potential,
    Algorithm 2, Pg 8
    """

    def __init__(self, env, init_policy_prob):
        self.t = 0
        S = env.S
        A = env.A
        self.init_policy_prob = init_policy_prob
        self.curr_policy_prob = init_policy_prob  # [S x A] dim
        self.sum_g = np.zeros((S, A))  # sum of gradients for pi update
        self.sum_prod_g_w = np.ones((S, A))  # \sum_t(g_t . w_t): sum of reward over time for policy
        self.w_t = np.ones((S, A))

    def update(self, g_bar):
        """
        Parameter:
               g_bar: Q_l (state-action value function), here g_bar is normalized between 0 and 1.
               Here, g_bar is (- gradient of primal regret on policy).
               R(\pi*, T) = \sum_t \sum_s <\pi*(.|s) - \pi(.|s), Q_l^t>, where Q_l^t = \hat Q_r^t + \lambda \hat Q_c^t.
        """
        self.t += 1
        g_tilde = self.get_g_tilde(g_bar, self.curr_policy_prob, self.w_t)
        self.sum_g += g_tilde
        self.sum_prod_g_w += np.multiply(g_tilde, self.w_t)
        self.w_t = np.multiply(self.sum_g, self.sum_prod_g_w) / self.t
        p_t = np.multiply(self.init_policy_prob, np.maximum(0, self.w_t))
        self.curr_policy_prob = self.get_normalized_p(p_t, self.init_policy_prob)

    def get_g_tilde(self, g_t, p_t, w_t):
        """
        Implements Line 6 in Algorithm 2.
        Parameter:
            g_t: bounded gradients for policy which is <= 1
            p_t: policy pi_t
            w_t: w_t
        Returns:
            g_tilde
        Here,
        g_tilde = g_{t,i} - <g_t, p_t> if w_{t,i}>0
        g_tilde = max(0,g_{t,i} - <g_t, p_t>) if w_{t,i}<=0
        """
        g_tilde = g_t - np.einsum('sa,sa->s', g_t, p_t)[:, None]
        x, y = np.where(w_t <= 0)
        for i in range(len(x)):
            g_tilde[x[i], y[i]] = np.maximum(0, g_tilde[x[i], y[i]])
        return g_tilde

    def get_normalized_p(self, p_hat, initial_dist):
        """
        Parameter:
            p_hat: [S X A] shape
            initial_dist = p_0
        Returns:
            p_mat = p_hat / ||p_hat||_1
        """
        norm_p = np.linalg.norm(p_hat, ord=1, axis=1)
        p_mat = np.zeros((initial_dist.shape))
        for s in range(len(norm_p)):
            if norm_p[s] == 0:
                p_mat[s] = initial_dist[s]
            else:
                p_mat[s] = np.divide(p_hat[s], norm_p[s])
        return p_mat


class CB_Dual:
    """
    Paper: Training Deep Networks without Learning Rates Through Coin Betting,
     Orabona [2016] (https://arxiv.org/pdf/1705.07795.pdf)
    Creates a CB algorithm to update the dual variable \lambda using COCOB-backprop, Algorithm 2, Pg 6
    """

    def __init__(self, num_constraints, alpha, lower_limit_lambd, upper_limit_lambd):
        self.lower_limit_lambd = lower_limit_lambd
        self.upper_limit_lambd = upper_limit_lambd
        self.num_constraints = num_constraints
        self.curr_lambd = self.get_projected_lambd(np.ones(num_constraints))  # lambda intialised to 1
        self.initial_lambd = self.curr_lambd
        self.alpha_lambd = alpha
        self.G_lambd = np.zeros(num_constraints)  # G_t = G_{t-1} + |g_t|, where G_0 = L
        self.L_lambd = np.zeros(num_constraints)  # L_t = max(L_{t−1}, |g_t|)
        self.reward_lambd = np.zeros(num_constraints)  # Reward_t = Reward_{t-1} + (w_t - w_1)g_t, where R_0 = 0
        self.theta_lambd = np.zeros(num_constraints)  # theta_{t} = theta_{t-1} + g_t, where theta_0 = 0
        self.w_lambd = np.ones(num_constraints)  # w_t = w_1 + \beta(L + Reward_t), where w_1 = projected(1)

    def update(self, g):
        """
        Implements COCOB Algorithm 2 for updating lambda variable a d-dim vector
        Parameter:
            g: negative value of un-normalized gradient of the regret for dual variable
            Here,
            R(\lambda^*, T) = \sum_t <\lambda_t - \lambda^*, (\hat V_c - b)>,
            where V_c is cost value function and b is threshold for cost constraint.
        """
        self.L_lambd = np.maximum(self.L_lambd, np.abs(g))
        self.G_lambd += np.abs(g)
        self.reward_lambd = np.maximum(self.reward_lambd + (self.w_lambd - self.initial_lambd) * g, 0)
        self.theta_lambd += g
        beta = self.theta_lambd / (
                self.L_lambd * np.maximum(self.G_lambd + self.L_lambd, self.alpha_lambd * self.L_lambd))
        self.curr_lambd = self.get_projected_lambd(self.initial_lambd + beta * (self.L_lambd + self.reward_lambd))

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
    gamma = 0.9  # fixing gamma for experiment [CHANGE LATER]
    return (1 + np.sum(ul_lambd)) / (1 - gamma)


def run_CB_agent(cmdp, optimal_performance, num_iterations, output_dir, ub_lambd, moving_avg_window,
                 full_average, initial_policy, alpha_lambd, update_dual_variable=True):
    # initialize storing data information
    og_list, avg_og_list, avg_cv_list, cv_list = [], [], [], []
    lambd_list, avg_lambd_list, V_r_list, V_g_list, policy_list, Q_l_list = [], [], [], [], [], []

    curr_policy = initial_policy
    ll_lambd, ul_lambd = get_lower_upper_limit_lambd(cmdp.num_constraints, ub_lambd, update_dual_variable)
    # to normalize the gradient of policy, we need max Q value
    grad_pol_max = get_max_grad_policy(cmdp.gamma, ul_lambd)

    # CB Primal: to update the policy
    primal = CB_Primal(cmdp, initial_policy.policy_prob)

    # CB Dual: to update the dual variable lambd
    # [NOTE]: if mdp is passed then it creates a dummy dual object
    dual = CB_Dual(cmdp.num_constraints, alpha_lambd, ll_lambd, ul_lambd)

    for t in range(num_iterations):
        print("Iteration:", t)
        
        curr_policy.set_policy_prob(primal.curr_policy_prob)
        Q_r = curr_policy.get_Q_function(cmdp.R)
        Q_l = Q_r
        V_r = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_r))
        V_r_list.append(V_r)

        if update_dual_variable:
            curr_lambd = dual.curr_lambd
            lambd_list.append(curr_lambd)
            V_g_rho = np.zeros(cmdp.num_constraints)
            for c in range(cmdp.num_constraints):
                Q_c = curr_policy.get_Q_function(cmdp.G[c])
                V_g_rho[c] = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_c))
                Q_l += curr_lambd[c] * Q_c
            V_g_list.append(V_g_rho)
            # update the dual variable lambda by passing the gradient of regret of lambda
            grad_lambd = -(V_g_rho - cmdp.b)
            dual.update(grad_lambd)
            cv = cmdp.b - V_g_rho
            cv_list.append(cv)

        # update the primal policy by passing normalized gradient
        primal.update(Q_l / grad_pol_max)
        og = optimal_performance - V_r
        og_list.append(og)
        policy_list.append(curr_policy.policy_prob)
        Q_l_list.append(Q_l)
        print(og)
        print(cv)
        update_store_info(optimal_performance, V_r_list, V_g_list, cmdp.b, lambd_list, moving_avg_window,
                          t, update_dual_variable, full_average, avg_og_list, avg_cv_list, avg_lambd_list)
        # if t % 100 == 0:
        #     # saving result after every 100 iterations
        #     save_data(output_dir, np.asarray(og_list), np.asarray(cv_list),
        #               np.asarray(avg_og_list), np.asarray(avg_cv_list),
        #               np.asarray(curr_policy.policy_prob),
        #               np.asarray(lambd_list), np.asarray(avg_lambd_list), np.asarray(policy_list), np.asarray(Q_l_list))
    return og_list, cv_list, avg_og_list, avg_cv_list
    # save_data(output_dir, np.asarray(og_list), np.asarray(cv_list),
    #           np.asarray(avg_og_list), np.asarray(avg_cv_list),
    #           np.asarray(curr_policy.policy_prob),
    #           np.asarray(lambd_list), np.asarray(avg_lambd_list), np.asarray(policy_list), np.asarray(Q_l_list))


if __name__ == '__main__':
    try_locally = True  # set True to run code locally, see "save_dir_loc" for setting save location of Results
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
    parser.add_argument('--alpha_lambd', help="alpha COCOB", type=int, default=8)

    # changing MDP by selecting discount factor
    parser.add_argument('--gamma', help="discount factor", type=float, default=0.9)

    # changing MDP by selecting discount factor
    parser.add_argument('--b_thresh', help="cost threshold", type=float, default=1.5)

    args = parser.parse_args()
    if try_locally:
        save_dir_loc = "../"
    else:
        save_dir_loc = "/network/scratch/j/jainarus/CMDPData/"

    # outer_file_name = "Iter" + str(args.num_iterations) + "_alpha" + str(args.alpha_lambd) + "_MC" + str(
    #     int(args.multiple_constraints)) + "_FullAvg" + str(int(args.full_average)) + "_gam" + str(
    #     args.gamma) + "_b" + str(args.b_thresh)

    # fold_name = os.path.join(save_dir_loc, "Results/Tabular/CBP/ModelBased")
    # output_dir_name = os.path.join(fold_name, outer_file_name)
    # inner_file_name = "R" + str(args.run)
    # output_dir = os.path.join(output_dir_name, inner_file_name)
    # args.num_iterations = int(args.num_iterations)
    # load_data_dir = output_dir

    args.run = int(args.run)
    args.num_iterations = int(args.num_iterations)

    current_dir = os.path.join(os.getcwd(), "Results/CBP_ModelBased/")
    param_dir = "R" + str(args.run) + "_iter" + str(args.num_iterations) + "_alphalam" + str(args.alpha_lambd)\
        +"_gamma" + str(args.gamma)


    output_dir = os.path.join(current_dir, param_dir)

    args.multiple_constraints = bool(args.multiple_constraints)
    args.full_average = bool(args.full_average)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # creates a cmdp/mdp based on arguments
    if not args.cmdp:
        # creates a mdp without constraints
        cmdp_env = TabularCMDP(add_constraints=False)
        cmdp_env.gamma = args.gamma
        lambd_star_upper_bound = 0
        _, optimal_perf, _ = get_optimal_pol_and_perf(cmdp_env, constraint=False)
    else:
        # creates a cmdp
        cmdp_env = TabularCMDP(add_constraints=True, multiple_constraints=args.multiple_constraints)
        cmdp_env.change_mdp(args.gamma, b=[args.b_thresh])
        lambd_star_upper_bound = get_lambd_upper_bound(args.multiple_constraints, cmdp_env.gamma, cmdp_env.b)
        optimal_policy, optimal_perf, _ = get_optimal_pol_and_perf(cmdp_env, constraint=True)

    seed_val = 0
    rng = np.random.RandomState(seed_val + args.run)
    initial_policy = Policy(cmdp_env, rng)
    run_agent_params = {'cmdp': cmdp_env,
                        'optimal_performance': optimal_perf,
                        'num_iterations': args.num_iterations,
                        'output_dir': output_dir,
                        'ub_lambd': np.asarray(lambd_star_upper_bound),
                        'moving_avg_window': args.moving_avg_window,
                        'full_average': args.full_average,
                        'initial_policy': initial_policy,
                        'alpha_lambd': args.alpha_lambd,
                        'update_dual_variable': bool(args.cmdp)
                        }
    og, cv, avg_og, avg_cv = run_CB_agent(**run_agent_params)
    np.save(output_dir+"/og.npy", og)
    np.save(output_dir+"/cv.npy", cv)
    np.save(output_dir+"/avgog.npy", avg_og)
    np.save(output_dir+"/avgcv.npy", avg_cv)