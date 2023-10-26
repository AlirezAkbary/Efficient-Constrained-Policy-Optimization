from CartPoleLFA.helper_utils import *
from LFAUtils.feature import TileCodingFeatures, CartPoleTileCoding
from LFAUtils.LSTD import LSTD
from LFAUtils.sampler import *
import os
import argparse
import time
from datetime import timedelta
import copy
from primal import FMAPGPrimal
from CartPoleLFA.Cartpole import *



class LFAsMDPOPrimal(FMAPGPrimal):

    def __init__(self, env, feature, num_cons, eta, alpha, m, init_theta):
        super().__init__(env, num_cons, eta, alpha, m)

        # feature object
        self.feature_obj = feature

        # representation is softmax and parametrization Tabular -> init_theta is |S*A| matrix with logits
        self.init_theta = copy.deepcopy(init_theta)

        # current theta
        self.current_theta = copy.deepcopy(init_theta)

        self.data = None

    def get_objective():
        pass

    def update_policy_param(self, Advpi_t):
        mu_t = np.zeros((len(self.data), self.env.A))
        for i in range(len(self.data)):
            sample = self.data[i]
            mu_t[i, sample.a] = 1/len(self.data)
        omega = copy.deepcopy(self.current_theta)
        updated_omega = self.update_inner_weight(omega, mu_t, Advpi_t)
        self.current_theta = copy.deepcopy(updated_omega)

    def update_inner_weight(self, omega, mu_t, Advpi_t):
        for i in range(self.m):
            update_direction = np.einsum('sa, sad->sad', mu_t * (Advpi_t + 1/self.eta), self.grad_prob_wrt_theta(omega)).sum(1).sum(0)
            omega = omega + self.alpha * update_direction
        return omega
        

    # given logits S*A, returns probability matrix S*A
    def get_policy_prob(self, x):
        e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
        out = e_x / e_x.sum(1).reshape(-1, 1)
        return out
    
    def get_logit(self, theta):
        z = np.zeros((len(self.data), self.env.A))
        for i in range(len(self.data)):
            sample = self.data[i]
            for a in range(self.env.A):
                phi = self.one_hot(self.feature_obj.get_feature_size(), self.feature_obj.get_feature(sample.s, a))
                z[i, a] = np.einsum('d,d', theta, phi)
        return z

    def grad_prob_wrt_theta(self, omega):

        # grad_prob_wrt_theta (for state x and action b) = \phi(b, x) - \sigma_{a}p(a|x)\phi(a, x)
        d = self.feature_obj.get_feature_size()
        ppi = self.get_policy_prob(self.get_logit(omega))
        features = np.zeros((len(self.data), self.env.A, d))
        for i in range(len(self.data)):
            sample = self.data[i]
            for a in range(self.env.A):
                features[i, a, :] = self.one_hot(d, self.feature_obj.get_feature(sample.s, a))
        avg_feature = np.einsum('SAd, SA->Sd', features, ppi)
        return features - avg_feature[:, np.newaxis, :]

    
    def adv_funcion(self, q_estimator):
        # |S*A*num_contraint+1|
        Q = np.zeros((len(self.data), self.env.A, self.num_cons+1))
        print(Q.shape)
        for i in range(len(self.data)):
            sample = self.data[i]
            Q[i, :, :] = q_estimator.get_estimated_Q(sample.s)

        # S*A
        ppi_t = self.get_policy_prob(self.get_logit(self.current_theta))

        V = np.einsum('SAn,SA->Sn', Q, ppi_t)
        A = Q - V[:, np.newaxis, :]
        return A

    def set_data(self, data):
        self.data = data

    # only for compatibility with LSTD implementation
    def policy_prob(self, state):
        d = self.feature_obj.get_feature_size()
        logits = []
        for a in range(self.env.A):
            feature = self.one_hot(d, self.feature_obj.get_feature(state, a))
            logits.append(np.einsum('d,d', feature, self.current_theta))
        e_logits = np.exp(logits - np.max(logits))
        prob = e_logits/e_logits.sum(0)
        return prob

    def one_hot(self, d, feature):
        tmp_feature = np.zeros(d)
        tmp_feature[feature] = 1
        return tmp_feature

             

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

def run_agent(cmdp, num_iterations, eta, alpha_policy, m, ub_lambd, initial_theta,
                  alpha_lambd, nsamples, feature, seed,update_dual_variable=True):
    """
    Parameter:
        initial policy : Policy class
    """            
    # For storing optimality gap and constriant violation
    cv_list, avg_cv_list = [], []

    # For storing primal and duals
    # Why we need avg_lambda list ?
    lambd_list, avg_lambd_list, policy_list = [], [], []


    A_l_hat_list = []

    #J_r = E[V_r(\pho)]
    #J_c = E[V_c(\pho)]
    J_r_hat_list, J_c_hat_list = [], []
    
    # set lower limit and upper limit for lambda
    ll_lambd, ul_lambd = get_lower_upper_limit_lambd(cmdp._num_constraints, ub_lambd, update_dual_variable)


    # initialize primal object used to update policy parameters
    primal = LFAsMDPOPrimal(cmdp, feature, cmdp._num_constraints, eta, alpha_policy, m, initial_theta)

    # initiliaze dual object to update lambda
    # [NOTE]: if mdp is passed then it creates a dummy dual object
    dual = Dual(cmdp._num_constraints, alpha_lambd, ll_lambd, ul_lambd)

    # TD Q estimator
    #q_estimator = TD(cmdp, nsamples, cmdp.num_constraints, update_dual_variable)

    data_sampler = LFASampling(cmdp, cmdp._num_constraints, nsamples, seed, update_dual_variable)

    # initialize Q function estimator; uses collected data from sampler
    q_estimator = LSTD(feature.get_feature_size(), feature, cmdp, data_sampler)

    for t in range(num_iterations):

        # Primal phase
        print("Iteration:", t)


        # runs LSTD object to get weights for Q estimation
        q_estimator.run_solver(primal)
        data = q_estimator.get_data()
        primal.set_data(data)
        A_hat = primal.adv_funcion(q_estimator)
        A_r_hat = copy.deepcopy(A_hat[:, :, 0])
        A_l_hat = copy.deepcopy(A_r_hat)

        # Dual phase
        curr_lambd = dual.curr_lambd
        lambd_list.append(curr_lambd)

        for c in range(cmdp._num_constraints):
            A_l_hat += curr_lambd[c] * A_hat[:, :, c+1]

        J_c_hat = data_sampler.get_V_c()
        J_c_hat_list.append(J_c_hat)


        A_l_hat_list.append(A_l_hat)

    
        # update the dual variable lambda by passing the gradient of regret of lambda
        # need to be updated
        grad_lambd = (J_c_hat - cmdp.b)
        dual.update(grad_lambd)

        # cv with true J_c
        cv = cmdp.b - J_c_hat
        cv_list.append(cv)
            
        # updating the primal policy
        primal.update_policy_param(A_l_hat)

        policy_list.append(primal.current_theta)

        ret = data_sampler.get_V_r()
        J_r_hat_list.append(ret)

        print(ret)
        print(cv)




    
    return J_r_hat_list, cv_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', help="iterations", type=int, default=150)
    # cmdp: 1 -> creates env with constraints and solve for Coin Betting on both policy and lambda
    # cmdp:0 -> creates a mdp and solve for Coin Betting on policy
    parser.add_argument('--cmdp', help="create a cmdp:1 or mdp:0", type=int, default=1)

    # Run is used to keep track of runs with different policy initialization.
    parser.add_argument('--run', help="run number", type=int, default=1)

    # alpha_lambd: parameter used for updating the lambda variable for cmdp
    parser.add_argument('--alpha_lambd', help="alpha COCOB", type=float, default=1)

    parser.add_argument('--num_tilings', type=int, default=8)  # iht number of tiles

    # num of samples for estimating Q value function
    parser.add_argument('--num_samples', help="num of samples", type=int, default=20)

    # entropy coefficient
    parser.add_argument('--entropy_coeff', help="entropy coefficient", type=float, default=5e-3)

    # number of inner loops
    parser.add_argument('--m', help="inner loop number", type=int, default=50)

    # 1/{\eta} is the parameter for divergence term
    parser.add_argument('--eta', help="divergence ", type=float, default=0.1)

    # step size to update omega 
    parser.add_argument('--alpha_policy', help="policylearning ", type=float, default=1)

    args = parser.parse_args()
    seed_val = int(args.run)
    cmdp_env = CartPoleEnvironment()
    cp_tc = CartPoleTileCoding(num_tilings=args.num_tilings)
    tc_feature = TileCodingFeatures(cmdp_env.A, cp_tc.get_tile_coding_args())
    initial_theta = np.random.normal(0, 1, tc_feature.get_feature_size())

    ubl = entropy_to_ubl_dict_cartpole(float(args.entropy_coeff))
    if not args.cmdp:
        # creates a mdp without constraints
        lambd_star_upper_bound = 0
    else:
        # creates a cmdp
        lambd_star_upper_bound = get_lambd_upper_bound(ubl)

    run_agent_params = {'cmdp': cmdp_env,
                        'num_iterations': args.num_iterations,
                        'alpha_policy': args.alpha_policy,
                        'eta': args.eta,
                        'm': args.m,
                        'ub_lambd': np.asarray(lambd_star_upper_bound),
                        'initial_theta': initial_theta,
                        'alpha_lambd': args.alpha_lambd,
                        'nsamples': args.num_samples,
                        'feature': tc_feature,
                        'seed': seed_val,
                        'update_dual_variable': bool(args.cmdp)
                        }
    
    ret, cv = run_agent(**run_agent_params)
    current_dir = os.path.join(os.getcwd(), "Results/sMDPO/")
    args.run = int(args.run)
    param_dir = "R" + str(args.run) + "_iter" + str(args.num_iterations) + "_alphalam" + str(args.alpha_lambd)\
        + "_Qnsamples" + str(args.num_samples) + \
        "_m" + str(args.m) + "_eta" + str(args.eta) + "_alphapol" + str(args.alpha_policy)
    output_dir = os.path.join(current_dir, param_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(output_dir+"/ret.npy", ret)
    np.save(output_dir+"/cv1.npy", np.asarray(cv)[:, 0])
    np.save(output_dir+"/cv2.npy", np.asarray(cv)[:, 1])
