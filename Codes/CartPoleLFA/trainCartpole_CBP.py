from CartPoleLFA.CBP import CB_Dual, CB_Primal
from LFAUtils.feature import CartPoleTileCoding, TileCodingFeatures
from LFAUtils.LSTD import LSTD
from LFAUtils.sampler import LFASampling
from CartPoleLFA.helper_utils import *
import os
import argparse
from CartPoleLFA.Cartpole import *
import time
from datetime import timedelta


def run_CB_agent(cmdp, num_iterations, ub_lambd, alpha_lambd, num_samples, entropy_coeff, feature, seed,
                 update_dual_variable):
    avg_reward_list, avg_cv_list, cv_list = [], [], []
    lambd_list, avg_lambd_list, V_r_list, V_c_list, Q_weight_list = [], [], [], [], []
    ll_lambd, ul_lambd = get_lower_upper_limit_lambd(cmdp._num_constraints, ub_lambd, update_dual_variable)
    # CB Primal: to update the policy
    primal = CB_Primal(cmdp, num_iterations, feature, cmdp._num_constraints, cmdp.gamma, entropy_coeff, ul_lambd)
    # CB Dual: to update the dual variable lambd
    # [NOTE]: if mdp is passed then it creates a dummy dual object
    dual = CB_Dual(cmdp._num_constraints, alpha_lambd, ll_lambd, ul_lambd)
    # sampler for sampling the data
    data_sampler = LFASampling(env, env._num_constraints, num_samples, seed, update_dual_variable)
    # estimate Q hat from LSTD
    q_estimator = LSTD(feature.get_feature_size(), feature, cmdp, data_sampler)
    for t in range(num_iterations):
        curr_lambd = dual.curr_lambd
        q_estimator.run_solver(primal)
        v_r = data_sampler.get_V_r()
        V_r_list.append(v_r)
        if update_dual_variable:
            lambd_list.append(curr_lambd)
            primal.update_lambda_until_t(curr_lambd)
            v_c = data_sampler.get_V_c()
            grad_lambd = -(v_c - cmdp.b)
            dual.update(grad_lambd)
            cv_list.append(cmdp.b - v_c)  # calculates cv with true V_g
            V_c_list.append(v_c)
        else:
            primal.update_lambda_until_t()  # maintains a dummy lambda variable
        # update the primal policy
        w_Q = q_estimator.weights
        primal.update_Q_weights(w_Q)
        primal.increment_t_counter()
        Q_weight_list.append(w_Q)
        update_store_info(V_r_list, V_c_list, cmdp.b, lambd_list, update_dual_variable, avg_reward_list, avg_cv_list,
                          avg_lambd_list)
        
        #print(Q_weight_list)
        print(v_r)
        print(grad_lambd)
        

        # if t % 10 == 0:
        #     # saving result after every 10 iterations
        #     save_data(output_dir, np.asarray(V_r_list), np.asarray(cv_list),
        #               np.asarray(avg_reward_list), np.asarray(avg_cv_list),
        #               np.asarray(lambd_list), np.asarray(avg_lambd_list), np.asarray(Q_weight_list))
    
    return V_r_list, cv_list
    # save_data(output_dir, np.asarray(V_r_list), np.asarray(cv_list), np.asarray(avg_reward_list),
    #           np.asarray(avg_cv_list), np.asarray(lambd_list), np.asarray(avg_lambd_list), np.asarray(Q_weight_list))


if __name__ == '__main__':
    start_main_t = time.time()
    try_locally = True  # parameter to try code locally
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', help="iterations", type=int, default=150)
    # cmdp: 1 -> creates env with constraints and solve for Coin Betting on both policy and lambda
    # cmdp:0 -> creates a mdp and solve for Coin Betting on policy
    parser.add_argument('--cmdp', help="create a cmdp:1 or mdp:0", type=int, default=1)

    # Run is used to keep track of runs with different policy initialization.
    parser.add_argument('--run', help="run number", type=int, default=1)

    # alpha_lambd: parameter used for updating the lambda variable for cmdp
    parser.add_argument('--alpha_lambd', help="alpha COCOB", type=float, default=750)

    parser.add_argument('--num_tilings', type=int, default=8)  # iht number of tiles

    # num of samples for estimating Q value function
    parser.add_argument('--num_samples', help="num of samples", type=int, default=5)

    # entropy coefficient
    parser.add_argument('--entropy_coeff', help="entropy coefficient", type=float, default=5e-3)
    args = parser.parse_args()
    print("CBP with args:")
    print(args)
    if try_locally:
        save_dir_loc = "./"
    else:
        save_dir_loc = "/network/scratch/j/jainarus/CMDPData/"
    outer_file_name = "Iter" + str(args.num_iterations) + "_alpha" + str(args.alpha_lambd) + \
                      "_ntile" + str(args.num_tilings) + "_num_samples" + str(args.num_samples) + "_Ent" + str(
        args.entropy_coeff)

    env = CartPoleEnvironment()
    # fold_name = os.path.join(save_dir_loc, "Results/LFA/Cartpole/Constraints/CBP")
    # output_dir_name = os.path.join(fold_name, outer_file_name)
    # inner_file_name = "R" + str(args.run)
    # output_dir = os.path.join(output_dir_name, inner_file_name)
    seed = int(args.run)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # creates a cmdp/mdp based on arguments
    cp_tc = CartPoleTileCoding(num_tilings=args.num_tilings)
    tc_feature = TileCodingFeatures(env.A, cp_tc.get_tile_coding_args())
    ubl = entropy_to_ubl_dict_cartpole(float(args.entropy_coeff))
    if not args.cmdp:
        # creates a mdp without constraints
        lambd_star_upper_bound = 0
    else:
        # creates a cmdp
        lambd_star_upper_bound = get_lambd_upper_bound(ubl)
    run_agent_params = {'cmdp': env,
                        'num_iterations': args.num_iterations,
                        'ub_lambd': np.asarray(lambd_star_upper_bound),
                        'alpha_lambd': args.alpha_lambd,
                        'num_samples': args.num_samples,
                        'entropy_coeff': args.entropy_coeff,
                        'feature': tc_feature,
                        'seed': seed,
                        'update_dual_variable': bool(args.cmdp)
                        }
    
    ret, cv = run_CB_agent(**run_agent_params)
    current_dir = os.path.join(os.getcwd(), "Results/CBP/")
    args.run = int(args.run)
    param_dir = "R" + str(args.run) + "_iter" + str(args.num_iterations) + "_alphalam" + str(args.alpha_lambd)\
        + "_Qnsamples" + str(args.num_samples) 
    output_dir = os.path.join(current_dir, param_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(output_dir+"/ret.npy", ret)
    np.save(output_dir+"/cv1.npy", np.asarray(cv)[:, 0])
    np.save(output_dir+"/cv2.npy", np.asarray(cv)[:, 1])

    end_main_t = time.time()
    time_to_finish = timedelta(seconds=end_main_t - start_main_t)
    time_log(time_to_finish, args.num_iterations, output_dir)
