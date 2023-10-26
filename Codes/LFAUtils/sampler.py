from collections import defaultdict
import itertools
import numpy as np

class Data:
    def __init__(self, s, a, next_s, next_a, r):
        self.s = s
        self.a = a
        self.next_s = next_s
        self.next_a = next_a
        self.r = r  # r is a vector containing both reward and costs


class Sampling:
    """
    Get samples for tabular class for all (s,a) pair according to current policy
    """

    def __init__(self, env, num_samples, tc_args, constraints=True):
        self.S = env.S
        self.A = env.A
        self.P = env.P
        self.R = env.R
        self.G = env.G
        self.num_samples = num_samples
        self.constraints = constraints  # whether the env has constraints or not (mdp/cmdp)
        self.tc_args = tc_args
        self.list_s_a_pair = self.get_s_a_pairs()
        self.len_s_a_list = len(self.list_s_a_pair)

    def get_s_a_pairs(self):
        return list(itertools.product(range(self.S), range(self.A)))

    def get_data(self, policy_prob):
        """
        policy_prob : policy probability dim [S X A]
        """
        data = []
        for k in range(self.num_samples * self.S * self.A):
            s, a = self.list_s_a_pair[np.random.choice(self.len_s_a_list)]
            next_s = np.random.choice(self.S, p=self.P[s, a])
            next_a = np.random.choice(self.A, p=policy_prob[next_s])
            reward_vec = [self.R[s, a]]
            if self.constraints:
                reward_vec.extend(self.G[:, s, a])
            s_rep = self.tc_args.get_state_representation(s)
            next_s_rep = self.tc_args.get_state_representation(next_s)
            data.append(Data(s_rep, a, next_s_rep, next_a, reward_vec))
        return data


class KW_Sampling:
    """
    Get samples for tabular class for (s,a) pairs in KW coreset according to current policy
    """

    def __init__(self, env, num_samples, tc_args, index_to_sa_pairs, constraints=True):
        self.S = env.S
        self.A = env.A
        self.P = env.P
        self.R = env.R
        self.G = env.G
        self.num_samples = num_samples
        self.constraints = constraints  # whether the env has constraints or not (mdp/cmdp)
        self.tc_args = tc_args
        self.index_to_sa_pairs = index_to_sa_pairs
        self.list_of_indexes = list(self.index_to_sa_pairs.keys())
        self.len_index = len(self.index_to_sa_pairs)

    def get_data(self, policy_prob):
        """
        policy_prob : policy probability dim [S X A]
        """
        data = []
        count_s_a_pair = defaultdict(int)
        for k in range(self.num_samples * self.len_index):
            list_s_a_pairs = self.index_to_sa_pairs[np.random.choice(self.list_of_indexes)]
            s, a = list_s_a_pairs[np.random.choice(len(list_s_a_pairs))]
            next_s = np.random.choice(self.S, p=self.P[s, a])
            next_a = np.random.choice(self.A, p=policy_prob[next_s])
            reward_vec = [self.R[s, a]]
            if self.constraints:
                reward_vec.extend(self.G[:, s, a])
            s_rep = self.tc_args.convert_state_to_two_coordinate(s)
            next_s_rep = self.tc_args.convert_state_to_two_coordinate(next_s)
            data.append(Data(s_rep, a, next_s_rep, next_a, reward_vec))
            count_s_a_pair[(s,a)] +=1
        print_dict(count_s_a_pair)
        return data

def print_dict(count_s_a_pair):
    for k,v in count_s_a_pair.items():
        print(k, ":", v)


class LFASampling:
    """
    Sampling data for LFA environment like cartpole
    """

    def __init__(self, env, num_constraints, num_traj, seed, constraints=True):
        self.env = env
        self.num_constraints = num_constraints
        self.num_traj = num_traj
        self.rng = np.random.RandomState(seed)
        self.constraints = constraints
        self.A = env.A

    def get_data(self, policy):
        """
        Collect samples using the current policy
        Parameter:
            policy = policy class object which has policy_prob function implemented and return prob of action given state
        Returns:
            data =  list of samples
        """
        data = []
        return_value = 0
        cost_value = np.zeros(self.num_constraints)
        for t in range(self.num_traj):
            state = self.env.get_init_state()
            action = self.rng.choice(self.A, p=policy.policy_prob(state))
            running = True
            n_step = 0
            return_traj = 0.0
            cost_traj = np.zeros(self.num_constraints)
            current_discount = 1.0
            while running:
                next_state, reward, cost = self.env.step(state, action)
                if n_step == self.env._max_steps or self.env.is_state_over_bounds(next_state):
                    running = False
                    continue
                next_action = self.rng.choice(self.A, p=policy.policy_prob(next_state))
                rew_vector = [reward]
                if self.constraints:
                    rew_vector.extend(cost)
                sample_data = Data(state, action, next_state, next_action, rew_vector)
                data.append(sample_data)
                return_traj += current_discount * reward
                cost_traj += current_discount * cost
                current_discount *= self.env.gamma
                state = next_state
                action = next_action
                n_step += 1
            return_value += return_traj
            cost_value += cost_traj
        self.set_V_r(return_value / self.num_traj)
        self.set_V_c(cost_value / self.num_traj)
        return data

    def get_V_r(self):
        return self.return_value

    def get_V_c(self):
        return self.cost_value

    def set_V_r(self, return_val):
        self.return_value = return_val

    def set_V_c(self, cost_val):
        self.cost_value = cost_val
