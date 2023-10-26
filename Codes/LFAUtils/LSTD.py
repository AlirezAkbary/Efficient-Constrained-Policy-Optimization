import numpy as np
import scipy.linalg

class LSTD:
    """
    Implements the LSTD paper.
    paper: https://www2.cs.duke.edu/research/AI/LSPI/jmlr03.pdf
    """

    def __init__(self, num_features, feature, env, data_sampler):
        self.num_features = int(num_features)
        self.feature = feature
        self.gamma = env.gamma
        self.A = env.A
        # sampler is object of Sampling class which has get_data() implemented
        self.sampler = data_sampler
        self.data = None

    def run_solver(self, policy_prob):
        """
        LSTDQ implementation to solve for standard matrix solvers.
        See Fig 5 in LSPI paper:https://www2.cs.duke.edu/research/AI/LSPI/jmlr03.pdf
        A = \phi^T.(\phi - \gamma P \Pi_\pi \phi)
        b = \phi^T.R

        Parameter:
            policy_prob: policy from which data is sampled
        Returns:
            w: returns the new weight vector [k X dim_reward],
            dim_reward = num_constraints + 1
        """
        data = self.sampler.get_data(policy_prob)
        self.set_data(data)
        A = np.zeros((self.num_features, self.num_features))
        dim_reward = len(data[0].r)
        # b = [\Phi^T.R, \Phi^T.C_1, \Phi^T.C_2]
        b = np.zeros((self.num_features, dim_reward))
        for sample in data:
            phi_sa_ids = set(self.feature.get_feature(sample.s, sample.a))
            phi_sa_next_ids = set(self.feature.get_feature(sample.next_s, sample.next_a))
            common_ids = phi_sa_ids.intersection(phi_sa_next_ids)
            a_minus_b_ids = phi_sa_ids - phi_sa_next_ids
            b_minus_a_ids = phi_sa_next_ids - phi_sa_ids
            b_vector = sample.r
            for common_id in common_ids:
                A[list(phi_sa_ids), common_id] += 1 - self.gamma
            for a_minus_b_id in a_minus_b_ids:
                A[list(phi_sa_ids), a_minus_b_id] += 1
            for b_minus_a_id in b_minus_a_ids:
                A[list(phi_sa_ids), b_minus_a_id] += -self.gamma
            b[list(phi_sa_ids)] += b_vector
        a_rank = np.linalg.matrix_rank(A)
        if a_rank == self.num_features:
            w = scipy.linalg.solve(A, b)  # [k X 3]
        else:
            w = scipy.linalg.lstsq(A, b)[0]  # [k X 3]
        self.weights = w  # weights of Q matrix
        return w

    def get_current_Q_weights(self):
        # returns the weight matrix of Q which is of dim [num_features X (num_constraints+1)]
        return self.weights

    def get_estimated_Q(self, state):
        """
        returns Q(s,a) for all a where Q is [num_actions X num_constraints+1] array
        """
        
        q = [np.sum(np.asarray([self.weights[k] for k in self.feature.get_feature(state, a)]), axis=0) for a in
             range(self.A)]
        
        return np.asarray(q)

    def get_error_norm(self, trueQ, estimatedQ):
        """
        returns relative error which is 2 norm of difference between ||trueQ - estimatedQ||_2/||trueQ||_2
        """
        return np.linalg.norm(trueQ - estimatedQ, ord='fro') / np.linalg.norm(trueQ, ord='fro')

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data