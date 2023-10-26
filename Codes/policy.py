import numpy as np
from tabularCMDPEnv import *


class Policy:
    """
    probability representation of a policy, i.e. set of simplex over states
    """

    def __init__(self, env, rng):
        self.env = env
        S = self.env.S
        A = self.env.A

        # initializes a random matrix [S, A] as the probability representation of policy
        self.policy_prob = rng.dirichlet(np.ones(A), size=S)

    def set_policy_prob(self, policy_prob):
        # set the input matrix as the proabability representation of policy
        self.policy_prob = policy_prob

    # true reward function [S, A] is given, returns value function V over states[S]
    def get_V_function(self, reward):
        # computes the V^{\pi} = (I - \gamma* P_\pi)^{-1} R_\pi
        # returns  a [S] dimensional vector
        P = self.env.P
        discount = self.env.gamma
        ppi = np.einsum('sat,sa->st', P, self.policy_prob)
        rpi = np.einsum('sa,sa->s', reward, self.policy_prob)
        vf, _, _, _ = np.linalg.lstsq(np.eye(P.shape[-1]) - discount * ppi, rpi,
                                      rcond=None)  # to resolve singular matrix issues, used least square method rather than solve
        return vf

    # true reward function [S, A] is given, returns state-action value function Q over (s, a) [S, A]
    def get_Q_function(self, reward):
        # computes the Q^{\pi} = R + (\gamma * P * V_\pi)
        # returns a [S x A] array
        P = self.env.P
        discount = self.env.gamma
        vf = self.get_V_function(reward)
        Qf = reward + (discount * np.einsum('sat,t->sa', P, vf))
        return Qf

    

