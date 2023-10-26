class FMAPGPrimal:

    def __init__(self, env, num_cons, eta, alpha, m):

        self.env = env

        # number of constraints
        self.num_cons = num_cons


        # number of inner loops
        self.m = m


        # step size for functional update (surrogate construction)
        self.eta = eta


        # step size for parameter update (gradinet ascent)
        self.alpha = alpha


    def update_policy_param():
        pass

    def update_inner_weight():
        pass

    def get_policy_prob():
        pass

    def get_objective():
        pass
