from __future__ import absolute_import, division, print_function

import numpy as np

from adopy.generic import ADOGeneric
from adopy.functions import inv_logit, log_lik_bern
from adopy.functions import make_vector_shape, make_grid_matrix

FUNC_POW = {'p', 'pow', 'power'}
FUNC_EXP = {'e', 'exp', 'exponential'}

FUNC_VALID = FUNC_POW ^ FUNC_EXP

NUM_ARGS = {k: 2 for k in FUNC_POW} ^ \
           {k: 2 for k in FUNC_EXP}

COND_PARAMS = {k: {} for k in FUNC_POW} ^ \
              {k: {} for k in FUNC_EXP}


class DelayedDiscounting(ADOGeneric):
    """ADO engine for delayed discounting task"""

    def __init__(self, func_type, durations, rewards, *args):
        super(DelayedDiscounting, self).__init__()

        self.label_design = ['d_early', 'd_delay', 'r_early', 'd_late']
        self.label_param = ['alpha', 'beta', 'gamma']

        assert func_type in FUNC_VALID
        self.func_type = func_type

        assert len(np.shape(durations)) == 2, 'Durations should have 2 dimensions.'
        self.durations = np.array(durations)
        assert len(np.shape(rewards)) == 2, 'Rewards should have 2 dimensions.'
        self.rewards = np.array(rewards)

        self.num_args = len(args)
        assert self.num_args == NUM_ARGS[self.func_type]
        self.args = [np.array(arg).flatten() for arg in args]

        # Binary response
        # - 0: the option with early reward
        # - 1: the option with late reward
        self.y_obs = np.array([0, 1])

        self.designs = [self.durations, self.rewards]
        self.params = self.args

        self.cond_param = {
            'gamma': lambda x: x >= 0
        }

        self.grid_design = make_grid_matrix(*self.designs)
        self.grid_param = make_grid_matrix(*self.params)
        self.grid_response = make_grid_matrix(self.y_obs)

        self.initialize()

    @classmethod
    def compute_p_obs(cls, func_type, prob, ambig, r_var, r_fix, alpha, beta, gamma):
        assert func_type in FUNC_VALID

        # Calculate the subjective value of a variable option (risky or ambiguous).
        sv_var = np.power(r_var, alpha)
        if func_type in FUNC_POW:
            sv_var = (prob - beta * np.divide(ambig, 2)) * sv_var
        elif func_type in FUNC_EXP:
            sv_var = np.power(prob, 1 + beta * ambig) * sv_var

        # Calculate the subjective value of a reference option with a fixed probability of 0.5.
        sv_fix = .5 * np.power(r_fix, alpha)

        # Using a logistic function, compute the probability to choose the variable option.
        p_obs = inv_logit(gamma * (sv_var - sv_fix))

        return p_obs

    def _compute_p_obs(self):
        shape_design = make_vector_shape(2, 0)
        shape_param = make_vector_shape(2, 1)

        prob = self.grid_design[:, 0].reshape(shape_design)
        ambig = self.grid_design[:, 1].reshape(shape_design)
        r_var = self.grid_design[:, 2].reshape(shape_design)
        r_fix = self.grid_design[:, 3].reshape(shape_design)
        alpha = self.grid_param[:, 0].reshape(shape_param)
        beta = self.grid_param[:, 1].reshape(shape_param)
        gamma = self.grid_param[:, 2].reshape(shape_param)

        return self.compute_p_obs(func_type=self.func_type,
                                  prob=prob, ambig=ambig, r_var=r_var, r_fix=r_fix,
                                  alpha=alpha, beta=beta, gamma=gamma)  # yapf: disable

    def _compute_log_lik(self):
        dim_p_obs = len(self.p_obs.shape)
        y = self.y_obs.reshape(make_vector_shape(dim_p_obs + 1, dim_p_obs))
        p = np.expand_dims(self.p_obs, dim_p_obs)

        return log_lik_bern(y, p)
