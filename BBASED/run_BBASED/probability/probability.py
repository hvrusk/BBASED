""" Probability functions for sampling procedure, require the prior and
    likelihood functions 
"""

import numpy as np

from ..prior.prior import log_prior_1
from ..prior.prior import log_prior_2
from ..likelihood.likelihood import log_likelihood_1
from ..likelihood.likelihood import log_likelihood_2

def log_probability_1(theta, **var_dict):
    lp = log_prior_1(theta, var_dict)
    if not np.isfinite(lp):
        return -np.inf, None
    return lp + log_likelihood_1(theta, var_dict), log_likelihood_1(theta, var_dict)

def log_probability_2(theta, **var_dict):
    lp = log_prior_2(theta, var_dict)
    if not np.isfinite(lp):
        return -np.inf, None
    return lp + log_likelihood_2(theta, var_dict), log_likelihood_2(theta, var_dict)
