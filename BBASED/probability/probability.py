""" Probability functions for sampling procedure, require the prior and
    likelihood functions 
"""


def log_probability_1(theta):
    lp = log_prior_1(theta)
    if not np.isfinite(lp):
        return -np.inf, None
    return lp + log_likelihood_1(theta), log_likelihood_1(theta)

def log_probability_2(theta):
    lp = log_prior_2(theta)
    if not np.isfinite(lp):
        return -np.inf, None
    return lp + log_likelihood_2(theta), log_likelihood_2(theta)
