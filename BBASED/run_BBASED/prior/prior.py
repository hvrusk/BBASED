""" log Prior functions - ones that are actually used in the probability
    function that combines all the priors used in the final log prior, plus
    some additional prior functions """

print("prior loading")

import numpy as np
from ..prior.prior_dust import Av_d_log_prior
from ..prior.prior_radii import rad1_prior
from ..prior.prior_radii import rad2_prior
from ..model_select.model_select import set_param_lim
from ..prior.load_radii import *

def sigma_theor_log_prior(log10_sigma_theor):
    """ log prior on the uncertainty of the theoretical models parameter
    """
    return (-(np.log(10))*log10_sigma_theor)


def Rv_log_prior(Rv,var_dict):
    """ prior on the dust parameter R_V - this is just a normal distribution
        centered on the empirical values of R_V
        sigma_rv is taken from Speagle et al 2024
    """
    sigma_rv = 0.18
    return (np.log(1.0/(np.sqrt(2*np.pi)*sigma_rv))-0.5*(Rv-var_dict['Rv_q'])**2/sigma_rv**2)

def jacobian_log(log_dist, Rv):
    return np.log( (10**log_dist) * (3.1 / Rv) )

def log_prior_1(theta, var_dict):
    """ Prior function for a single star system, used in sampling
    """
    
    if var_dict['lib1'] == 'Kurucz2003': #<-- this could be updated in the future
        log_teff1, logg1, log_R1,meta1, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

        #parameter ranges
        if not (var_dict['teff1_l'] <= log_teff1 <= var_dict['teff1_u'] and
                var_dict['logg1_l'] <= logg1 <= var_dict['logg1_u'] and
                var_dict['log_R1_l']<=log_R1<=var_dict['log_R1_u'] and
                var_dict['meta1_l']<=meta1<=var_dict['meta1_u'] and
                -3<= log_dist <= 1 and
                Av_unbounded <= 5 and
                2.5<=Rv<=5.5 and
                -2 < log10_sigma_theor < 0):
            return -np.inf
        return sigma_theor_log_prior(log10_sigma_theor) + Rv_log_prior(Rv,var_dict) + Av_d_log_prior(Av_unbounded, (10**log_dist), var_dict) + jacobian_log(log_dist, Rv) + rad1_prior(log_R1,log_teff1, var_dict) + np.log10(met_prior([meta1]))

    else:
        log_teff1, logg1, log_R1, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

        #parameter ranges
        if not (var_dict['teff1_l'] <= log_teff1 <= var_dict['teff1_u'] and
                var_dict['logg1_l'] <= logg1 <= var_dict['logg1_u'] and
                var_dict['log_R1_l']<=log_R1<=var_dict['log_R1_u'] and
                -3<= log_dist <= 1 and
                Av_unbounded <= 5 and
                2.5<=Rv<=5.5 and
                -2 < log10_sigma_theor < 0):
            return -np.inf


        return sigma_theor_log_prior(log10_sigma_theor) + Rv_log_prior(Rv,var_dict) + Av_d_log_prior(Av_unbounded, (10**log_dist), var_dict) + jacobian_log(log_dist, Rv) + rad1_prior(log_R1,log_teff1, var_dict)


def log_prior_2(theta, var_dict):
    """ Prior function for a binary star system, used in sampling
    """
    
    if "meta1" in var_dict['labels']:
        log_teff1, logg1, log_R1, meta1, log_teff2, logg2, log_R2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta


        #parameter ranges
        if not (var_dict['teff1_l'] <= log_teff1 <= var_dict['teff1_u'] and
                var_dict['logg1_l'] <= logg1 <= var_dict['logg1_u'] and
                var_dict['log_R1_l']<=log_R1<=var_dict['log_R1_u'] and
                var_dict['meta1_l']<=meta1<=var_dict['meta1_u'] and
                var_dict['teff2_l'] <= log_teff2 <= var_dict['teff2_u'] and
                log_teff2 < log_teff1 and
                var_dict['logg2_l'] <= logg2 <= var_dict['logg2_u'] and
                var_dict['log_R2_l']<=log_R2<=var_dict['log_R2_u'] and
                -3<= log_dist <= 1 and
                Av_unbounded <= 5 and
                2.5<=Rv<=5.5 and
                -2 < log10_sigma_theor < 0):
            return -np.inf

        met1 = np.log10(met_prior([meta1]))

    elif "meta2" in var_dict['labels']:
        log_teff1, logg1, log_R1, log_teff2, logg2, log_R2, meta2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta
        
        #parameter ranges
        if not (var_dict['teff1_l'] <= log_teff1 <= var_dict['teff1_u'] and
                var_dict['logg1_l'] <= logg1 <= var_dict['logg1_u'] and
                var_dict['log_R1_l']<=log_R1<=var_dict['log_R1_u'] and
                var_dict['teff2_l'] <= log_teff2 <= var_dict['teff2_u'] and
                log_teff2 < log_teff1 and
                var_dict['logg2_l'] <= logg2 <= var_dict['logg2_u'] and
                var_dict['log_R2_l']<=log_R2<=var_dict['log_R2_u'] and
                var_dict['meta2_l']<=meta2<=var_dict['meta2_u'] and
                -3<= log_dist <= 1 and
                Av_unbounded <= 5 and
                2.5<=Rv<=5.5 and
                -2 < log10_sigma_theor < 0):
            return -np.inf
        
        met1 = 0
        met2 = np.log10(met_prior([meta2]))

    if "meta1" and "meta2" in var_dict['labels']:
        log_teff1, logg1, log_R1, meta1, log_teff2, logg2, log_R2, meta2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

        #parameter ranges
        if not (var_dict['teff1_l'] <= log_teff1 <= var_dict['teff1_u'] and
                var_dict['logg1_l'] <= logg1 <= var_dict['logg1_u'] and
                var_dict['log_R1_l']<=log_R1<=var_dict['log_R1_u'] and
                var_dict['meta1_l']<=meta1<=var_dict['meta1_u'] and
                var_dict['teff2_l'] <= log_teff2 <= var_dict['teff2_u'] and
                log_teff2 < log_teff1 and
                var_dict['logg2_l'] <= logg2 <= var_dict['logg2_u'] and
                var_dict['log_R2_l']<=log_R2<=var_dict['log_R2_u'] and
                var_dict['meta2_l']<=meta2<=var_dict['meta2_u'] and
                -3<= log_dist <= 1 and
                Av_unbounded <= 5 and
                2.5<=Rv<=5.5 and
                -2 < log10_sigma_theor < 0):
            return -np.inf

        met1 = np.log10(met_prior([meta1]))
        met2 = np.log10(met_prior([meta2]))

    if len(labels) == 10:
        log_teff1, logg1, log_R1, log_teff2, logg2, log_R2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

        #parameter ranges
        if not (var_dict['teff1_l'] <= log_teff1 <= var_dict['teff1_u'] and
                var_dict['logg1_l'] <= logg1 <= var_dict['logg1_u'] and
                var_dict['log_R1_l']<=log_R1<=var_dict['log_R1_u'] and
                var_dict['teff2_l'] <= log_teff2 <= var_dict['teff2_u'] and
                log_teff2 < log_teff1 and
                var_dict['logg2_l'] <= logg2 <= var_dict['logg2_u'] and
                var_dict['log_R2_l']<=log_R2<=var_dict['log_R2_u'] and
                -3<= log_dist <= 1 and
                Av_unbounded <= 5.0 and
                2.5<=Rv<=5.5 and
                -2 < log10_sigma_theor < 0):
            return -np.inf

        met1 = 0
        met2 = 0

    Jacobian_log = np.log( (10**log_dist) * (3.1 / Rv) )

    return sigma_theor_log_prior(log10_sigma_theor) + Rv_log_prior(Rv,var_dict) + Av_d_log_prior(Av_unbounded, (10**log_dist), var_dict) + Jacobian_log +rad1_prior(log_R1,log_teff1, var_dict) + rad2_prior(log_R2,log_teff2, var_dict) + met1 + met2


