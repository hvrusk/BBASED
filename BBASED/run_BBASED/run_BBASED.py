""" Module for running the main BBASED program"""

print("loading run_BBASED")

from .getting_SEDs.observed_SED import get_source
from .set_up.model_funct import model_funct
from .model_select.model_select import select_model
from .model_select.model_select import set_labels
from .model_select.model_select import set_param_num
from .model_select.model_select import set_param_lim
from .model_select.model_select import set_pos
from .getting_SEDs.observed_SED import get_SED
from .probability.probability import log_probability_1
from .probability.probability import log_probability_2
from .prior.prior import log_prior_1
from .prior.prior import log_prior_2


import numpy as np
import scipy
import pandas as pd
import emcee
import dill
import pickle
from multiprocess import Pool


def run_BBASED(info, info_type, walkers, sample_num, model_1, model_2 = None):

    """ For Running BBASED:

        Parameters:
            info: the coords, or name of the source
            info_type: whether the coords you've given are "ICRS_coord", "gal_coord", or "source_name"
            walkers: number of walkers to use in your sampling procedure (try 128 for single-stars and 256 for binaries)
            sample_num: number of steps the walkers should take during sampling, set at your discretion, 3000 works well enough
            model_1: first stellar model type, required, either "bt-settl-cifist", "koester2" or "Kurucz2003" are available
            model_2: second stellar model type, optional but necessary if you want to fit the data as a binary, same options are available for models

        This function will return the random samples from the fitting procedure
    """

    #1. Set up
    #1.1 we use the user information to get the source data
    
    data = get_source(info, info_type)
    main_id = data[0]

    l = float(data[1])  #galactic coord in degrees
    b = float(data[2])  #galactic coord in degrees
    ra = float(data[5]) #ICRS coord in degrees
    dec = float(data[6])#ICRS coord in degrees
    plx = data[3]       #mas
    plx_err = data[4]   #mas

    #1.2 we use the data from 1.1 to determine distance and distance uncertainty

    d_mu = (1/plx)        #kpc

    d_minus = 1/(plx+plx_err)
    d_plus  = 1/(plx-plx_err)

    d_sigma = ( (d_plus - d_mu) + (d_mu - d_minus) ) / 2   #kpc

    #1.3 we set the empirical value of Rv

    Rv_q = 3.1

    #1.4 we pull the model libraries based on the user input

    funct_list1 = model_funct(model_1)
    if model_2:
        funct_list2 = model_funct(model_2)
        num_comps = 2
    else:
        num_comps = 1

    #1.5 we label model libraries, if there's only one model, then lib2 is labeled as "NAN"

    lib1, lib2 = select_model(model_1, model_2)
    print("lib1: ", lib1, "lib2: ", lib2)

    #1.6 we set our parameter labels

    labels = set_labels(lib1, lib2)

    #1.7 we set the parameter number

    param_num = set_param_num(labels)

    #2. Collect the data
    #2.1 we use the main_id found in the data gathering process to then gather the photometry

    SED = get_SED(main_id)

    #2.2  we separate the SED into its respective parts: effective wavelengths, log flux, flux err, and respective filters

    waves = []
    log_flux = []
    yerr_reported = []
    filts = []
    for key in SED:
        waves.append(SED[key]['eff_wave'])
        log_flux.append(np.log10(SED[key]["flux"]))
        yerr_reported.append((SED[key]["flux_err"]))
        filts.append(key)

    #3. Build Priors
    #3.1 we build the dust-distance prior - we need to determine the averge Av value and the Av variance
    
    from .prior.prior_dust import build_dust_prior

    Av_mu, Av_sigma_sq = build_dust_prior(d_mu, d_sigma, l, b)

    #3.2 we build the radius prior - will need either one or two dependinig on model selection

    #this should have happened automatically when you loaded the prior_radii file <-- maybe include import in this line?

    #3.3 we set parameter limits based on the model parameters

    if lib2 == "NAN":
        teff1_l, teff1_u, logg1_l, logg1_u, log_R1_l, log_R1_u, meta1_l, meta1_u = set_param_lim(lib1, lib2)
    elif lib2 != "NAN":
        (teff1_l, teff1_u, logg1_l, logg1_u, log_R1_l, log_R1_u, meta1_l, meta1_u,
            teff2_l, teff2_u, logg2_l, logg2_u, log_R2_l, log_R2_u, meta2_l, meta2_u) = set_param_lim(lib1, lib2)


    #4.Values dictionary - for accessing local variables in other modules

    var_dict = {}
    var_dict['l'] = l
    var_dict['b'] = b
    var_dict['d_mu'] = d_mu
    var_dict['d_sigma'] = d_sigma
    var_dict['Rv_q'] = Rv_q
    var_dict['Av_mu'] = Av_mu
    var_dict['Av_sigma_sq'] = Av_sigma_sq
    var_dict['num_comps'] = num_comps
    var_dict['lib1'] = lib1
    var_dict['lib2'] = lib2
    var_dict['labels'] = labels
    var_dict['param_num'] = param_num
    var_dict['SED'] = SED
    var_dict['filts'] = filts
    var_dict['teff1_l'] = teff1_l
    var_dict['teff1_u'] = teff1_u
    var_dict['logg1_l'] = logg1_l
    var_dict['logg1_u'] = logg1_u
    var_dict['log_R1_l'] = log_R1_l
    var_dict['log_R1_u'] = log_R1_u
    var_dict['meta1_l'] = meta1_l
    var_dict['meta1_u'] = meta1_u
    
    if lib2 != "NAN":
        var_dict['teff2_l'] = teff2_l
        var_dict['teff2_u'] = teff2_u
        var_dict['logg2_l'] = logg2_l
        var_dict['logg2_u'] = logg2_u
        var_dict['log_R2_l'] = log_R2_l
        var_dict['log_R2_u'] = log_R2_u
        var_dict['meta2_l'] = meta2_l
        var_dict['meta2_u'] = meta2_u

    #delete later
    print(var_dict)

    #5. Run sampler
    #5.1 Set args as a global variable for running sampling

    args = np.array(log_flux), yerr_reported
    var_dict['args'] = args

    #5.2. we set the initial positions for the walkers

    pos = set_pos(lib1, lib2, d_mu)

    #5.3 emcee <-- test, this might have errors with the parallelization, also change the add. prior to be incorporated into the blob

    with Pool() as pool:

        pos = pos + 1e-2 * np.random.randn(walkers, param_num)
        nwalkers, ndim = pos.shape


        if num_comps == 1:
            BBASED_results = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_1, pool=pool, kwargs=var_dict
            )

            prior_emcee = emcee.EnsembleSampler(
                nwalkers, ndim, log_prior_1, pool=pool, kwargs=var_dict
            )

        if num_comps == 2:
            BBASED_results = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability_2, pool=pool, kwargs=var_dict
            )

            prior_emcee = emcee.EnsembleSampler(
                nwalkers, ndim, log_prior_2, pool=pool, kwargs=var_dict
            )

        BBASED_results.run_mcmc(pos, sample_num, progress=True);
        prior_emcee.run_mcmc(pos, sample_num, progress=True);

    return BBASED_results, prior_emcee

