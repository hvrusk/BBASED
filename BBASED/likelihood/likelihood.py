""" Likelihood functions - depend on which models you're using, required for
    the sampling procedure """

import numpy as np
import math 

def log_likelihood_1(theta):
    """ Log likelihood function - requires that the data (args) are defined as a global variable,
        that the model libraries have been defined
    """

    log10_y, yerr_dex = args

    #parameters
    if lib1 =='Kurucz2003':
        log_teff1, logg1, log_R1, meta1, log_dist,  Av_unbounded, Rv, log10_sigma_theor = theta

    else:
        log_teff1, logg1, log_R1, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

    #clip negative values of Av to 0:
    Av_clipped = np.clip(Av_unbounded, 0, None)
    
    #TEMP
    if lib1 =='Kurucz2003':
        flx_dict1 = get_spec_funct(funct_list1, 10**(log_teff1), logg1, Av_clipped, Rv, meta=meta1)
    else:
        flx_dict1 = get_spec_funct(funct_list1, 10**(log_teff1), logg1, Av_clipped, Rv) 

    R1_sq = ((10**log_R1)**2)
    dist_sq = ((10**log_dist)**2)

    #log value of the flux
    model = {}
    for key in filts:
        y_model = (flx_dict1[key]['filt_flux'] + np.log10(R_kpc_conversion * R1_sq / dist_sq)) 
        wavelength = (flx_dict1[key]['eff_wave'])
        model[key] = {'y_model':y_model, 'wavelength':wavelength}

    #check for kurucz uneven grid, parameters that fall within range
    #heather what does this line do?? i think this might be wrong??
    if math.isnan(model[filts[0]]['y_model'])==True:
        return -np.inf

    
    #deviations
    sigma_theory_dex = 10**log10_sigma_theor

    log_like_tot = []
    for key in SED:
        if SED[key]['ph_qual'] == 'U':
            #linear / upper limit case
            sigma2 = SED[key]['flux']**2 + sigma_theory_dex**2
            log_like_i = -0.5 * (10**(model[key]['y_model']))**2 / sigma2 - 0.5*np.log(sigma2)
        else:
            sigma_dex2 = SED[key]['flux_err']**2 + sigma_theory_dex**2
            log_like_i = -0.5 * ((np.log10(SED[key]['flux']) - model[key]['y_model'])**2) / sigma_dex2 - 0.5*np.log(sigma_dex2)
        log_like_tot.append(log_like_i)

    return np.sum(log_like_tot)


def log_likelihood_2(theta):
    """ Log Likelihood function for a binary system - theta, our parameters, are the only input
        Returns the sum of the log likelihood
        The data (args) and model libraries should once again be defined as global variables
        Parameter number is also a global variable - this might be updated in the future to read the labels
        instead of the model selection to determine theta
    """
    log10_y, yerr_dex = args

    #parameters
    if param_num == 10:
        log_teff1, logg1, log_R1, log_teff2, logg2, log_R2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta
    
    if lib1 =='Kurucz2003':
        log_teff1, logg1, log_R1, meta1,log_teff2, logg2, log_R2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta
    if lib2 == 'Kurucz2003':
        log_teff1, logg1, log_R1, log_teff2, logg2, log_R2, meta2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta
    if param_num == 12:
        log_teff1, logg1, log_R1, meta1,log_teff2, logg2, log_R2, meta2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

    #clip negative values of Av to 0:
    Av_clipped = np.clip(Av_unbounded, 0, None)

    #dict of [filter_id] : eff_wavelengths, filter_flux_values
    if lib1 =='Kurucz2003':
        flx_dict1 = get_spec_funct(funct_list1, 10**(log_teff1), logg1, Av_clipped, Rv, meta=meta1)
    else:
        flx_dict1 = get_spec_funct(funct_list1, 10**(log_teff1), logg1, Av_clipped, Rv)
    
    if lib2 == 'Kurucz2003':
        flx_dict2 = get_spec_funct(funct_list2, 10**(log_teff2), logg2, Av_clipped, Rv, meta=meta2)
    else:
        flx_dict2 = get_spec_funct(funct_list2, 10**(log_teff2), logg2, Av_clipped, Rv)

    R1_sq = ((10**log_R1)**2)
    R2_sq = ((10**log_R2)**2)
    dist_sq = ((10**log_dist)**2)

    #log value of the flux
    #model is the values of filter_flux from the dict
    model = {}
    for key in filts:
        y_model = np.log10(( (10**flx_dict1[key]['filt_flux'] * (R1_sq)) + (10**(flx_dict2[key]['filt_flux']) * (R2_sq)) ) * (R_kpc_conversion/(dist_sq) ))
        wavelength = (flx_dict1[key]['eff_wave'])
        model[key] = {'y_model':y_model, 'wavelength':wavelength}

    if math.isnan(model[filts[0]]['y_model'])==True:    
        return -np.inf
    

    #deviations
    sigma_theory_dex = 10**log10_sigma_theor
    
    log_like_tot = []
    for key in SED:
        if SED[key]['ph_qual'] == 'U':
            #linear / upper limit case
            sigma2 = SED[key]['flux']**2 + sigma_theory_dex**2
            log_like_i = -0.5 * (10**(model[key]['y_model']))**2 / sigma2 - 0.5*np.log(sigma2)
        else:
            sigma_dex2 = SED[key]['flux_err']**2 + sigma_theory_dex**2
            log_like_i = -0.5 * ((np.log10(SED[key]['flux']) - model[key]['y_model'])**2) / sigma_dex2 - 0.5*np.log(sigma_dex2)
        log_like_tot.append(log_like_i)
    return np.sum(log_like_tot)
