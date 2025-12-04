""" Likelihood functions - depend on which models you're using, required for
    the sampling procedure """
print("likelihood loading")

import numpy as np
import math 
import csv
from ..set_up.model_funct import model_funct
R_kpc_conversion = 5.08327 * 10**-22 #(kpc/R)**2


def filter_eff_wave(filter_name):
    with open('BBASED/run_BBASED/set_up/eff_waves.csv', newline='') as data: #opens file with filters and eff waves from svo
        reader = csv.DictReader(data) #reads file as a dictionary

        for row in reader:
            if row['filtername'] == filter_name:
                wave = row['eff_wave'] #gets effective wavelength
                break
            else:
                wave = False  #edit
    return wave

def get_spec_funct(funct_list, teff, logg, Av, Rv, var_dict, meta="na"):
    fltr_flxs = {}
    if meta == "na":
        for key in var_dict['filts']:
            eff_wave = float(filter_eff_wave(key))
            filt_flux = float(funct_list[key]([teff,logg, Av, Rv])[0])
            fltr_flxs[key] = {'eff_wave':eff_wave, 'filt_flux':filt_flux}
    else: #case where we have meta is when we're using the MS models
        for key in var_dict['filts']:
            eff_wave = float(filter_eff_wave(key))
            filt_flux = float(funct_list[key]([teff,logg,meta,Av, Rv])[0])
            fltr_flxs[key] = {'eff_wave':eff_wave, 'filt_flux':filt_flux}
    #returns dict of flux and eff waves for every filter in a set
    return fltr_flxs


def log_likelihood_1(theta, var_dict):
    """ Log likelihood function - requires that the data (args) are defined as a global variable,
        that the model libraries have been defined
    """

    log10_y, yerr_dex = var_dict['args']

    #parameters
    if var_dict['lib1'] =='Kurucz2003':
        log_teff1, logg1, log_R1, meta1, log_dist,  Av_unbounded, Rv, log10_sigma_theor = theta

    else:
        log_teff1, logg1, log_R1, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

    #clip negative values of Av to 0:
    Av_clipped = np.clip(Av_unbounded, 0, None)
    
    #assigning SED model function F_i(θ)
    funct_list1 = model_funct(var_dict['lib1'])

    #TEMP
    if var_dict['lib1'] =='Kurucz2003':
        flx_dict1 = get_spec_funct(funct_list1, 10**(log_teff1), logg1, Av_clipped, Rv, var_dict, meta=meta1)
    else:
        flx_dict1 = get_spec_funct(funct_list1, 10**(log_teff1), logg1, Av_clipped, Rv, var_dict) 

    R1_sq = ((10**log_R1)**2)
    dist_sq = ((10**log_dist)**2)

    #log value of the flux
    model = {}
    for key in var_dict['filts']:
        y_model = (flx_dict1[key]['filt_flux'] + np.log10(R_kpc_conversion * R1_sq / dist_sq)) 
        wavelength = (flx_dict1[key]['eff_wave'])
        model[key] = {'y_model':y_model, 'wavelength':wavelength}

    #check for kurucz uneven grid, parameters that fall within range
    #heather what does this line do?? i think this might be wrong??
    #if math.isnan(model[filts[0]]['y_model'])==True:
    #    return -np.inf

    
    #deviations
    sigma_theory_dex = 10**log10_sigma_theor

    log_like_tot = []
    for key in var_dict['filts']:
        if var_dict['SED'][key]['ph_qual'] == 'U':
            #linear / upper limit case
            sigma2 = var_dict['SED'][key]['flux']**2 + sigma_theory_dex**2
            log_like_i = -0.5 * (10**(model[key]['y_model']))**2 / sigma2 - 0.5*np.log(sigma2)
        else:
            sigma_dex2 = var_dict['SED'][key]['flux_err']**2 + sigma_theory_dex**2
            log_like_i = -0.5 * ((np.log10(var_dict['SED'][key]['flux']) - model[key]['y_model'])**2) / sigma_dex2 - 0.5*np.log(sigma_dex2)
        log_like_tot.append(log_like_i)

    return np.sum(log_like_tot)


def log_likelihood_2(theta, var_dict):
    """ Log Likelihood function for a binary system - theta, our parameters, are the only input
        Returns the sum of the log likelihood
        The data (args) and model libraries should once again be defined as global variables
        Parameter number is also a global variable - this might be updated in the future to read the labels
        instead of the model selection to determine theta
    """
    log10_y, yerr_dex = var_dict['args']

    #parameters
    if param_num == 10:
        log_teff1, logg1, log_R1, log_teff2, logg2, log_R2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta
    
    if var_dict['lib1'] =='Kurucz2003':
        log_teff1, logg1, log_R1, meta1,log_teff2, logg2, log_R2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta
    if var_dict['lib2'] == 'Kurucz2003':
        log_teff1, logg1, log_R1, log_teff2, logg2, log_R2, meta2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta
    if param_num == 12:
        log_teff1, logg1, log_R1, meta1,log_teff2, logg2, log_R2, meta2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

    #clip negative values of Av to 0:
    Av_clipped = np.clip(Av_unbounded, 0, None)
    
    #assigning SED model functions F_i(θ)
    funct_list1 = model_funct(var_dict['lib1'])
    funct_list2 = model_funct(var_dict['lib2'])

    #dict of [filter_id] : eff_wavelengths, filter_flux_values
    if var_dict['lib1'] =='Kurucz2003':
        flx_dict1 = get_spec_funct(funct_list1, 10**(log_teff1), logg1, Av_clipped, Rv, var_dict, meta=meta1)
    else:
        flx_dict1 = get_spec_funct(funct_list1, 10**(log_teff1), logg1, Av_clipped, Rv, var_dict)
    
    if var_dict['lib2'] == 'Kurucz2003':
        flx_dict2 = get_spec_funct(funct_list2, 10**(log_teff2), logg2, Av_clipped, Rv, var_dict, meta=meta2)
    else:
        flx_dict2 = get_spec_funct(funct_list2, 10**(log_teff2), logg2, Av_clipped, Rv, var_dict)

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
    for key in var_dict['filts']:
        if SED[key]['ph_qual'] == 'U':
            #linear / upper limit case
            sigma2 = var_dict['SED'][key]['flux']**2 + sigma_theory_dex**2
            log_like_i = -0.5 * (10**(model[key]['y_model']))**2 / sigma2 - 0.5*np.log(sigma2)
        else:
            sigma_dex2 = var_dict['SED'][key]['flux_err']**2 + sigma_theory_dex**2
            log_like_i = -0.5 * ((np.log10(var_dict['SED'][key]['flux']) - model[key]['y_model'])**2) / sigma_dex2 - 0.5*np.log(sigma_dex2)
        log_like_tot.append(log_like_i)
    return np.sum(log_like_tot)
