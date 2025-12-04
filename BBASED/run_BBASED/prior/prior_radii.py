""" priors for radii - based on stellar types
    requires you to have already defined the stellar models as a global variable
    requires you to load the radius prior files for MS and WD stars"""

import numpy as np

def rad1_prior(log_R1,log_teff1, var_dict):
    if var_dict['lib1'] == "bt-settl-cifist":
        r_mu = 0.08529303
        r_sigma = 0.0205525
        return (np.log(1.0/(np.sqrt(2*np.pi)*r_sigma))-0.5*((10**log_R1)-r_mu)**2/r_sigma**2)
    if var_dict['lib1'] == "koester2":
        return np.log10(WD_radius_prior([log_R1,log_teff1]))
    if var_dict['lib1'] == "Kurucz":
        rad = np.clip(MS_radius_prior([log_R1,log_teff1]), 0.000001, None)
        return np.log10(rad)

def rad2_prior(log_R2,log_teff2, var_dict):
    if var_dict['lib2'] == "bt-settl-cifist":
        r_mu = 0.08529303
        r_sigma = 0.0205525
        return (np.log(1.0/(np.sqrt(2*np.pi)*r_sigma))-0.5*((10**log_R1)-r_mu)**2/r_sigma**2)
    if var_dict['lib2'] == "koester2":
        return np.log10(WD_radius_prior([log_R2,log_teff2]))
    if var_dict['lib2'] == "Kurucz":
        rad = np.clip(MS_radius_prior([log_R2,log_teff2]), 0.000001, None)
        return np.log10(rad)
