""" Selects model libraries for the components - can either be 1 or 2 models
    If selecting two models for a binary, the order matters!
    The first model will be the component with a higher effective temperature.

    The availble models are:
    "bt-settl-cifist" - the brown dwarf atmospheric models from Allard et al. 2011
    "koester2" - the white dwarf atmospheric models from Koester et al. 2010
    "Kurucz2003" - the main sequence ATLAS9 model atmospheres from Castelli & Kurucz 2003

    There are future plans to increase the number of available models / release
    documentation for how to integrate your own models into the code

"""
import numpy as np
import random

def select_model(Model_1, Model_2=None):
    """ Model input to run BBASED with
        At least one model is required (if comparing your data to a single star system)
        Up to two models are allowed, the order of these matters, the first model you
        input will be the binary component with a higher effective temperature

        The availble models are:
        "bt-settl-cifist" - the brown dwarf atmospheric models from Allard et al. 2011
        "koester2" - the white dwarf atmospheric models from Koester et al. 2010
        "Kurucz2003" - the main sequence ATLAS9 model atmospheres from Castelli & Kurucz 2003

        There are future plans to increase the number of available models / release
        documentation for how to integrate your own models into the code """

    lib1 = Model_1

    if Model_2:
        lib2 = Model_2
    else:
        lib2 = "NAN"

    return lib1, lib2


def set_param_lim(lib1, lib2):
    """ This function will set the parameter limits for the system based on the selected models
        This will be important for the prior function used during sampling and setting the
        number of parameteres (as models vary in dimension) and the initial positions of the
        walkers used during sampling """

    if lib1 == 'bt-settl-cifist':
        teff1_l = np.log10(1200)    # log of T_eff (K)
        teff1_u = np.log10(7000)
        logg1_l = 2.5               # log surface gravity
        logg1_u = 5.5
        log_R1_l = -1.5             # log radius (solar radii)
        log_R1_u = -0.5
        meta1_l = None              # metallicity
        meta1_u = None
    if lib1 == 'koester2':
        teff1_l = np.log10(6000)
        teff1_u = np.log10(20000)
        logg1_l = 6.5
        logg1_u = 9.5
        log_R1_l = -2.09794001
        log_R1_u = -1.69897
        meta1_l = None
        meta1_u = None
    if lib1 == 'Kurucz2003':
        teff1_l = np.log10(3500)
        teff1_u = np.log10(12500)
        logg1_l = 0
        logg1_u = 5
        log_R1_l = -1
        log_R1_u = 1.2
        meta1_l = -2.5              
        meta1_u =  0.5

    if lib2 == 'bt-settl-cifist':
        teff2_l = np.log10(1200)
        teff2_u = np.log10(7000)
        logg2_l = 2.5
        logg2_u = 5.5
        log_R2_l = -1.5
        log_R2_u = -0.5
        meta2_l = None
        meta2_u = None
    if lib2 == 'koester2':
        teff2_l = np.log10(6000)
        teff2_u = np.log10(20000)
        logg2_l = 6.5
        logg2_u = 9.5
        log_R2_l = -2.09794001
        log_R2_u = -1.69897
        meta2_l = None
        meta2_u = None
    if lib2 == 'Kurucz2003':
        teff2_l = np.log10(3500) 
        teff2_u = np.log10(12500)
        logg2_l = 0
        logg2_u = 5
        log_R2_l = -1
        log_R2_u = 1.2
        meta2_l = -2.5
        meta2_u  = 0.5


    if lib2 != "NAN":   #binary star system
        return (teff1_l, teff1_u, logg1_l, logg1_u, log_R1_l, log_R1_u, meta1_l, meta1_u,
                teff2_l, teff2_u, logg2_l, logg2_u, log_R2_l, log_R2_u, meta2_l, meta2_u)
    elif lib2 == "NAN": #single star system
        return teff1_l, teff1_u, logg1_l, logg1_u, log_R1_l, log_R1_u, meta1_l, meta1_u

def set_labels(lib1, lib2):
    """ This function will set the labels used for the graphical representation of the BBASED
        output - this is determined by the parameter types in each model"""

    label_dict1 = {}
    label_dict1["bt-settl-cifist"] = ['log_teff1', 'logg1', 'log_R1']
    label_dict1["koester2"] = ['log_teff1', 'logg1', 'log_R1']
    label_dict1["Kurucz2003"] = ['log_teff1', 'logg1', 'log_R1', 'meta1']

    label_dict2 = {}
    label_dict2["bt-settl-cifist"] = ['log_teff2', 'logg2', 'log_R2']
    label_dict2["koester2"] = ['log_teff2', 'logg2', 'log_R2']
    label_dict2["Kurucz2003"] = ['log_teff2', 'logg2', 'log_R2', 'meta2']

    if lib2 != "NAN":   #binary star system
        labels = label_dict1[lib1] + label_dict2[lib2] + ['log_dist', 'Av', 'Rv', 'log_sigma_theory']
    elif lib2 == "NAN": #single star system
        labels = label_dict1[lib1]  + ['log_dist', 'Av', 'Rv', 'log_sigma_theory']

    return labels


def set_param_num(labels):
    """ This function sets the parameter number of the system. This is important for the sampling
        process and varies with model choice """

    param_num = len(labels)

    return param_num #easypeasylemonsquezy

def set_pos(lib1, lib2, d_mu):
    """ The sampling algorithm our program uses requires us to set an initial position for our
        walkers to begin. We choose to randomly select this position based on the parameter limits
        as determined by the model selections

        Setting the initial pos will require we know d_mu <-- mean dist set by the plx"""
        #apologies to my suprvisor for how wack this function is...
    
    if lib1 == 'bt-settl-cifist':
        teff1_l = np.log10(1200)    # log of T_eff (K)
        teff1_u = np.log10(7000)
        logg1_l = 2.5               # log surface gravity
        logg1_u = 5.5
        log_R1_l = -1.5             # log radius (solar radii)
        log_R1_u = -0.5
        meta1_l = None              # metallicity
        meta1_u = None
    if lib1 == 'koester2':
        teff1_l = np.log10(6000)
        teff1_u = np.log10(20000)
        logg1_l = 6.5
        logg1_u = 9.5
        log_R1_l = -2.09794001
        log_R1_u = -1.69897
        meta1_l = None
        meta1_u = None
    if lib1 == 'Kurucz2003':
        teff1_l = np.log10(3500)
        teff1_u = np.log10(12500)
        logg1_l = 0
        logg1_u = 5
        log_R1_l = -1
        log_R1_u = 1.2
        meta1_l = -2.5              
        meta1_u =  0.5

    if lib2 == 'bt-settl-cifist':
        teff2_l = np.log10(1200)
        teff2_u = np.log10(7000)
        logg2_l = 2.5
        logg2_u = 5.5
        log_R2_l = -1.5
        log_R2_u = -0.5
        meta2_l = None
        meta2_u = None
    if lib2 == 'koester2':
        teff2_l = np.log10(6000)
        teff2_u = np.log10(20000)
        logg2_l = 6.5
        logg2_u = 9.5
        log_R2_l = -2.09794001
        log_R2_u = -1.69897
        meta2_l = None
        meta2_u = None
    if lib2 == 'Kurucz2003':
        teff2_l = np.log10(3500) 
        teff2_u = np.log10(12500)
        logg2_l = 0
        logg2_u = 5
        log_R2_l = -1
        log_R2_u = 1.2
        meta2_l = -2.5
        meta2_u  = 0.5


    if lib2 != "NAN":   #binary star system
        teff1_guess = random.uniform(teff1_l, teff1_u)
        teff2_guess = random.uniform(teff2_l, teff2_u)
        #need to make sure that teff1 > teff2, even in our guess
        ginger = "good dog"
        while ginger == "good dog":
            if teff1_guess > teff2_guess:
                ginger = "very good dog"
            elif teff1_guess < teff2_guess:
                teff1_guess = random.uniform(teff1_l, teff1_u)
                teff2_guess = random.uniform(teff2_l, teff2_u)

        if meta1_l and meta2_l:   #both components have a met. parameter
            pos = [teff1_guess,
                   random.uniform(logg1_l, logg1_u),
                   random.uniform(log_R1_l, log_R1_u),
                   random.uniform(meta1_l, meta1_u),
                   teff2_guess,
                   random.uniform(logg2_l, logg2_u),
                   random.uniform(log_R2_l, log_R2_u),
                   random.uniform(meta2_l, meta2_u),
                   np.log10(d_mu),
                   0.9,
                   3.1,
                   -1.5 ]
        elif meta1_l:           #only the first component has a met. parameter
            pos = [teff1_guess,
                    random.uniform(logg1_l, logg1_u),
                   random.uniform(log_R1_l, log_R1_u),
                   random.uniform(meta1_l, meta1_u),
                   teff2_guess,
                   random.uniform(logg2_l, logg2_u),
                   random.uniform(log_R2_l, log_R2_u),
                   np.log10(d_mu),
                   0.9,
                   3.1,
                   -1.5 ]
        elif meta2_l:           #only the second component has a met. parameter
             pos = [teff1_guess,
                   random.uniform(logg1_l, logg1_u),
                   random.uniform(log_R1_l, log_R1_u),
                   teff2_guess,
                   random.uniform(logg2_l, logg2_u),
                   random.uniform(log_R2_l, log_R2_u),
                   random.uniform(meta2_l, meta2_u),
                   np.log10(d_mu),
                   0.9,
                   3.1,
                   -1.5 ]
        else:                   #neither component has a met. parameter
            pos = [teff1_guess,
                   random.uniform(logg1_l, logg1_u),
                   random.uniform(log_R1_l, log_R1_u),
                   teff2_guess,
                   random.uniform(logg2_l, logg2_u),
                   random.uniform(log_R2_l, log_R2_u),
                   np.log10(d_mu),
                   0.9,
                   3.1,
                   -1.5 ]

    elif lib2 == "NAN": #single star system
        pos = [random.uniform(teff1_l, teff1_u),
               random.uniform(logg1_l, logg1_u),
               random.uniform(log_R1_l, log_R1_u),
               np.log10(d_mu),
               0.9,
               3.1,
               -1.5 ]

    return pos

    
