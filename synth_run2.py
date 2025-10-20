#October 7th - updated syth_run:

############################################################################


#Imports:
import numpy as np
import pandas as pd
import csv
import astropy.io as io
import io

#for emcee and plots
import emcee
import corner
from IPython.display import display, Math
import math
import random
import matplotlib.pyplot as plt

#for parallelization
from multiprocessing import Pool
from multiprocess import Pool
import pickle
import dill

#for dust prior
from astropy.coordinates import SkyCoord
from dustmaps.bayestar import BayestarQuery
import astropy.units as units

#for radius prior
from scipy.stats import gaussian_kde

#checks
import time
import pdb

############################################################################

#Turning off any other potential parallelization operations

import os

os.environ["OMP_NUM_THREADS"] = "1"

###########################################################################

#Functions:

#to get effective wavelength of a given filter:
def filter_eff_wave(filter_name):
    with open('eff_waves.csv', newline='') as data: #opens file with filters and eff waves from svo
        reader = csv.DictReader(data) #reads file as a dictionary

        for row in reader:
            if row['filtername'] == filter_name:
                wave = row['eff_wave'] #gets effective wavelength
                break
            else:
                wave = False  #edit
    return wave

#to get the flux v wavelength points of a system
def get_spec_funct(funct_list, teff, logg, Av, Rv):

    fltr_flxs = {}
    #if meta == None:
    #    #the keys in the funct list are the filter ids
    #    for key in funct_list:
    #        eff_wave = float(filter_eff_wave(key))
    #        filt_flux = float(funct_list[key]([teff,logg, Av, Rv]))
    #        fltr_flxs[key] = {'eff_wave':eff_wave, 'filt_flux':filt_flux}
    #else:
    for key in funct_list:
        eff_wave = float(filter_eff_wave(key))
        filt_flux = float(funct_list[key]([teff,logg, Av, Rv])[0])
        fltr_flxs[key] = {'eff_wave':eff_wave, 'filt_flux':filt_flux}

    #returns dict of flux and eff waves for every filter in a set
    return fltr_flxs

#function for getting SED functions based on a specific spectral library:
def SED_Functs(lib):
    if lib == 'bt-settl-cifist':
        with open(f'lib_function_BTSC_CUBIC.pkl', 'rb') as f:
            functs = pickle.load(f)
    if lib == 'koester2':
        with open(f'lib_function_K2_CUBIC.pkl', 'rb') as f:
            functs = pickle.load(f)
    if lib == 'Kurucz2003':
        with open('lib_function_K2003_CUBIC.pkl', 'rb') as f:
            functs = pickle.load(f)
    return functs

#get samples of d in kpc from p(d):
def samples_d(d_mu, d_sigma, samp_num):
    dist_samples = d_sigma * np.random.randn(samp_num) + d_mu
    dist_samples = [d for d in dist_samples if d>0]
    return dist_samples

#get samples of Av from d:
def samples_Av(l, b, d):
    L = l * units.deg
    B = b * units.deg
    D = d * units.kpc

    #empirical Rv
    Rv_e = 3.1

    #get coordinates from SkyCoord
    coords = SkyCoord(L, B, D, frame='galactic')

    ebv = bayestar(coords, mode='random_sample')

    #scatter ebv:
    Av_delta = 0.2
    st_dev = Av_delta / Rv_e
    scat_ebv = ebv + (st_dev * np.random.randn(len(d)))
    
    return scat_ebv * Rv_e

def mean_Av(l,b,d):
    #define coordinates:
    L = l * units.deg
    B = b * units.deg
    D = d * units.kpc

    #empirical Rv
    Rv_e = 3.1

    #get coordinates from SkyCoord
    coords = SkyCoord(L, B, distance=D, frame='galactic')

    #find mean ebv:
    ebv = bayestar(coords, mode='mean')

    return ebv * Rv_e

############################################################################

#Prerequisites:

#load dustmaps - this step takes ~ 2 min
bayestar = BayestarQuery(version='bayestar2019')

R_kpc_conversion = 5.08327 * 10**-22 #(kpc/R)**2

############################################################################
#files for radius prior <-- move this section later

with open(f'WD_Rad_prior.pkl', 'rb') as f:
    WD_radius_prior = pickle.load(f)

with open('MS_Rad_prior.pkl', 'rb') as f:
    MS_radius_prior = pickle.load(f)
############################################################################

#INPUTS - at the moment for synthesizing sytems
#first function creates the synthetic system you're testing
#second function will take the parameters involved in running emcee with you synthetic data
#all the major imports should occur OUTSIDE of these two callable functions

def input_system(l,b,d_mu,d_sigma,Rv,component_num,lib1,teff1,logg1,R1, lib2, teff2, logg2, R2, yerr_true, yerr_reported):
    """ Creates a synthetic system based on the input parameters """

    #Coordinates:
    l = l                   #galactic coord in degrees
    b = b                   #galactic coord in degrees
    d_mu = d_mu             #distance in kpc
    d_sigma = d_sigma       #distance error

    #dust
    Av_q = mean_Av(l,b,d_mu)
    Rv_q = Rv          #empirical value of Rv


    #Components:
    num_comps = component_num               #number of components (1 or 2)

    lib1 = lib1           #spectral library for first component (hotter)
    teff1 = teff1               #effective temperature in Kelvin
    logg1 = logg1                 #log surface gravity
    meta1 = 0                   #metalicity - set to None for BD or WD models
    R1 = R1                  #radius in solar radii

    lib2 = lib2    #spectral library for second component (cooler)
    teff2 = teff2                #effective temperature in Kelvin
    logg2 = logg2                   #log surface gravity
    R2 = R2                    #radius in solar radii

    #Noise:
    yerr_true = yerr_true            #actual noise added to the flux
    yerr_reported = yerr_reported   #amount of noise reported to emcee for each wavelength


    #Synthesizing system:

    #system flux
    if num_comps ==1:
        funct_list1 = SED_Functs(lib1)
    
        #TEMP - PLEAS INCORP META INTO YOUR PARAMS
        if lib1 =='Kurucz2003':
            flux_dict1 = get_spec_funct(funct_list1, teff1, logg1, meta1, Av_q, Rv_q)
        else:
            flux_dict1 = get_spec_funct(funct_list1, teff1, logg1, Av_q, Rv_q)
    
        funct_list2 = 'NAN'

        #scale values
        R1_sq = R1**2 
        d_sq = d_mu**2

        if lib1 == "koester2":
            wavelengths = []
            log_flux = []
            for key in ["GALEX/GALEX.FUV","GALEX/GALEX.NUV","TYCHO/TYCHO.B","TYCHO/TYCHO.V","Hipparcos/Hipparcos.Hp","PAN-STARRS/PS1.g","PAN-STARRS/PS1.r","PAN-STARRS/PS1.i","PAN-STARRS/PS1.z","PAN-STARRS/PS1.y","SkyMapper/SkyMapper.u","SkyMapper/SkyMapper.v","SkyMapper/SkyMapper.g","SkyMapper/SkyMapper.r","SkyMapper/SkyMapper.i","SkyMapper/SkyMapper.z","GAIA/GAIA0.G","GAIA/GAIA0.Gbp","GAIA/GAIA0.Grp","SLOAN/SDSS.u","SLOAN/SDSS.g","SLOAN/SDSS.r","SLOAN/SDSS.i","SLOAN/SDSS.z","CTIO/DECam.u","CTIO/DECam.g","CTIO/DECam.r","CTIO/DECam.i","CTIO/DECam.z","CTIO/DECam.Y","UKIRT/UKIDSS.Z","UKIRT/UKIDSS.Y","UKIRT/UKIDSS.J","UKIRT/UKIDSS.H","UKIRT/UKIDSS.K","Paranal/VISTA.Z","Paranal/VISTA.Y","Paranal/VISTA.J","Paranal/VISTA.H","Paranal/VISTA.Ks","2MASS/2MASS.H","2MASS/2MASS.J","2MASS/2MASS.Ks","WISE/WISE.W1"]:
                wavelengths.append(flux_dict1[key]['eff_wave'])
                log_flux.append(flux_dict1[key]['filt_flux'] + np.log10(R_kpc_conversion*(R1_sq)/(d_sq)))
        else:
            wavelengths = []
            log_flux = []
            for key in flux_dict1:
                wavelengths.append(flux_dict1[key]['eff_wave'])
                log_flux.append(flux_dict1[key]['filt_flux'] + np.log10( R_kpc_conversion*(R1_sq)/(d_sq)))

    elif num_comps==2:
        funct_list1 = SED_Functs(lib1)
        flux_dict1 = get_spec_funct(funct_list1, teff1, logg1, Av_q, Rv_q)

        funct_list2 = SED_Functs(lib2)
        flux_dict2 = get_spec_funct(funct_list2, teff2, logg2, Av_q, Rv_q)

        #scale values
        R1_sq = R1**2 
        R2_sq = R2**2
        d_sq = d_mu**2

        wavelengths = []
        flux1 = []
        flux2 = []
        for key in flux_dict1:
            wavelengths.append(flux_dict1[key]['eff_wave'])
            flux1.append(flux_dict1[key]['filt_flux'])
            flux2.append(flux_dict2[key]['filt_flux'])
        log_flux = []
        for i in np.arange(len(flux1)):
            log_flux.append(np.log10((( 10**(flux1[i]) * (R1_sq)) + (10**(flux2[i]) * (R2_sq)))  * (R_kpc_conversion/(d_sq))))

    #adding noise to log flux values
    log_flux_scat = log_flux + yerr_true * np.random.randn(len(wavelengths))

    #estimate the amount of error from the spectral models
    log_sigma_true = 0.5 * np.log10(yerr_true**2 - yerr_reported[30]**2)

    truths = [np.log10(teff1), logg1, np.log10(R1), np.log10(teff2), logg2, np.log10(R2), np.log10(d_mu), Av_q, Rv_q, log_sigma_true]
    parameters = l,b,d_mu,d_sigma,Rv,component_num,lib1, teff1, logg1, R1, lib2, teff2, logg2, R2, yerr_true, funct_list1, funct_list2

    #return log_flux_scat, yerr_reported, parameters, truths
    return setup_BBASED(log_flux_scat, yerr_reported, parameters, truths)

#FINISHHH heather you gotta put flux, paramters, errors etc here to be transfered into the emcee function
# Heather, please update the inputs to run_BBASED as well
############################################################################
##
##def log_probability_2(theta):
##        lp = log_prior_2(theta)
##        if not np.isfinite(lp):
##            return -np.inf
##        return lp + log_likelihood_2(theta)

############################################################################

# second function can run on its own? like after the user runs their inputs they initialize the bayesian functions before actually running emcee??

def setup_BBASED(log_flux_scat, yerr_reported, parameters, truths):
    """takes the flux, flux error, and known parameters (number of components and comp type) 
    and establishes model and bayesian functions to be used in sampling"""
    
    #labeling parameters:
    l = parameters[0]
    b = parameters[1]
    d_mu = parameters[2]
    d_sigma = parameters[3] 
    Rv =  parameters[4] 
    num_comps = parameters[5]
    lib1 = parameters[6]
    teff1 = parameters[7]
    logg1 = parameters[8] 
    R1 = parameters[9]
    lib2 = parameters[10]
    teff2 = parameters[11]
    logg2 = parameters[12] 
    R2 = parameters[13]
    yerr_true = parameters[14]
    funct_list1 = parameters[15]
    funct_list2 = parameters[16]

    #parameter limits by spectral library:

    if lib1 == 'bt-settl-cifist':
        teff1_l = np.log10(1200)
        teff1_u = np.log10(7000)
        logg1_l = 2.5
        logg1_u = 5.5
        log_R1_l = -1.5 #units are log solar radii
        log_R1_u = -0.5
    if lib1 == 'koester2':
        teff1_l = np.log10(6000)
        teff1_u = np.log10(20000)
        logg1_l = 6.5
        logg1_u = 9.5
        log_R1_l = -2.09794001
        log_R1_u = -1.69897
    if lib1 == 'Kurucz2003':
        teff1_l = np.log10(3500)
        teff1_u = np.log10(12500)
        logg1_l = 0
        logg1_u = 5
        log_R1_l = -1
        log_R1_u = 1.2

    if lib2 == 'bt-settl-cifist':
        teff2_l = np.log10(1200)
        teff2_u = np.log10(7000)
        logg2_l = 2.5
        logg2_u = 5.5
        log_R2_l = -1.5
        log_R2_u = -0.5
    if lib2 == 'koester2':
        teff2_l = np.log10(6000)
        teff2_u = np.log10(20000)
        logg2_l = 6.5
        logg2_u = 9.5
        log_R2_l = -2.09794001
        log_R2_u = -1.69897
    if lib2 == 'Kurucz2003':
        teff2_l = np.log10(3500)
        teff2_u = np.log10(12500)
        logg2_l = 0
        logg2_u = 5
        log_R2_l = -1
        log_R2_u = 1.2

    ############################################################################
    #build dust-distance prior

    #get samples of distancee & Av
    samp_num = 10000
    d = samples_d(d_mu, d_sigma, samp_num)
    Av = samples_Av(l, b, d)

    Av_mu = np.mean(Av)
    Rv_q = 3.1

    diff_sq_list = []
    for i in Av:
        diff = i - Av_mu
        diff_sq = diff**2
        diff_sq_list.append(diff_sq)

    sum_diff = sum(diff_sq_list)

    Av_sigma_sq = (1/len(Av)) * sum_diff 

    def Av_d_log_prior(Av_unbounded, dist):
        prior = np.log(1/(2*np.pi*d_sigma*(Av_sigma_sq**0.5))) - 0.5*( ((dist-d_mu)/d_sigma)**2 + ((Av_unbounded-Av_mu)**2)/Av_sigma_sq  )
        return prior

    ############################################################################
    #build radii prior - dep on spec library

    #BD library - BT-Settl (CIFIST)
    if lib1 == "bt-settl-cifist":
        r_mu = 0.08529303
        r_sigma = 0.0205525

        def rad1_prior(log_R1,log_teff1):
            return (np.log(1.0/(np.sqrt(2*np.pi)*r_sigma))-0.5*((10**log_R1)-r_mu)**2/r_sigma**2)
    if lib2 == "bt-settl-cifist":
        r_mu = 0.08529303
        r_sigma = 0.0205525

        def rad2_prior(log_R2, log_teff2):
            return (np.log(1.0/(np.sqrt(2*np.pi)*r_sigma))-0.5*((10**log_R2)-r_mu)**2/r_sigma**2)

    #WD library - koester2WD
    if lib1 == "koester2":
        def rad1_prior(log_R1, log_teff1):
            return np.log10(WD_radius_prior([log_R1,log_teff1]))
    if lib2== "koester2":
        def rad2_prior(log_R2, log_teff2):
            return np.log10(WD_radius_prior([log_R2,log_teff2]))

    #MS library - kurucz2003 / ATLAS9
    if lib1 == "Kurucz2003":
        def rad1_prior(log_R1,log_teff1):
            return np.log10(MS_radius_prior([log_R1,log_teff1]))
    if lib2 == "Kurucz2003":
        def rad2_prior(log_R2,log_teff2):
            return np.log10(MS_radius_prior([log_R2,log_teff2]))

    ############################################################################
        ############################################################################

    #position and labels based on component count and library selection:
    if num_comps == 1:
        labels = ['log_teff1', 'logg1', 'log_R1', 'log_dist', 'Av', 'Rv', 'log10_sigma_theory']     #parameter labels
        pos = [random.uniform(teff1_l, teff1_u), random.uniform(logg1_l, logg1_u), random.uniform(log_R1_l, log_R1_l), np.log10(d_mu), 0.9, 3.1, -1.5 ]
    if num_comps == 2:
        labels = ['log_teff1', 'logg1', 'log_R1', 'log_teff2', 'logg2', 'log_R2', 'log_dist', 'Av', 'Rv', 'log10_sigma_theory']
        teff1_guess = random.uniform(teff1_l, teff1_u)
        teff2_guess = random.uniform(teff2_l, teff2_u)
        #need to make sure that teff1 > teff2, even in our guess? sure i guess
        ginger = "good dog"
        while ginger == "good dog":
            if teff1_guess > teff2_guess:
                ginger = "very good dog"
            elif teff1_guess < teff2_guess:
                teff1_guess = random.uniform(teff1_l, teff1_u)
                teff2_guess = random.uniform(teff2_l, teff2_u)
        pos = [teff1_guess, random.uniform(logg1_l, logg1_u), random.uniform(log_R1_l, log_R1_l), teff2_guess, random.uniform(logg2_l, logg2_u), random.uniform(log_R2_l, log_R2_u), np.log10(d_mu), 0.9, 3.1, -1.5 ]

    ############################################################################

    #feed into emcee

    args = log_flux_scat, yerr_reported 


    #functions

    def log_likelihood_1(theta):

        log10_y, yerr_dex = args

        #parameters
        log_teff1, logg1, met, log_R1, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

        #clip negative values of Av to 0:
        Av_clipped = np.clip(Av_unbounded, 0, None)
    
        #TEMP
        if lib1 =='Kurucz2003':
            flx_dict1 = get_spec_funct(funct_list1, 10**(log_teff1), logg1, met, Av_clipped, Rv)
        else:
            #dict of [filter_id] : eff_wavelengths, filter_flux_values
            flx_dict1 = get_spec_funct(funct_list1, 10**(log_teff1), logg1, Av_clipped, Rv) #note here we're specifying the spectral library ourselves

        R1_sq = ((10**log_R1)**2)
        dist_sq = ((10**log_dist)**2)

        #log value of the flux
        if lib1 == "koester2":
            wavelengths = []
            model = []
            for key in ["GALEX/GALEX.FUV","GALEX/GALEX.NUV","TYCHO/TYCHO.B","TYCHO/TYCHO.V","Hipparcos/Hipparcos.Hp","PAN-STARRS/PS1.g","PAN-STARRS/PS1.r","PAN-STARRS/PS1.i","PAN-STARRS/PS1.z","PAN-STARRS/PS1.y","SkyMapper/SkyMapper.u","SkyMapper/SkyMapper.v","SkyMapper/SkyMapper.g","SkyMapper/SkyMapper.r","SkyMapper/SkyMapper.i","SkyMapper/SkyMapper.z","GAIA/GAIA0.G","GAIA/GAIA0.Gbp","GAIA/GAIA0.Grp","SLOAN/SDSS.u","SLOAN/SDSS.g","SLOAN/SDSS.r","SLOAN/SDSS.i","SLOAN/SDSS.z","CTIO/DECam.u","CTIO/DECam.g","CTIO/DECam.r","CTIO/DECam.i","CTIO/DECam.z","CTIO/DECam.Y","UKIRT/UKIDSS.Z","UKIRT/UKIDSS.Y","UKIRT/UKIDSS.J","UKIRT/UKIDSS.H","UKIRT/UKIDSS.K","Paranal/VISTA.Z","Paranal/VISTA.Y","Paranal/VISTA.J","Paranal/VISTA.H","Paranal/VISTA.Ks","2MASS/2MASS.H","2MASS/2MASS.J","2MASS/2MASS.Ks","WISE/WISE.W1"]:
                wavelengths.append(flx_dict1[key]['eff_wave'])
                model.append(flx_dict1[key]['filt_flux'] + np.log10( R_kpc_conversion*(R1_sq)/(dist_sq)))
        else:
            wavelength = []
            model = []
            for key in flx_dict1:
                model.append(flx_dict1[key]['filt_flux'] + np.log10(R_kpc_conversion * R1_sq / dist_sq))#edit for logform
                wavelength.append(flx_dict1[key]['eff_wave'])

        #check for kurucz uneven grid, parameters that fall within range
        if math.isnan(model[0])==True:
            return -np.inf


        #deviations
        sigma_theory_dex = 10**log10_sigma_theor
        sigma_dex2 = []
        for i in np.arange(len(yerr_dex)):
            sigma_dex2.append((yerr_dex[i])**2 + sigma_theory_dex**2)

        return np.sum(-0.5 * ((log10_y - model) ** 2) / sigma_dex2 - 0.5*np.log(sigma_dex2)) 


    def log_prior_1(theta):

        log_teff1, logg1, met,  log_R1, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

        #parameter ranges
        if not (teff1_l <= log_teff1 <= teff1_u and logg1_l <= logg1 <= logg1_u and -2.5 <= met <= 0.5 and log_R1_l<=log_R1<=log_R1_u and -3<= log_dist <= 1 and Av_unbounded <= 5 and 2.5<=Rv<=5.5 and -2 < log10_sigma_theor < 0):
            return -np.inf

        #param priors:
        def sigma_theor_log_prior(log10_sigma_theor):
            return (-(np.log(10))*log10_sigma_theor)

        def Rv_log_prior(Rv):
            sigma_rv = 0.18
            return (np.log(1.0/(np.sqrt(2*np.pi)*sigma_rv))-0.5*(Rv-Rv_q)**2/sigma_rv**2)

        Jacobian_log = np.log( (10**log_dist) * (3.1 / Rv) )

        return sigma_theor_log_prior(log10_sigma_theor) + Rv_log_prior(Rv) + Av_d_log_prior(Av_unbounded, (10**log_dist)) + Jacobian_log + rad1_prior(log_R1,log_teff1)


    def log_probability_1(theta):
        lp = log_prior_1(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood_1(theta)


    def log_likelihood_2(theta):

        log10_y, yerr_dex = args

        #parameters
        log_teff1, logg1, log_R1, log_teff2, logg2, log_R2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta
    
        #clip negative values of Av to 0:
        Av_clipped = np.clip(Av_unbounded, 0, None)

        #dict of [filter_id] : eff_wavelengths, filter_flux_values
        flx_dict1 = get_spec_funct(funct_list1, 10**(log_teff1), logg1, Av_clipped, Rv)
        flx_dict2 = get_spec_funct(funct_list2, 10**(log_teff2), logg2, Av_clipped, Rv)

        R1_sq = ((10**log_R1)**2)
        R2_sq = ((10**log_R2)**2)
        dist_sq = ((10**log_dist)**2)

        #log value of the flux
        #model is the values of filter_flux from the dict
        model = []
        for key in flx_dict1:
            model.append(np.log10((( 10**(flx_dict1[key]['filt_flux']) * (R1_sq)) + (10**(flx_dict2[key]['filt_flux']) * (R2_sq)))  * (R_kpc_conversion/(dist_sq))))
        #check for kurucz uneven grid
        if math.isnan(model[0])==True:
            return -np.inf

        #deviations
        sigma_theory_dex = 10**log10_sigma_theor
        sigma_dex2 = []
        for i in np.arange(len(yerr_dex)):
            sigma_dex2.append((yerr_dex[i])**2 + sigma_theory_dex**2)

        return np.sum(-0.5 * ((log10_y - model) ** 2) / sigma_dex2 - 0.5*np.log(sigma_dex2))


    def log_prior_2(theta):
        log_teff1, logg1, log_R1, log_teff2, logg2, log_R2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta
    
        #parameter ranges
        if not (teff1_l <= log_teff1 <= teff1_u and logg1_l <= logg1 <= logg1_u and log_R1_l<=log_R1<=log_R1_u and teff2_l <= log_teff2 <= teff2_u and log_teff2 < log_teff1 and logg2_l <= logg2 <= logg2_u and log_R2_l<=log_R2<=log_R2_u and -3<= log_dist <= 1.5 and Av_unbounded <= 5.0 and 2.5<=Rv<=5.5 and -2 < log10_sigma_theor < 0):
            return -np.inf
    
        #param priors
        def sigma_theor_log_prior(log10_sigma_theor):
            return (-(np.log(10))*log10_sigma_theor)

        def Rv_log_prior(Rv):
            sigma_rv = 0.18
            return (np.log(1.0/(np.sqrt(2*np.pi)*sigma_rv))-0.5*(Rv-Rv_q)**2/sigma_rv**2)

        Jacobian_log = np.log( (10**log_dist) * (3.1 / Rv) )

        return sigma_theor_log_prior(log10_sigma_theor) + Rv_log_prior(Rv) + Av_d_log_prior(Av_unbounded, (10**log_dist)) + Jacobian_log + rad1_prior(log_R1,log_teff1) + rad2_prior(log_R2,log_teff2)


    def log_probability_2(theta):
        lp = log_prior_2(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood_2(theta)

    functions = log_likelihood_1, log_likelihood_2, log_prior_1, log_prior_2, log_probability_1, log_probability_2 

    return (pos, functions)

    ############################################################################

def run_BBASED(inputs, walkers, sample_num, param_num, pool=None):
    """runs emcee with the log likelihood, log prior, and log prob functions established by data
        takes:
            number of walkers (128 = single star, 256 = binary star),
            sample number (3000 works well)
            param_num (depends on the number of stars in your system and the number of model parameters)
        returns:
            the chains from emcee which can be interpreted by the outputs function,
            the prior values, and
            the corresponding labels"""
    #defining inputs
    pos = inputs[0]
    log_likelihood_1 = inputs[1][0]
    log_likelihood_2 = inputs[1][1]
    log_prior_1 = inputs[1][2]
    log_prior_2 = inputs[1][3]
    log_probability_1 = inputs[1][4]
    log_probability_2 = inputs[1][5]

    if param_num > 9:
        num_comps = 2
    elif param_num < 9:
        num_comps = 1

    #emcee
    if pool is None:
        with Pool() as pool:

            pos = pos + 1e-2 * np.random.randn(walkers, param_num)
            nwalkers, ndim = pos.shape


            if num_comps == 1:
                emcee_results = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_1, pool=pool
                )

                prior_emcee = emcee.EnsembleSampler(
                    nwalkers, ndim, log_prior_1, pool=pool
                )

            if num_comps == 2:
                emcee_results = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_2, pool=pool
                )

                prior_emcee = emcee.EnsembleSampler(
                    nwalkers, ndim, log_prior_2, pool=pool
                )

            emcee_results.run_mcmc(pos, sample_num, progress=True);
            prior_emcee.run_mcmc(pos, sample_num, progress=True);

    ############################################################################
    
    return emcee_results, prior_emcee, labels

