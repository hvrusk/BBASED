#September 1st - Updated Program
#Updated test_prgrm with improved formatting, no input prompts
#inputs are manually inputted by filter

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
import pickle

#for dust prior
from astropy.coordinates import SkyCoord
from dustmaps.bayestar import BayestarQuery
import astropy.units as units

#for collecting system info:
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from astropy.table import Table
from astroquery.ipac.irsa import Irsa
from astroquery.sdss import SDSS
from astroquery.ukidss import Ukidss
from astropy import coordinates as Coords

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
def get_spec_funct(funct_list, teff, logg, Av, Rv, meta="na"):
    fltr_flxs = {}
    if meta == "na":
        for key in filts:
            eff_wave = float(filter_eff_wave(key))
            filt_flux = float(funct_list[key]([teff,logg, Av, Rv])[0])
            fltr_flxs[key] = {'eff_wave':eff_wave, 'filt_flux':filt_flux}
    else: #case where we have meta is when we're using the MS models
        for key in filts:
            eff_wave = float(filter_eff_wave(key))
            filt_flux = float(funct_list[key]([teff,logg,meta,Av, Rv])[0])
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

#new functions for collecting source information:

#get plx & gal coord from ra and dec:
def get_plx(ra, dec):
    job = Gaia.launch_job(f"""SELECT source_id, ra, dec, l, b, parallax, parallax_error
                      FROM gaiadr3.gaia_source 
                      WHERE 
                      ra BETWEEN {ra-0.002} AND {ra+0.001} AND
                      dec BETWEEN {dec-0.01} AND {dec+0.01}""")#check ra and dec
    results = job.get_results()
    
    result_plx = float(results["parallax"][0]) #in some cases gaia might return a negative parallax (Lindegren 2018)
    results_plx_err = float(results["parallax_error"][0])
    
    result_l = results["l"][0]
    result_b = results["b"][0]
   
    return result_plx, results_plx_err, result_l, result_b

#get object id from gal coord:
def get_info_gal(l,b):
    table = Simbad.query_region(SkyCoord(l, b, unit=(units.deg, units.deg),frame='galactic'), radius=2 * units.arcsec)
    main_id = table['MAIN_ID'][0]
    return main_id

#get info from object name:
def source_info(object_name):

    result_table = Simbad.query_objectids(object_name) #accepts any object name

    for x in result_table:
        if 'gaia dr3' in x['id'].lower():
            gaia_source_id = eval(x['id'].replace("Gaia DR3 ",""))
    # then we create an ADQL query
    job = Gaia.launch_job(f"""SELECT source_id, ra, dec, l, b, parallax, parallax_error
                      FROM gaiadr3.gaia_source 
                      WHERE 
                      source_id = {gaia_source_id}""")
    results = job.get_results()
    
    return results

#function to combine above to get relevant source info from either coord or object name:
#info = object name or (coords)
#method = ICRS_coord, gal_coord, or source_name

def get_source(info, method):
    if method == "ICRS_coord":
        query = Gaia.launch_job("SELECT TOP 10 "
                      "source_id,ra,dec,l,b,parallax, parallax_error, "
                      "phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag "
                      "FROM gaiadr3.gaia_source as gaia "
                      f"WHERE gaia.ra = {info[0]} AND "
                      f" gaia.dec = {info[1]}")
        table = query.get_results()
        main_id = table['source_id'][0]
        l = table['l'][0]
        b = table['b'][0]
        ra = info[0]
        dec = info[1]
        plx = (table['parallax'][0],table['parallax_error'][0])

    if method == "gal_coord":
        l = info[0]
        b = info[1]
        main_id = get_info_gal(l,b)
        data = source_info(main_id)
        ra = data["ra"][0]
        dec = data["dec"][0]
        plx = get_plx(ra,dec)
        
    if method =="source_name":
        data = source_info(info)
        main_id = data['source_id'][0]
        ra = data["ra"][0]
        dec = data["dec"][0]
        plx = data['parallax'][0], data['parallax_error'][0]
        l = data['l'][0]
        b = data['b'][0]
    
    #parallax & gal coord
    parallax = plx[0]
    parallax_err = plx[1]

    
    return main_id, l, b, parallax, parallax_err, ra, dec

#taking the lowest error flux for a filter
def pick_flux(flux, eflux): #should take all the flux and eflux values associated with a single filter
    flux1 = flux[0]
    eflux1 = eflux[0]
    for f in np.arange(len(flux)-1):
        if eflux1 > eflux[f+1]:
            flux1 = flux[f+1]
            eflux1 = eflux[f+1]
            
    return flux1, eflux1

#creates an SED the same way the synth function would but from real data
def get_SED(main_id):
    #the following will collect the magnitudes as listed in each catalogue
    #WISE:
    job = Gaia.launch_job("SELECT gaia.source_id, gaia.ra, gaia.dec, gaia.parallax, gaia.pmra, gaia.pmdec, allwise.* "
                      "FROM gaiadr3.gaia_source AS gaia "
                      "JOIN gaiadr3.allwise_best_neighbour AS xmatch USING (source_id) "
                      "JOIN gaiadr1.allwise_original_valid AS allwise "
                      "ON xmatch.original_ext_source_id = allwise.designation "
                      f"WHERE gaia.source_id = {main_id}")
    results_WISE =  job.get_results()


    #SDSS:
    job = Gaia.launch_job("SELECT TOP 100 gaia.source_id, gaia.ra, gaia.dec, gaia.parallax, gaia.pmra, gaia.pmdec, sdss.ra, sdss.dec, sdss.u, sdss.g, sdss.r, sdss.i, sdss.z, sdss.err_u, sdss.err_g, sdss.err_r, sdss.err_i, sdss.err_z " 
        "FROM gaiadr3.gaia_source AS gaia "
        "JOIN gaiadr3.sdssdr13_best_neighbour AS xmatch USING (source_id) "
        "JOIN gaiadr3.sdssdr13_join AS xjoin USING (clean_sdssdr13_oid) "
        "JOIN external.sdssdr13_photoprimary AS sdss "
        "ON xjoin.original_ext_source_id = sdss.objid "
        f"WHERE gaia.source_id = {main_id}")
    results_SDSS = job.get_results()

    #Pan-STARRS:
    job = Gaia.launch_job("SELECT TOP 100 gaia.source_id, gaia.ra, gaia.dec, gaia.parallax, gaia.pmra, gaia.pmdec, ps.* "
        "FROM gaiadr3.gaia_source AS gaia "
        "JOIN gaiadr3.panstarrs1_best_neighbour AS xmatch USING (source_id) "
        "JOIN gaiadr3.panstarrs1_join AS xjoin USING (clean_panstarrs1_oid) "
        "JOIN gaiadr2.panstarrs1_original_valid AS ps "
        "ON xjoin.original_ext_source_id = ps.obj_id "
        f"WHERE gaia.source_id = {main_id}")
    results_PS = job.get_results()

    #SkyMapper:
    job = Gaia.launch_job("SELECT TOP 100 gaia.source_id, gaia.ra, gaia.dec, gaia.parallax, gaia.pmra, gaia.pmdec, SM.raj2000, SM.dej2000, SM.u_psf, SM.v_psf, SM.g_psf, SM.r_psf, SM.i_psf, SM.z_psf, SM.e_u_psf, SM.e_v_psf, SM.e_g_psf, SM.e_r_psf, SM.e_i_psf, SM.e_z_psf " 
        "FROM gaiadr3.gaia_source AS gaia "
        "JOIN gaiadr3.skymapperdr2_best_neighbour AS xmatch USING (source_id) "
        "JOIN gaiadr3.skymapperdr2_join AS xjoin USING(original_ext_source_id) "
        "JOIN external.skymapperdr2_master AS SM ON xjoin.original_ext_source_id = SM.object_id "
        f"WHERE gaia.source_id = {main_id}")
    results_SM = job.get_results()
    
    #UKIDSS:
    #results_UKIDSS = Ukidss.query_region(Coords.SkyCoord(ra, dec,
                                           #unit=(units.deg, units.deg),
                                           #frame='icrs'),
                                           # radius=5 * units.arcsec,
                                           #database='UKIDSSDR8PLUS',
                                           # programme_id='LAS')
    #GAIA:
    job = Gaia.launch_job("SELECT TOP 10 "
                      "ra,dec,parallax, "
                      "phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag "
                      "FROM gaiadr3.gaia_source as gaia "
                      f"WHERE gaia.source_id = {main_id}")
    results_Gaia = job.get_results()

    #2MASS:
    job = Gaia.launch_job("SELECT gaia.source_id, gaia.ra, gaia.dec, gaia.parallax, gaia.pmra, gaia.pmdec, tmass.* "
        "FROM gaiadr3.gaia_source AS gaia "
        "JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch USING (source_id) "
        "JOIN gaiadr3.tmass_psc_xsc_join AS xjoin USING (clean_tmass_psc_xsc_oid) "
        "JOIN gaiadr1.tmass_original_valid AS tmass ON xjoin.original_psc_source_id = tmass.designation "
        f"WHERE gaia.source_id = {main_id}")
    results_2MASS = job.get_results()

    #quality check:
    quality_check = {}
    if len(results_2MASS)>0:
        quality_check['2MASS/2MASS.J'] = results_2MASS['ph_qual'][0][0]
        quality_check['2MASS/2MASS.H'] = results_2MASS['ph_qual'][0][1]
        quality_check['2MASS/2MASS.Ks'] = results_2MASS['ph_qual'][0][2]
    if len(results_WISE)>0:
        quality_check['WISE/WISE.W1'] = results_WISE['ph_qual'][0][0]
        quality_check['WISE/WISE.W2'] = results_WISE['ph_qual'][0][1]

    #retriving and organizing dict of mag with filter names:
    mag_dict = {}
    #Gaia
    mag_dict['GAIA/GAIA0.G']={'mag':results_Gaia['phot_g_mean_mag'][0], 'sig':0}
    mag_dict['GAIA/GAIA0.Gbp']={'mag':results_Gaia['phot_bp_mean_mag'][0], 'sig':0}
    mag_dict['GAIA/GAIA0.Grp']={'mag':results_Gaia['phot_rp_mean_mag'][0], 'sig':0}
    
    #UKIDSS
    #if len(results_UKIDSS) > 0:
    #    mag_dict['UKIRT/UKIDSS.Y']={'mag':results_UKIDSS['YAperMag3'][0], 'sig':0}
    #    mag_dict['UKIRT/UKIDSS.J']={'mag':results_UKIDSS['J_1AperMag3'][0], 'sig':0}
    #    mag_dict['UKIRT/UKIDSS.H']={'mag':results_UKIDSS['HAperMag3'][0], 'sig':0}
    #    mag_dict['UKIRT/UKIDSS.K']={'mag':results_UKIDSS['KAperMag3'][0], 'sig':0}
    
    #SDSS
    if len(results_SDSS) > 0:
        mag_dict['SLOAN/SDSS.u']={'mag':results_SDSS['u'][0], 'sig':results_SDSS['err_u'][0]}
        mag_dict['SLOAN/SDSS.g']={'mag':results_SDSS['g'][0], 'sig':results_SDSS['err_g'][0]}
        mag_dict['SLOAN/SDSS.r']={'mag':results_SDSS['r'][0], 'sig':results_SDSS['err_r'][0]}
        mag_dict['SLOAN/SDSS.i']={'mag':results_SDSS['i'][0], 'sig':results_SDSS['err_i'][0]}
        mag_dict['SLOAN/SDSS.z']={'mag':results_SDSS['z'][0], 'sig':results_SDSS['err_z'][0]}
    
    #PAN-STARRS
    if len(results_PS) > 0:
        mag_dict['PAN-STARRS/PS1.g']={'mag':results_PS['g_mean_psf_mag'][0], 'sig':results_PS['g_mean_psf_mag_error'][0]}
        mag_dict['PAN-STARRS/PS1.r']={'mag':results_PS['r_mean_psf_mag'][0], 'sig':results_PS['r_mean_psf_mag_error'][0]}
        mag_dict['PAN-STARRS/PS1.i']={'mag':results_PS['i_mean_psf_mag'][0], 'sig':results_PS['i_mean_psf_mag_error'][0]}
        mag_dict['PAN-STARRS/PS1.z']={'mag':results_PS['z_mean_psf_mag'][0], 'sig':results_PS['z_mean_psf_mag_error'][0]}
        mag_dict['PAN-STARRS/PS1.y']={'mag':results_PS['y_mean_psf_mag'][0], 'sig':results_PS['y_mean_psf_mag_error'][0]}
    
    #SkyMapper
    if len(results_SM) > 0:
        mag_dict['SkyMapper/SkyMapper.v']={'mag':results_SM['v_psf'][0], 'sig':results_SM['e_v_psf'][0]}
        mag_dict['SkyMapper/SkyMapper.u']={'mag':results_SM['u_psf'][0], 'sig':results_SM['e_u_psf'][0]}
        mag_dict['SkyMapper/SkyMapper.g']={'mag':results_SM['g_psf'][0], 'sig':results_SM['e_g_psf'][0]}
        mag_dict['SkyMapper/SkyMapper.r']={'mag':results_SM['r_psf'][0], 'sig':results_SM['e_r_psf'][0]}
        mag_dict['SkyMapper/SkyMapper.i']={'mag':results_SM['i_psf'][0], 'sig':results_SM['e_i_psf'][0]}
        mag_dict['SkyMapper/SkyMapper.z']={'mag':results_SM['z_psf'][0], 'sig':results_SM['e_z_psf'][0]}
    
    #2MASS
    if len(results_2MASS)>0:    
        mag_dict['2MASS/2MASS.J']={'mag':results_2MASS['j_m'][0], 'sig':results_2MASS['j_msigcom'][0]}
        mag_dict['2MASS/2MASS.H']={'mag':results_2MASS['h_m'][0], 'sig':results_2MASS['h_msigcom'][0]}
        mag_dict['2MASS/2MASS.Ks']={'mag':results_2MASS['ks_m'][0], 'sig':results_2MASS['ks_msigcom'][0]}
    #WISE
    if len(results_WISE)>0:    
        mag_dict['WISE/WISE.W1']={'mag':results_WISE['w1mpro'][0], 'sig':results_WISE['w1mpro_error'][0]}
        #mag_dict['WISE/WISE.W2']={'mag':results_WISE['w2mpro'][0], 'sig':results_WISE['w2mpro_error'][0]}
    
        for i in np.arange(len(results_WISE['ph_qual'][0])):
            quality_check[f'WISE/WISE.W{i+1}'] = results_WISE['ph_qual'][0][i]

    #zero point dict - maybe this can be appended to the filter-eff_wave dict
    #units of erg/cm^2/s/A
    zeropoints = {}
    zeropoints['GAIA/GAIA0.G'] = 2.50*10**-9
    zeropoints['GAIA/GAIA0.Gbp'] = 4.11*10**-9
    zeropoints['GAIA/GAIA0.Grp'] = 1.24*10**-9

    zeropoints['UKIRT/UKIDSS.Y'] = 5.73*10**-10
    zeropoints['UKIRT/UKIDSS.J'] = 2.94*10**-10
    zeropoints['UKIRT/UKIDSS.H'] = 1.15*10**-10
    zeropoints['UKIRT/UKIDSS.K'] = 3.90*10**-11

    zeropoints['SLOAN/SDSS.u'] = 3.75*10**-9
    zeropoints['SLOAN/SDSS.g'] = 5.45*10**-9
    zeropoints['SLOAN/SDSS.r'] = 2.50*10**-9
    zeropoints['SLOAN/SDSS.i'] = 1.39*10**-9
    zeropoints['SLOAN/SDSS.z'] = 8.39*10**-10
    
    zeropoints['PAN-STARRS/PS1.g'] = 5.05*10**-9
    zeropoints['PAN-STARRS/PS1.r'] = 2.47*10**-9
    zeropoints['PAN-STARRS/PS1.i'] = 1.36*10**-9
    zeropoints['PAN-STARRS/PS1.z'] = 9.01*10**-10
    zeropoints['PAN-STARRS/PS1.y'] = 7.05*10**-10

    zeropoints['SkyMapper/SkyMapper.v'] = 5.45*10**-9
    zeropoints['SkyMapper/SkyMapper.u'] = 3.23*10**-9
    zeropoints['SkyMapper/SkyMapper.g'] = 4.48*10**-9
    zeropoints['SkyMapper/SkyMapper.r'] = 2.57*10**-9
    zeropoints['SkyMapper/SkyMapper.i'] = 1.24*10**-9
    zeropoints['SkyMapper/SkyMapper.z'] = 7.93*10**-10


    zeropoints['2MASS/2MASS.J'] = 3.13*10**-10
    zeropoints['2MASS/2MASS.H'] = 1.13*10**-10
    zeropoints['2MASS/2MASS.Ks'] = 4.28*10**-11

    zeropoints['WISE/WISE.W1'] = 8.18*10**-12
    #zeropoints['WISE/WISE.W2'] = 2.42*10**-12
    #zeropoints['WISE/WISE.W3'] = 6.52*10**-14
    #zeropoints['WISE/WISE.W4'] = 5.09*10**-15

    #convert mags to Mags to Jy to erg/cm^2/s/A
    def mag_flux(mag,Fzp):
        #Mag = mag - 5*np.log10(dist)+5
        Flux = Fzp * 10**(-mag/2.5)  #units erg/cm^2/s/A
        return Flux

    #sort out values
    fltr_dict = {}
    eff_waves = pd.read_csv("eff_waves.csv")
    for key in mag_dict:
        if np.ma.is_masked(mag_dict[key]['mag'])==False: #want to take out entries with no values
            mag = float(mag_dict[key]['mag'])
            #want to weed out bad points
            if mag > -10:
                Fzp = zeropoints[key]
                flux = mag_flux(mag,Fzp)
                mags_err = (float(mag_dict[key]['sig']))
                index = list(eff_waves['filtername']).index(key)
                wave = eff_waves['eff_wave'][index]
                #quality check column for upper limits
                if key in quality_check:
                    ph_qual = quality_check[key]
                else:
                    ph_qual = 'A'
                
                fltr_dict[key] = {"flux":flux, 'flux_err':mags_err, 'eff_wave':wave, 'ph_qual':ph_qual}

    return fltr_dict

############################################################################

#Prerequisites:

#load dustmaps - this step takes ~ 2 min
bayestar = BayestarQuery(version='bayestar2019')

R_kpc_conversion = 5.08327 * 10**-22 #(kpc/R)**2

fltr_id = ['GALEX:FUV', "GALEX:NUV", "TYCHO:B", "TYCHO:V", "Hipparcos/Hipparcos.Hp", "PAN-STARRS/PS1:g", "PAN-STARRS/PS1:r", "PAN-STARRS/PS1:i", 'PAN-STARRS/PS1:z', 'PAN-STARRS/PS1:y', 'SkyMapper/SkyMapper:u', 'SkyMapper/SkyMapper:v', 'SkyMapper/SkyMapper:g', 'SkyMapper/SkyMapper:r', 'SkyMapper/SkyMapper:i', 'SkyMapper/SkyMapper:z', 'GAIA/GAIA3:G', "GAIA/GAIA3:Gbp", "GAIA/GAIA3:Grp", 'SDSS:u', 'SDSS:g', 'SDSS:r', 'SDSS:i', 'SDSS:z', 'CTIO/DECam:u', 'CTIO/DECam:g', 'CTIO/DECam:r', 'CTIO/DECam:i', 'CTIO/DECam:z', 'CTIO/DECam:Y', 'UKIDSS:Z', 'UKIDSS:Y', 'UKIDSS:J', 'UKIDSS:H', 'UKIDSS:K', 'VISTA:Z', 'VISTA:Y', 'VISTA:J', 'VISTA:H', 'VISTA:Ks', '2MASS:H', "2MASS:J", "2MASS:Ks", 'WISE:W1']
og_filters = ["GALEX/GALEX.FUV","GALEX/GALEX.NUV","TYCHO/TYCHO.B","TYCHO/TYCHO.V","Hipparcos/Hipparcos.Hp","PAN-STARRS/PS1.g","PAN-STARRS/PS1.r","PAN-STARRS/PS1.i","PAN-STARRS/PS1.z","PAN-STARRS/PS1.y","SkyMapper/SkyMapper.u","SkyMapper/SkyMapper.v","SkyMapper/SkyMapper.g","SkyMapper/SkyMapper.r","SkyMapper/SkyMapper.i","SkyMapper/SkyMapper.z","GAIA/GAIA0.G","GAIA/GAIA0.Gbp","GAIA/GAIA0.Grp","SLOAN/SDSS.u","SLOAN/SDSS.g","SLOAN/SDSS.r","SLOAN/SDSS.i","SLOAN/SDSS.z","CTIO/DECam.u","CTIO/DECam.g","CTIO/DECam.r","CTIO/DECam.i","CTIO/DECam.z","CTIO/DECam.Y","UKIRT/UKIDSS.Z","UKIRT/UKIDSS.Y","UKIRT/UKIDSS.J","UKIRT/UKIDSS.H","UKIRT/UKIDSS.K","Paranal/VISTA.Z","Paranal/VISTA.Y","Paranal/VISTA.J","Paranal/VISTA.H","Paranal/VISTA.Ks","2MASS/2MASS.H","2MASS/2MASS.J","2MASS/2MASS.Ks","WISE/WISE.W1"]

eff_waves = pd.read_csv("eff_waves.csv")

############################################################################

#INPUT

#getting source info:
#input either coords (gal_coord / ICRS_coord) or source name
info = ("HD 140283")
method = "source_name"

#Components:
num_comps = 1               #number of components (1 or 2)
lib1 = "Kurucz2003"           #spectral library for first component (hotter)
lib2 = "koester2"           #spectral library for second component (cooler)

############################################################################

#info
data = get_source(info, method)
main_id = data[0]

l = float(data[1])  #galactic coord in degrees
b = float(data[2])  #galactic coord in degrees
ra = float(data[5]) #ICRS coord in degrees
dec = float(data[6])#ICRS coord in degrees
plx = data[3]       #mas
plx_err = data[4]   #mas

#dist
d_mu = (1/plx)        #kpc

d_minus = 1/(plx+plx_err)
d_plus  = 1/(plx-plx_err)

d_sigma = ( (d_plus - d_mu) + (d_mu - d_minus) ) / 2   #kpc

#dust
#Av_q = mean_Av(l,b,d_mu)
Rv_q = 3.1          #empirical value of Rv

#functs lists:
funct_list1 = SED_Functs(lib1)
funct_list2 = SED_Functs(lib2)

#recover SED
SED = get_SED(main_id)

#getting values in lists
waves = []
log_flux = []
yerr_reported = []
filts = []
for key in SED:
    waves.append(SED[key]['eff_wave'])
    log_flux.append(np.log10(SED[key]["flux"]))
    yerr_reported.append((SED[key]["flux_err"]))
    filts.append(key)

############################################################################

#Sampling:
samp_num = 10000    #number of samples for finding p(d,Av)
sample_num = 3000   #number of samples for emcee

if num_comps == 1:
    walkers = 128       #number of walkers for emceee
    param_num = 7       #number of parameters
    if lib1 == 'Kurucz2003':
        param_num = 8       #number of parameters 
if num_comps == 2:
    walkers = 256
    param_num = 10
    if lib1 == 'Kurucz2003':
        param_num = 11
    if lib2 == 'Kurucz2003':
        param_num = 11
    if lib1 == 'Kurucz2003' and lib2 == 'Kurucz2003':
        param_num = 12
#############################################################################

#files for radius prior <-- move this section later
with open(f'WD_Rad_prior.pkl', 'rb') as f:
    WD_radius_prior = pickle.load(f)

with open('MS_Rad_prior.pkl', 'rb') as f:
    MS_radius_prior = pickle.load(f)

############################################################################

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
    meta1_l = -2.5
    meta1_u =  0.5

if lib2 == 'bt-settl-cifist':
    teff2_l = np.log10(1200)
    teff2_u = np.log10(7000)
    logg2_l = 2.5
    logg2_u = 5.5
    log_R2_l = -1.5
    log_R2_u = -0.5
if lib2 == 'koester2':
    teff2_l = np.log10(5000)
    teff2_u = np.log10(20000)
    logg2_l = 6.5
    logg2_u = 9.5
    log_R2_l = -2.09794001
    log_R2_u = -1.69897
if lib2 == 'Kurucz2003':
    teff2_l = np.log10(3500) #YOUR RADIUS PRIOR IS FUCKED
    teff2_u = np.log10(12500)
    logg2_l = 0
    logg2_u = 5
    log_R2_l = -1
    log_R2_u = 1.2
    meta2_l = -2.5
    meta2_u  = 0.5

############################################################################

#position and labels based on component count and library selection:
if num_comps == 1:
    labels = ['log_teff1', 'logg1', 'log_R1', 'log_dist', 'Av', 'Rv', 'log10_sigma_theory']     #parameter labels
    pos = [random.uniform(teff1_l, teff1_u), random.uniform(logg1_l, logg1_u), random.uniform(log_R1_l, log_R1_u), np.log10(d_mu), 0.9, 3.1, -1.5 ]

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
    pos = [teff1_guess, random.uniform(logg1_l, logg1_u), random.uniform(log_R1_l, log_R1_u), teff2_guess, random.uniform(logg2_l, logg2_u), random.uniform(log_R2_l, log_R2_u), np.log10(d_mu), 0.9, 3.1, -1.5 ]

#additional labels and pos if MS models are used:
if param_num == 8:
    labels = ['log_teff1', 'logg1', 'log_R1', 'meta1', 'log_dist', 'Av', 'Rv', 'log10_sigma_theory']
    pos = [random.uniform(teff1_l, teff1_u), random.uniform(logg1_l, logg1_u), random.uniform(log_R1_l, log_R1_u), random.uniform(meta1_l, meta1_u), np.log10(d_mu), 0.9, 3.1, -1.5 ]
if param_num == 11:
    if lib1 == 'Kurucz2003':
        labels = ['log_teff1', 'logg1', 'log_R1', 'meta1','log_teff2', 'logg2', 'log_R2', 'log_dist', 'Av', 'Rv', 'log10_sigma_theory']
        pos = [teff1_guess, random.uniform(logg1_l, logg1_u), random.uniform(log_R1_l, log_R1_u), random.uniform(meta1_l, meta1_u), teff2_guess, random.uniform(logg2_l, logg2_u), random.uniform(log_R2_l, log_R2_u), np.log10(d_mu), 0.9, 3.1, -1.5 ]
    if lib2 == 'Kurucz2003':
        labels = ['log_teff1', 'logg1', 'log_R1', 'log_teff2', 'logg2', 'log_R2','meta2', 'log_dist', 'Av', 'Rv', 'log10_sigma_theory']
        pos = [teff1_guess, random.uniform(logg1_l, logg1_u), random.uniform(log_R1_l, log_R1_u), teff2_guess, random.uniform(logg2_l, logg2_u), random.uniform(log_R2_l, log_R2_u), random.uniform(meta2_l, meta2_u), np.log10(d_mu), 0.9, 3.1, -1.5 ]
if param_num == 12:
    labels = ['log_teff1', 'logg1', 'log_R1', 'meta1','log_teff2', 'logg2', 'log_R2','meta2', 'log_dist', 'Av', 'Rv', 'log10_sigma_theory']
    pos = [teff1_guess, random.uniform(logg1_l, logg1_u), random.uniform(log_R1_l, log_R1_u), random.uniform(meta1_l, meta1_u), teff2_guess, random.uniform(logg2_l, logg2_u), random.uniform(log_R2_l, log_R2_u), random.uniform(meta2_l, meta2_u), np.log10(d_mu), 0.9, 3.1, -1.5 ]

############################################################################
#build dust-distance prior

#get samples of distancee & Av
d = samples_d(d_mu, d_sigma, samp_num)
Av = samples_Av(l, b, d)
Av_mu = np.mean(Av)

diff_sq_list = []
for i in Av:
    diff = i - Av_mu
    diff_sq = diff**2
    diff_sq_list.append(diff_sq)

sum_diff = sum(diff_sq_list)

Av_sigma_sq = 1/(len(Av)-1) * sum_diff 

#update this!!
if dec < -30:
    Av_mu = 0.1
    Av_sigma_sq = 0.001

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
        rad = np.clip(MS_radius_prior([log_R1,log_teff1]), 0.000001, None)
        return np.log10(rad)
if lib2 == "Kurucz2003":
    def rad2_prior(log_R2,log_teff2):
        #return (np.log(1.0/(np.sqrt(2*np.pi)*0.02))-0.5*((10**log_R2)-1)**2/0.02**2)#PLACEHOLDER WHILE YOU WORK OUT WHAT"S WRONG W MS PRIOR RANGES
       rad = np.clip(MS_radius_prior([log_R2,log_teff2]),0.000001, None)
       return np.log10(rad)


############################################################################

#feed into emcee

args = np.array(log_flux), yerr_reported #global variable


#functions

def log_likelihood_1(theta):

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
        #dict of [filter_id] : eff_wavelengths, filter_flux_values
        flx_dict1 = get_spec_funct(funct_list1, 10**(log_teff1), logg1, Av_clipped, Rv) #note here we're specifying the spectral library ourselves

    R1_sq = ((10**log_R1)**2)
    dist_sq = ((10**log_dist)**2)

    #log value of the flux
    model = {}
    for key in filts:
        y_model = (flx_dict1[key]['filt_flux'] + np.log10(R_kpc_conversion * R1_sq / dist_sq))#edit for logform
        wavelength = (flx_dict1[key]['eff_wave'])
        model[key] = {'y_model':y_model, 'wavelength':wavelength}

    #check for kurucz uneven grid, parameters that fall within range
    #heather what does this line do?? i think this might be wrong??
    if math.isnan(model[filts[0]]['y_model'])==True:#might need to change this up sinc you'v gone from a list to a dict for the model to handle upper lims easier
        return -np.inf

    #model = np.array(model)
    
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


def log_prior_1(theta):
    
    if lib1 == 'Kurucz2003':
        log_teff1, logg1, log_R1,meta1, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

        #parameter ranges
        if not (teff1_l <= log_teff1 <= teff1_u and
                logg1_l <= logg1 <= logg1_u and
                log_R1_l<=log_R1<=log_R1_u and
                meta1_l<=meta1<=meta1_u and
                -3<= log_dist <= 1 and
                Av_unbounded <= 5 and
                2.5<=Rv<=5.5 and
                -2 < log10_sigma_theor < 0):
            return -np.inf


    else:
        log_teff1, logg1, log_R1, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

        #parameter ranges
        if not (teff1_l <= log_teff1 <= teff1_u and 
                logg1_l <= logg1 <= logg1_u and 
                log_R1_l<=log_R1<=log_R1_u and 
                -3<= log_dist <= 1 and 
                Av_unbounded <= 5 and 
                2.5<=Rv<=5.5 and 
                -2 < log10_sigma_theor < 0):
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
        return -np.inf, None
    return lp + log_likelihood_1(theta), log_likelihood_1(theta)


def log_likelihood_2(theta):

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
    model = []
    for key in filts:
        model.append(np.log10((( 10**(flx_dict1[key]['filt_flux']) * (R1_sq)) + (10**(flx_dict2[key]['filt_flux']) * (R2_sq)))  * (R_kpc_conversion/(dist_sq))))
    #check for kurucz uneven grid
    if math.isnan(model[0])==True:
        return -np.inf
    
    model = np.array(model)

    #deviations
    sigma_theory_dex = 10**log10_sigma_theor
    sigma_dex2 = []
    for i in np.arange(len(yerr_dex)):
        sigma_dex2.append((yerr_dex[i])**2 + sigma_theory_dex**2)

    return np.sum(-0.5 * ((log10_y - model) ** 2) / sigma_dex2 - 0.5*np.log(sigma_dex2))


def log_prior_2(theta):
    
    if lib1 == 'Kurucz2003':
        log_teff1, logg1, log_R1, meta1, log_teff2, logg2, log_R2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

        #parameter ranges
        if not (teff1_l <= log_teff1 <= teff1_u and
                logg1_l <= logg1 <= logg1_u and
                log_R1_l<=log_R1<=log_R1_u and
                meta1_l<=meta1<=meta1_u and
                teff2_l <= log_teff2 <= teff2_u and
                log_teff2 < log_teff1 and
                logg2_l <= logg2 <= logg2_u and
                log_R2_l<=log_R2<=log_R2_u and
                -3<= log_dist <= 1 and
                Av_unbounded <= 5 and
                2.5<=Rv<=5.5 and
                -2 < log10_sigma_theor < 0):
            return -np.inf
    if lib2  == 'Kurucz2003':
        log_teff1, logg1, log_R1, log_teff2, logg2, log_R2, meta2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta
        
        #parameter ranges
        if not (teff1_l <= log_teff1 <= teff1_u and
                logg1_l <= logg1 <= logg1_u and
                log_R1_l<=log_R1<=log_R1_u and
                teff2_l <= log_teff2 <= teff2_u and
                log_teff2 < log_teff1 and
                logg2_l <= logg2 <= logg2_u and
                log_R2_l<=log_R2<=log_R2_u and
                meta2_l<=meta2<=meta2_u and
                -3<= log_dist <= 1 and
                Av_unbounded <= 5 and
                2.5<=Rv<=5.5 and
                -2 < log10_sigma_theor < 0):
            return -np.inf

    if param_num == 12:
        log_teff1, logg1, log_R1, meta1, log_teff2, logg2, log_R2, meta2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta

        #parameter ranges
        if not (teff1_l <= log_teff1 <= teff1_u and
                logg1_l <= logg1 <= logg1_u and
                log_R1_l<=log_R1<=log_R1_u and
                meta1_l<=meta1<=meta1_u and
                teff2_l <= log_teff2 <= teff2_u and
                log_teff2 < log_teff1 and
                logg2_l <= logg2 <= logg2_u and
                log_R2_l<=log_R2<=log_R2_u and
                meta2_l<=meta2<=meta2_u and
                -3<= log_dist <= 1 and
                Av_unbounded <= 5 and
                2.5<=Rv<=5.5 and
                -2 < log10_sigma_theor < 0):
            return -np.inf
    if param_num == 10:
        log_teff1, logg1, log_R1, log_teff2, logg2, log_R2, log_dist, Av_unbounded, Rv, log10_sigma_theor = theta
    
        #parameter ranges
        if not (teff1_l <= log_teff1 <= teff1_u and 
                logg1_l <= logg1 <= logg1_u and 
                log_R1_l<=log_R1<=log_R1_u and 
                teff2_l <= log_teff2 <= teff2_u and 
                log_teff2 < log_teff1 and 
                logg2_l <= logg2 <= logg2_u and 
                log_R2_l<=log_R2<=log_R2_u and 
                -3<= log_dist <= 1 and 
                Av_unbounded <= 5.0 and 
                2.5<=Rv<=5.5 and 
                -2 < log10_sigma_theor < 0):
            return -np.inf
    
    #param priors
    def sigma_theor_log_prior(log10_sigma_theor):
        return (-(np.log(10))*log10_sigma_theor)

    def Rv_log_prior(Rv):
        sigma_rv = 0.18
        return (np.log(1.0/(np.sqrt(2*np.pi)*sigma_rv))-0.5*(Rv-Rv_q)**2/sigma_rv**2)

    Jacobian_log = np.log( (10**log_dist) * (3.1 / Rv) )

    return sigma_theor_log_prior(log10_sigma_theor) + Rv_log_prior(Rv) + Av_d_log_prior(Av_unbounded, (10**log_dist)) + Jacobian_log +rad1_prior(log_R1,log_teff1) + rad2_prior(log_R2,log_teff2)


def log_probability_2(theta):
    lp = log_prior_2(theta)
    if not np.isfinite(lp):
        return -np.inf, None
    #if not np.isfinite(log_likelihood_2(theta)):
        #return -np.inf, None
    return lp + log_likelihood_2(theta), log_likelihood_2(theta)

############################################################################

#emcee

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

#output plots

#thin sample and get rid of samples from burn-in phase
samples = emcee_results.get_chain(discard=500, thin=100, flat=True)
prior_samples = prior_emcee.get_chain(discard=1000, thin=35, flat=True)

#corner plot
fig = corner.corner(
    samples, quantiles=[0.16, 0.50, 0.84], show_titles = True, labels=labels
);

corner.corner(
    prior_samples, fig=fig, plot_datapoints = False, plot_density = False, plot_contours = False, color = 'lightskyblue'
);

plt.savefig('emcee_output_corner.png')

#time series
fig, axes = plt.subplots(param_num, figsize=(10, 7), sharex=True)
samps = emcee_results.get_chain()
labels = labels
for i in range(0,param_num): #you don't have to plot all the samples you could change the range here
    ax = axes[i]
    ax.plot(samps[:, :, i], "k", alpha=0.05)
    ax.set_xlim(0, len(samps))
    ax.set_ylabel(labels[i])

axes[-1].set_xlabel("step number");

plt.savefig('emcee_output_series.png')


#output F vs λ plot

#collect random sample of parameters and find flux values for each set
numbers = range(len(samples))
random_sample = random.sample(numbers, 200)
flux_100 = []
#individual components:
flux1_100 = []
flux2_100 = []
for i in random_sample:

    if param_num == 7:
        flx_dict1 = get_spec_funct(funct_list1, 10**samples[i][0], samples[i][1], np.clip(samples[i][4], 0, None), samples[i][5])
        flux = {}
        for key in flx_dict1:
            flux[key] = (flx_dict1[key]['filt_flux'] + np.log10(R_kpc_conversion) + 2*samples[i][2] - 2*samples[i][3])
    
    if param_num == 8:
        flx_dict1 = get_spec_funct(funct_list1, 10**samples[i][0], samples[i][1], np.clip(samples[i][5], 0, None), samples[i][6],samples[i][3])
        flux = {}
        for key in flx_dict1:
            flux[key] = (flx_dict1[key]['filt_flux'] + np.log10(R_kpc_conversion) + 2*samples[i][2] - 2*samples[i][4])

    if param_num == 10:
        flx_dict1 = get_spec_funct(funct_list1, 10**samples[i][0], samples[i][1], np.clip(samples[i][7], 0, None), samples[i][8])
        flx_dict2 = get_spec_funct(funct_list2, 10**samples[i][3], samples[i][4], np.clip(samples[i][7], 0, None), samples[i][8])
        flux = {}
        flux1 = {}
        flux2 = {}
        for key in flx_dict1:
            flux[key] = np.log10( ((10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2) + (10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][5])**2))) * (R_kpc_conversion) / (10**samples[i][6])**2)
            flux1[key] = np.log10(10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2 * (R_kpc_conversion) / (10**samples[i][6])**2)
            flux2[key] = np.log10(10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][5])**2)  * (R_kpc_conversion) / (10**samples[i][6])**2)
    
    if param_num == 11:
        if lib1 == "Kurucz2003":
            flx_dict1 = get_spec_funct(funct_list1, 10**samples[i][0], samples[i][1], np.clip(samples[i][8], 0, None), samples[i][9], samples[i][3])
            flx_dict2 = get_spec_funct(funct_list2, 10**samples[i][4], samples[i][5], np.clip(samples[i][8], 0, None), samples[i][9])
            flux = {}
            flux1 = {}
            flux2 = {}
            for key in flx_dict1:
                flux[key] = np.log10( ((10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2) + (10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][6])**2))) * (R_kpc_conversion) / (10**samples[i][7])**2)
                flux1[key] = np.log10(10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2 * (R_kpc_conversion) / (10**samples[i][7])**2)
                flux2[key] = np.log10(10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][6])**2)  * (R_kpc_conversion) / (10**samples[i][7])**2)
        elif lib2 == "Kurucz2003":
            flx_dict1 = get_spec_funct(funct_list1, 10**samples[i][0], samples[i][1], np.clip(samples[i][8], 0, None), samples[i][9])
            flx_dict2 = get_spec_funct(funct_list2, 10**samples[i][3], samples[i][4], np.clip(samples[i][8], 0, None), samples[i][9], samples[i][6])
            flux = {}
            flux1 = {}
            flux2 = {}
            for key in flx_dict1:
                flux[key] = np.log10( ((10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2) + (10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][5])**2))) * (R_kpc_conversion) / (10**samples[i][7])**2)
                flux1[key] = np.log10(10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2 * (R_kpc_conversion) / (10**samples[i][7])**2)
                flux2[key] = np.log10(10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][5])**2)  * (R_kpc_conversion) / (10**samples[i][7])**2)

    if param_num == 12:
        flx_dict1 = get_spec_funct(funct_list1, 10**samples[i][0], samples[i][1], np.clip(samples[i][9], 0, None), samples[i][10],samples[i][3])
        flx_dict2 = get_spec_funct(funct_list2, 10**samples[i][4], samples[i][5], np.clip(samples[i][9], 0, None), samples[i][10],samples[i][7])
        flux = {}
        flux1 = {}
        flux2 = {}
        for key in flx_dict1:
            flux[key] = np.log10( ((10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2) + (10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][6])**2))) * (R_kpc_conversion) / (10**samples[i][8])**2)
            flux1[key] = np.log10(10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2 * (R_kpc_conversion) / (10**samples[i][8])**2)
            flux2[key] = np.log10(10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][6])**2)  * (R_kpc_conversion) / (10**samples[i][8])**2)

    if param_num > 9:
        flux_100.append(flux)
        flux1_100.append(flux1)
        flux2_100.append(flux2)
    else:
        flux_100.append(flux)

#take the average value of flux at each filter:
avg_flux_values = []
lower_ferr = []
upper_ferr = []
for key in flx_dict1:
    flux_values = []
    for flux in flux_100:
        flux_values.append(flux[key])

    per = np.percentile(flux_values, [16, 50, 84])
    lower_ferr.append(per[0])
    avg_flux_values.append(per[1])
    upper_ferr.append(per[2])

#residuals:
residuals = np.array(log_flux) - np.array(avg_flux_values)

#errorbar for residuals:
l= np.array(log_flux) - np.array(lower_ferr)
u= np.array(log_flux) - np.array(upper_ferr)
res_err = (residuals - u, l-residuals)

#plot original with sample average:
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9,9), gridspec_kw={'height_ratios': [2, 1]})

if param_num > 9:
    for sed in flux1_100:
        forest_flux = []
        forest_wave = []
        forest_wave_old = []
        for key in flx_dict1:
            forest_flux.append(sed[key])
            forest_wave.append(SED[key]['eff_wave'])
            forest_wave_old.append(SED[key]['eff_wave'])
        forest_wave.sort()
        forest_flux_new = []
        for i in forest_wave:
            index = forest_wave_old.index(i)
            forest_flux_new.append(forest_flux[index])
        ax1.plot(forest_wave, forest_flux_new, c='salmon', alpha=0.05)

    for sed in flux2_100:
        forest_flux = []
        forest_wave = []
        forest_wave_old = []
        for key in flx_dict1:
            forest_flux.append(sed[key])
            forest_wave.append(SED[key]['eff_wave'])
            forest_wave_old.append(SED[key]['eff_wave'])
        forest_wave.sort()
        forest_flux_new = []
        for i in forest_wave:
            index = forest_wave_old.index(i)
            forest_flux_new.append(forest_flux[index])
        ax1.plot(forest_wave, forest_flux_new, c='lightskyblue', alpha=0.05)
else:
    for sed in flux_100:
        forest_flux = []
        forest_wave = []
        forest_wave_old = []
        for key in flx_dict1:
            forest_flux.append(sed[key])
            forest_wave.append(SED[key]['eff_wave'])
            forest_wave_old.append(SED[key]['eff_wave'])
        forest_wave.sort()
        forest_flux_new = []
        for i in forest_wave:
            index = forest_wave_old.index(i)
            forest_flux_new.append(forest_flux[index])
        ax1.plot(forest_wave, forest_flux_new, c='m', alpha=0.05)

ax1.errorbar(waves,np.array(log_flux), yerr=yerr_reported, fmt='.k', capsize=4,label = 'Data')
upper_lims = []
waves_ul = []
for key in SED:
    if SED[key]['ph_qual'] =='U':
        waves_ul.append(SED[key]['eff_wave'])
        upper_lims.append(np.log10(SED[key]['flux']))
ax1.scatter(waves_ul,np.array(upper_lims), marker='v', color = 'gold', label = 'Upper Limit')
ax1.scatter(waves,avg_flux_values, label = 'Posterior Predictive Mean', c='blueviolet')
ax1.set_ylabel("Log Mean Spectral Flux Density ($erg/cm^2/s/\AA$)")
#ax1.set_ylim(-17.5,-14) #automate this!!
ax1.legend()

ax2.errorbar(waves, residuals, yerr= res_err, fmt='.c', capsize=4)
ax2.set_ylabel("Residuals dex")
ax2.set_xlabel("Wavelengths ($\AA$)")

ax2.axhline(y=0, linestyle='dotted', color='k')

plt.xscale('log')

plt.savefig("output_system.png")


# Model Comparision CHECK PLEASE
log_like = emcee_results.get_blobs(flat=True)

#sort out nan values from ll samples
non_none_ll = []
for i in log_like:
    if np.isnan(i) == False:
        non_none_ll.append(i)
#compute lppd
ll_mu = np.sum(non_none_ll) / len(non_none_ll)

var_all = []
for i in non_none_ll:
    diff_sq = (i-ll_mu)**2
    var_all.append(diff_sq)
#compute pwaic
var = np.sum(var_all) / len(non_none_ll)

WAIC = -2*ll_mu + 2*var

print("WAIC: ", WAIC)
