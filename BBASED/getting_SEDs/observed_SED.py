""" observed_SED.py – Functions for building observed SED by querying Gaia
    and the cross matched catalogues using the provided source information.
"""
#for collection source data
from astropy.coordinates import SkyCoord #??
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from astropy.table import Table
import astropy.units as units


def get_plx(ra, dec):
    """ If source info is given as ra and dec this function will query Gaia
        using the ra and dec to find the plx, plx_err and gal. coord.
    """

    
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


def get_info_gal(l,b):
    """ This function takes the gal.coords. (l & b in degrees) to query Simbad
        to find the main id / name of the source you're querying. Returns that name
    """
    
    table = Simbad.query_region(SkyCoord(l, b, unit=(units.deg, units.deg),frame='galactic'), radius=2 * units.arcsec)
    main_id = table['MAIN_ID'][0]
    return main_id


def source_info(object_name):
    """ From the object name, finds the Gaia id name to query the Gaia Archive
        selects and returns the source_id, ra & dec, galactic coords, parallax
        & parallax_err 
    """
    
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

def get_source(info, method):
    """ Takes the user submitted info (ICRS, gal, source_name) and the info
        type (either "ICRS_coord", "gal_coord", or "source_name") as the method
        for how they've choosen to retrieve the source information
        returns: main_id, l, b, parallax, parallax_err, ra, dec
    """
    
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


def get_SED(main_id):
    """ This function takes your the main_id from get_source and gathers
        photometry from gaia and cross-matched catalogues, matched based on Gaia
        source_id. returns a dict to be used by the likelihood function that contains
        flux (erg/cm^2/s/λ), flux err, effective wavelength, and phot_quality
    """
    
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


