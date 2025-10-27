""" priors for radii - based on stellar types
    requires you to have already defined the stellar models as a global variable
    requires you to load the radius prior files for MS and WD stars"""

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
       rad = np.clip(MS_radius_prior([log_R2,log_teff2]),0.000001, None)
       return np.log10(rad)
