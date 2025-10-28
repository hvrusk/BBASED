""" Prior functions for running BBASED """

def samples_d(d_mu, d_sigma, samp_num):
    """ get samples of d in kpc from p(d)
        d_mu is the distance we get from 1/plx
        d_sigma is the uncertainty we infer from the plx_err
        samp_num is the number of samples we want to take from
        the normal distribution of distance"""
    dist_samples = d_sigma * np.random.randn(samp_num) + d_mu
    dist_samples = [d for d in dist_samples if d>0]
    return dist_samples


def samples_Av(l, b, d):
    """ get samples of Av from d
        we take the galactic coordinates (l & b in degrees)
        to take random samples of e(b-v) from the 3D dust maps
        from Green et al. 2019
        random samples are taken at the coordinates and random distance
        draws: d

        This function requires you to load up the Bayestar19 maps which
        take roughly a minute to do - note that BS19 doesn't have
        coverage beneath dec= -30 deg
    """
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

def build_dust_prior(d_mu, d_sigma, l, b):
    """ this function creates a sample of distance and dust values
        for the purpose of constructing the prior on dust
        returns the average value of Av and the variance"""

    #get samples of distance & Av
    samp_num = 10000 #this is the number of draws from the distance distribution we take
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

    return Av_mu, Av_sigma_sq

    
def Av_d_log_prior(Av_unbounded, dist):
    """ log prior used in the sampling procedure
        requires you to define the parameters of the 2d gaussian as global variables
        before running 
    """
    prior = np.log(1/(2*np.pi*d_sigma*(Av_sigma_sq**0.5))) - 0.5*( ((dist-d_mu)/d_sigma)**2 + ((Av_unbounded-Av_mu)**2)/Av_sigma_sq  )
    return prior
