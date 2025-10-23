""" plots based on the results from BBASED
    they all take the samples from BBASED"""

def corner_plot(results, prior_output, labels, truths=None):
    """ produces a corner plot displaying the posterior distributions
        for each of the parameters in your model - inputs are the BBASED samples,
        prior samples, and labels. Truths optional if you're plotting a synthetic
        system, or wish to include known parameter values, should be formatted in a list"""

    #thin sample and get rid of samples from burn-in phase
    samples = results.get_chain(discard=500, thin=100, flat=True)
    prior_samples = prior_output.get_chain(discard=1000, thin=35, flat=True)

    #corner plot
    fig = corner.corner(
        samples, quantiles=[0.16, 0.50, 0.84], show_titles = True, labels=labels
    );

    corner.corner(
        prior_samples, fig=fig, plot_datapoints = False, plot_density = False, plot_contours = False, color = 'lightskyblue'
    );

    plt.savefig('emcee_output_corner.png')

    return


def time_series_plot(results, labels)
    """ produces a plot of the time series, or the chains throughout the samples you've taken for each
        parameter in your model. Useful for gaining some insight into what the walkers are doing if
        you're having issues with your runs"""
    
    fig, axes = plt.subplots(param_num, figsize=(10, 7), sharex=True)
    samps = results.get_chain()
    labels = labels
    for i in range(0,param_num): #you don't have to plot all the samples you could change the range here
        ax = axes[i]
        ax.plot(samps[:, :, i], "k", alpha=0.05)
        ax.set_xlim(0, len(samps))
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("step number");

    plt.savefig('emcee_output_series.png')

    return


def param_values(results, labels):
    """ Function that takes the results from BBASED and outputs the avg parameter values from the posterior distribution,
    the uncertainties quoted for each of these values is the 2 sigma value
    requires the results and labels from your model"""

    #thin out samples and discard the burn-in phase
    samples = results.get_chain(discard=500, thin=100, flat=True)

    #number of dimensions for the samples
    ndim = samples.shape[1]
    for i in range(ndim):
        percentiles = np.percentile(samples[:, i], [2.5, 50, 97.5]
        q = np.diff(percentiles)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(percentiles[1], q[0], q[1], labels[i])
        print("Parameter posterior predictive means: ") #check if this is correct phrasing
        display(Math(txt))

    return


def model_comparison(results):
    """ function that takes the results from BBASED and computes the WAIC value,
    a lower value of the WAIC value indicates a higher prob for that model pairing """

    log_like = results.get_blobs(flat=True)

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

    return WAIC


def flux_wavelength_plot(results, labels):
    """ This function plots your data, with the mean log flux values from your results,
        and the posterior predictive SEDs onto a F_λ vs λ plot
    """

    #first we take a random set of samples from your results
    numbers = range(len(samples))
    random_sample = random.sample(numbers, 200)
    #combined components for the posterior predictive mean:
    flux_100 = []
    #individual components:
    flux1_100 = []
    flux2_100 = []

    for i in random_sample:
        #oh god this is so inefficient -  you're gonna recheck if met is a parameter for each sample??! it'll
        #be the same each  time! UPDATE
        if "meta1" in labels:
            if "meta2" in labels: #case where we have a binary where both models contain metallicity
                flx_dict1 = get_spec_funct(funct_list1, 10**samples[i][0], samples[i][1], np.clip(samples[i][9], 0, None), samples[i][10],samples[i][3])
                flx_dict2 = get_spec_funct(funct_list2, 10**samples[i][4], samples[i][5], np.clip(samples[i][9], 0, None), samples[i][10],samples[i][7])

                flux = {}
                flux1 = {}
                flux2 = {}

                for key in flx_dict1:
                    flux[key] = np.log10( ((10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2) + (10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][6])**2))) * (R_kpc_conversion) / (10**samples[i][8])**2)
                    flux1[key] = np.log10(10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2 * (R_kpc_conversion) / (10**samples[i][8])**2)
                    flux2[key] = np.log10(10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][6])**2)  * (R_kpc_conversion) / (10**samples[i][8])**2)

            elif "logg2" in labels: #case wheere we have a binary, but only the first modl has met as a parameter
                flx_dict1 = get_spec_funct(funct_list1, 10**samples[i][0], samples[i][1], np.clip(samples[i][8], 0, None), samples[i][9],samples[i][3])
                flx_dict2 = get_spec_funct(funct_list2, 10**samples[i][4], samples[i][5], np.clip(samples[i][8], 0, None), samples[i][9])

                flux = {}
                flux1 = {}
                flux2 = {}

                for key in flx_dict1:
                    flux[key] = np.log10( ((10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2) + (10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][6])**2))) * (R_kpc_conversion) / (10**samples[i][7])**2)
                    flux1[key] = np.log10(10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2 * (R_kpc_conversion) / (10**samples[i][7])**2)
                    flux2[key] = np.log10(10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][6])**2)  * (R_kpc_conversion) / (10**samples[i][7])**2)

            else: #case where we have a single star with met as a parameter
                flx_dict1 = get_spec_funct(funct_list1, 10**samples[i][0], samples[i][1], np.clip(samples[i][5], 0, None), samples[i][6],samples[i][3])
                flux = {}
                for key in flx_dict1:
                    flux[key] = (flx_dict1[key]['filt_flux'] + np.log10(R_kpc_conversion) + 2*samples[i][2] - 2*samples[i][4])

        elif "meta2" in labels: #case where we have a binary with no met parameter in the first component, but met in the second component
            flx_dict1 = get_spec_funct(funct_list1, 10**samples[i][0], samples[i][1], np.clip(samples[i][8], 0, None), samples[i][9])
            flx_dict2 = get_spec_funct(funct_list2, 10**samples[i][3], samples[i][4], np.clip(samples[i][8], 0, None), samples[i][9], samples[i][6])
            flux = {}
            flux1 = {}
            flux2 = {}
            for key in flx_dict1:
                flux[key] = np.log10( ((10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2) + (10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][5])**2))) * (R_kpc_conversion) / (10**samples[i][7])**2)
                flux1[key] = np.log10(10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2 * (R_kpc_conversion) / (10**samples[i][7])**2)
                flux2[key] = np.log10(10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][5])**2)  * (R_kpc_conversion) / (10**samples[i][7])**2)

        elif "logg2" in labels: #case where we have a binary with no additional parameters
            flx_dict1 = get_spec_funct(funct_list1, 10**samples[i][0], samples[i][1], np.clip(samples[i][7], 0, None), samples[i][8])
            flx_dict2 = get_spec_funct(funct_list2, 10**samples[i][3], samples[i][4], np.clip(samples[i][7], 0, None), samples[i][8])
            flux = {}
            flux1 = {}
            flux2 = {}
            for key in flx_dict1:
                flux[key] = np.log10( ((10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2) + (10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][5])**2))) * (R_kpc_conversion) / (10**samples[i][6])**2)
                flux1[key] = np.log10(10**flx_dict1[key]['filt_flux'] * (10**samples[i][2])**2 * (R_kpc_conversion) / (10**samples[i][6])**2)
                flux2[key] = np.log10(10**(flx_dict2[key]['filt_flux']) * ((10**samples[i][5])**2)  * (R_kpc_conversion) / (10**samples[i][6])**2)
        
        else: #single star, no met case
            flx_dict1 = get_spec_funct(funct_list1, 10**samples[i][0], samples[i][1], np.clip(samples[i][4], 0, None), samples[i][5])
            flux = {}
            for key in flx_dict1:
                flux[key] = (flx_dict1[key]['filt_flux'] + np.log10(R_kpc_conversion) + 2*samples[i][2] - 2*samples[i][3])
        
        if len(labels) > 9:
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


    if len(labels) > 9:
        #for the posterior predicitive SEDs of Comp1
        for sed in flux1_100:
            forest_flux = []#this one is for F_λ
            forest_F = [] #this one is for λF_λ
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
                forest_F.append((10**forest_flux[index] * i))
            ax1.plot(forest_wave, forest_flux_new, c='salmon', alpha=0.05)
            #ax1.plot(forest_wave, forest_F, c='salmon', alpha=0.05)

        #for the posterior predicitive SEDs of Comp2
        for sed in flux2_100:
            forest_flux = [] #this one is for F_λ
            forest_F = [] #this one is for λF_λ
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
                forest_F.append((10**forest_flux[index] * i))
            ax1.plot(forest_wave, forest_flux_new, c='lightskyblue', alpha=0.05)
            #ax1.plot(forest_wave, forest_F, c='lightskyblue', alpha=0.05)

    else:
        for sed in flux_100:
            forest_flux = [] #this one is for F_λ
            forest_F = [] #this one is for λF_λ
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
                forest_F.append(np.log10(10**forest_flux[index] * i))
            ax1.plot(forest_wave, forest_flux_new, c='m', alpha=0.05)
            #ax1.plot(forest_wave, forest_F, c='m', alpha=0.05)

    ax1.errorbar(waves,np.array(log_flux), yerr=yerr_reported, fmt='.k', capsize=4,label = 'Data')
    upper_lims = []
    waves_ul = []
    for key in SED:
        if SED[key]['ph_qual'] =='U':
            waves_ul.append(SED[key]['eff_wave'])
            upper_lims.append(np.log10(SED[key]['flux']))
    #Upper limits:
    #ax1.scatter(waves_ul,np.array(upper_lims), marker='v', color = 'gold', label = 'Upper Limit')

    #what's all this rubbish? thanks for asking! in case you don't like the fact that the plots are in
    #erg/cm^2/s/λ you can use these values to plot them in terms of erg/cm^2/s instead!
    ppm_flux = []
    data_flux = []
    for i in np.arange(len(avg_flux_values)):
        flux =((10**avg_flux_values[i] * waves[i]))
        ppm_flux.append(flux) #this one is for λF_λ for the posterior predictive mean
        data = ((10**(np.array(log_flux))[i] * waves[i]))
        data_flux.append(data) #this one is for λF_λ but for the original data

    #ax1.errorbar(waves,data_flux, yerr=yerr_reported, fmt='.k', capsize=4,label = 'Data')
    #ax1.scatter(waves,data_flux, c='k', label = 'Data')
    #ax1.scatter(waves,ppm_flux, label = 'Posterior Predictive Mean', c='blueviolet')
    ax1.scatter(waves,avg_flux_values, label = 'Posterior Predictive Mean', c='blueviolet')
    ax1.set_ylabel("Log Mean Spectral Flux Density ($erg/cm^2/s/\AA$)")
    #ax1.set_ylabel("Mean Spectral Flux ($erg/cm^2/s$)")
    #ax1.set_ylim(-17.5,-14) #automate this!!
    ax1.legend()

    ax2.errorbar(waves, residuals, yerr= res_err, fmt='.c', capsize=4)
    ax2.set_ylabel("Residuals dex")
    ax2.set_xlabel("Wavelengths ($\AA$)")

    ax2.axhline(y=0, linestyle='dotted', color='k')

    plt.xscale('log')

    plt.savefig("output_SED.png")

    return
