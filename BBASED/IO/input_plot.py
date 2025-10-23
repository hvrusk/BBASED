"""Plots based on the input data"""


def input_plot(SED, info=None):
    """ Takes the SED - synthesized or observed data and corresponding wavelengths
        and plots that data on a F_λ vs λ plot
        Optionally takes info to label the plot"""

    waves = []
    log_flux = []
    yerr_reported = []
    filts = []
    for key in SED:
        waves.append(SED[key]['eff_wave'])
        log_flux.append(np.log10(SED[key]["flux"]))
        yerr_reported.append((SED[key]["flux_err"]))
        filts.append(key)

    #input plot - data
    fig, ax = plt.subplots(figsize=(10,8))
    ax.errorbar(waves, log_flux, yerr=yerr_reported, fmt = '.k', label="data")

    #labels:
    ax.set_xlabel("Wavelengths (λ)")
    ax.set_ylabel("Log Mean Spectral Flux (erg/$cm^2$/s/λ)")
    plt.xscale("log")

    #plot label:
    if info:
        name = info.replace(" ", "_") #plots will be labeled with the name/coords of the source
    else:
        name = None
    plt.savefig(f"input_plot_{name}.png")

    return
