""" Radii files - prior maps for MS and WD stars in Gaia to inform the radii
    parameters in the model - maybe as an update later you can make this
    into a function like with the model files"""

with open('rad_prior_files/WD_Rad_prior.pkl', 'rb') as f:
    WD_radius_prior = pickle.load(f)

with open('rad_prior_files/MS_Rad_prior.pkl', 'rb') as f:
    MS_radius_prior = pickle.load(f)

with open('rad_prior_files/MS_met_prior.pkl', 'rb') as f:
    met_prior = pickle.load(f)
