""" set_up.py - sets up the necessary libraries needed to create the models
    to fit to your data """

import pickle

def model_funct(lib):
    """ given a selected model this function will call and assign the respective
        interpolation function associated with that model to build model SEDs
    """
    if lib == 'bt-settl-cifist':
        with open('BBASED/run_BBASED/set_up/model_interp/lib_function_BTSC_CUBIC.pkl', 'rb') as f:
            functs = pickle.load(f)
    if lib == 'koester2':
        with open('BBASED/run_BBASED/set_up/model_interp/lib_function_K2_CUBIC.pkl', 'rb') as f:
            functs = pickle.load(f)
    if lib == 'Kurucz2003':
        with open('BBASED/run_BBASED/set_up/model_interp/lib_function_K2003_CUBIC.pkl', 'rb') as f:
            functs = pickle.load(f)
    return functs
