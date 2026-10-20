# calc_ecap.py -- calculates a proxy for electrical compound action potential
# David J. Perkel 26 November 2022

import numpy as np
from common_params import *
import get_thresholds as gT
import cylinder3d_makeprofile as c3dm
import thr_function


def db_to_cur(val):
    return np.power(10, val/20.0)


def cur_to_db(val):
    return np.log10(val)*20.0


def calc_ecap(act_vals, field_params, sim_params, thresh, thr_sim_db, n_stim):

    # Calculate ecap from threshold up to activation of half of all neurons. We have threshold.
    # Next get stimulus intensity to activate half of all neurons
    nz = len(sim_params['grid']['z'])
    if isinstance(sim_params['electrodes']['rpos'], list):
        nelec = len(sim_params['electrodes']['rpos'])
    elif not isinstance(sim_params['electrodes']['rpos'], np.ndarray):  # special case for 2D version of fwd model
        nelec = 1
    else:
        nelec = len(sim_params['electrodes']['rpos'])
    # half_act_thresh = np.zeros(nelec)  # Array to hold half_activation thresholds

    # low end should be 5dB below threshold
    # high end should be 1000 neurons (given by calling function)
    # Normal dynamic range is 20 dB
    orig_targ = sim_params['neurons']['thrtarg']  # save old value
    sim_params['neurons']['thrtarg'] = thresh
    high_thresh, nact = gT.get_thresholds(act_vals, field_params, sim_params)
    sim_params['neurons']['thrtarg'] = orig_targ  # restore old value

    ecap = np.zeros([nelec, n_stim, 2])

    # Precompute once: depends only on sim_params['neurons']['rlvl'], which doesn't change across
    # electrodes or stimulus currents below, but thr_function.thr_function() is called nelec * n_stim
    # times (n_stim can be in the thousands), so recomputing it inline each time was wasted work.
    prop_const = thr_function.compute_prop_const(sim_params)

    # Loop on electrodes
    for k in range(nelec):
        sim_params['channel']['number'] = k
        aprofile = c3dm.cylinder3d_makeprofile(act_vals, field_params, sim_params)
        e_field = abs(aprofile)  # uV^2/mm^2
        # currents = np.linspace(db_to_cur(thr_sim_db[k]), db_to_cur(high_thresh[k]), num=n_stim)
        # currents = np.linspace(db_to_cur(thr_sim_db[k] - 10), db_to_cur(80), num=n_stim)
        currents = np.linspace(db_to_cur(30), db_to_cur(60), num=n_stim)
        # Now we have threshold and activation profile. Next calculate ECAP by calling thrFunction
        # with a range of current arguments from the treshold to a maximum value
        # a, b = thr_function.thr_function(db_to_cur(22), e_field, sim_params)[0:2]
        for i, cur in enumerate(currents):
            ecap[k, i, 0] = cur_to_db(cur)
            ecap[k, i, 1] = thr_function.thr_function(cur, e_field, sim_params, prop_const)[0]

    return ecap
