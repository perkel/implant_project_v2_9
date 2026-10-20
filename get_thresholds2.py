# get_thresholds2.py -- based on get_thresholds.py (version of 25 October 2023)
# Original code by Steven Bierer
# Translated to python by David J. Perkel
#
# Same interface and return values as get_thresholds.py, but replaces the hand-rolled bisection
# loop with scipy.optimize.brentq. The original comment on that loop said "I couldn't find a
# solver in scipy.optimize that took advantage of the known monotonicity of the function" --
# brentq (and toms748) are exactly bracketed root-finders for that case: given a bracket where the
# function changes sign, they combine bisection with inverse-quadratic interpolation and converge
# superlinearly, so they typically need noticeably fewer function evaluations than plain bisection
# for a smooth, monotonic function like the sigmoid-based activation curve used here.
#
# Used by fwd_model5_2d.py; FwdModel4_2D.py and the inverse models continue to use the original
# get_thresholds.py so the two bisection strategies can be compared directly.

import numpy as np
from scipy.optimize import brentq
import thr_function as thr_f
import cylinder3d_makeprofile as c3dm


# The function that is used by the optimization algorithm
def objectivefunc(x, e_field, sim_params):
    target = sim_params['neurons']['thrtarg']  # threshold # neurons
    tv1 = thr_f.thr_function(x, e_field, sim_params)[0]  # returns nCount, NeuralProfile, ActiveProfile
    retval = (tv1 - target) ** 2
    return retval


# Constants
fit_precision = 0.01


def _neuron_count_error(current, e_field, sim_params, prop_const, target):
    """The function whose root (in 'current') is the threshold: predicted neuron count minus
    target neuron count. Same quantity as 'error' in the original bisection loop."""
    n_neur = thr_f.thr_function(current, e_field, sim_params, prop_const)[0]
    return n_neur - target


def get_thresholds(field_table: np.ndarray, field_params: dict, sim_params: dict) -> list:
    # function that is called by forward, forward_2D and inverse models
    # Preallocate the arrays needed
    nz = len(sim_params['grid']['z'])
    if isinstance(sim_params['electrodes']['rpos'], list):
        nelec = len(sim_params['electrodes']['rpos'])
    elif not isinstance(sim_params['electrodes']['rpos'], np.ndarray):  # special case for 2D version of fwd model
        nelec = 1
    else:  # regular situation when called from forward or inverse model
        nelec = len(sim_params['electrodes']['rpos'])
    thresholds = np.empty(nelec)  # Array to hold threshold data for different stim electrodes and varied sigma values
    thresholds[:] = np.nan
    e_field = np.empty(nz)
    e_field[:] = np.nan

    if nelec == 1:
        elec_vals = range(7, 8)  # use electrode 7 to be in middle of cochlear length
        # set the position manually in the center of the array
        espace = sim_params['electrodes']['zpos'][elec_vals[0]] - sim_params['electrodes']['zpos'][elec_vals[0] - 1]
        neuron_midpoint = np.max(sim_params['grid']['z']) / 2.0
        elec_pos = neuron_midpoint  # fixed position in center of neuron array
        sim_params['electrodes']['zpos'][elec_vals[0]] = elec_pos
        sim_params['electrodes']['zpos'][elec_vals[0] - 1] = elec_pos - espace
        sim_params['electrodes']['zpos'][elec_vals[0] + 1] = elec_pos + espace
    else:
        if sim_params['channel']['sigma'] == 0.0:  # monopolar
            elec_vals = range(0, nelec)
        else:
            elec_vals = range(1, nelec - 1)  # Can't do tripolar stimulation at end electrodes

    nvalarray = []
    nvals = 0

    # Precompute once: depends only on sim_params['neurons']['rlvl'], not on stim electrode
    # or current, but thr_function is called many times below (once per solver iteration,
    # for every electrode), so recomputing it inline each time was wasted work.
    prop_const = thr_f.compute_prop_const(sim_params)

    for j in elec_vals:  # Loop on stimulating electrodes
        sim_params['channel']['number'] = j
        aprofile = c3dm.cylinder3d_makeprofile(field_table, field_params, sim_params)
        # This is the biophysically correct behavior, to include sidelobes
        e_field = abs(aprofile)  # uV^2/mm^2

        if sim_params['channel']['behavior'] == 0:
            target = sim_params['neurons']['thrtarg']
        else:
            target = sim_params['neurons']['M_targ']

        lo = 0.0
        hi = sim_params['channel']['current']

        err_hi = _neuron_count_error(hi, e_field, sim_params, prop_const, target)
        if err_hi < -fit_precision:
            # Even the maximum available current can't reach the target neuron count --
            # matches the original bisection's early exit when nextstim == current on its
            # very first iteration (undershoot at the upper bound).
            nextstim = hi
        else:
            err_lo = _neuron_count_error(lo, e_field, sim_params, prop_const, target)
            if err_lo > fit_precision:
                # Target already exceeded at zero current. The original bisection loop had no
                # explicit handling for this (it shouldn't occur physically), but guard against
                # brentq's "f(a) and f(b) must have different signs" error just in case.
                nextstim = lo
            else:
                nextstim = brentq(_neuron_count_error, lo, hi,
                                  args=(e_field, sim_params, prop_const, target),
                                  xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100)

        # Neural profile at the converged current (used below for nelec==1; the nelec>1 case
        # already does its own explicit recompute a few lines down, same as the original).
        _, nvals, _aa = thr_f.thr_function(nextstim, e_field, sim_params, prop_const)

        if nelec == 1:
            thresholds[0] = nextstim
        else:
            thresholds[j] = nextstim

        # Here call thrFunction to get the neural profile at threshold
        if len(thresholds) > 1:
            profileval = thr_f.thr_function(thresholds[j], e_field, sim_params, prop_const)[1]
            nvalarray.append(profileval)

    thresholds = 20 * np.log10(thresholds)  # return thresholds in dB
    ncount = np.sum(np.asarray(nvals))

    #  Sanity check alarm to test if the fitting algorithm didn't converge
    if sim_params['channel']['behavior'] == 0:
        if np.abs(ncount - sim_params['neurons']['thrtarg']) > 10:
            print("getThresholds returning ncount: ", ncount, ' , with thrtarg == ', sim_params['neurons']['thrtarg'])
    else:
        if np.abs(ncount - sim_params['neurons']['M_targ']) > 10:
            print("getThresholds returning ncount: ", ncount, ' , with thrtarg == ', sim_params['neurons']['thrtarg'])

    if len(thresholds) == 1:
        return [thresholds, np.asarray(nvals)]
    else:
        return [thresholds, np.asarray(nvalarray)]
