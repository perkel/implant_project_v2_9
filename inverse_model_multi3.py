# inverse_model_multi3.py
# Script to fit various cochlear implant measures using the model by Goldwyn et al. (2010) and Perkel et al. (2025)
# for cochlear implant activation.
#
# Parallelized version of inverse_model_multi2.py (which is left untouched as the tested serial
# baseline): the scenario/subject loop is fully independent per iteration -- each scenario only
# reads its own row of the summary accumulator arrays and writes its own output files -- so it's
# distributed across a process pool the same way fwd_model5_2d.py and voltage_calc.py parallelize
# their own independent-iteration loops.
#
# Original code by Steve Bierer.
# Translated to python 3 and adapted to fit both survival and rpos values by David J. Perkel December 2020
# Modified 6 August 2021 by DJP to start by fitting a single electrode at a time. This gives initial conditions
# for each electrode. Then there's a holistic fitting process using all electrode parameters
# Latest version 1 December 2023.
# Major reworking Dec 2025 by DJP to take in multiple types of clinical or research data for fitting.
# These include MP and TP thresholds, most comfortble level (M-level), ecap parameters and perhaps others

import cProfile
import io
import pstats
from pstats import SortKey
import os
import pickle
import concurrent.futures as cf
import scipy.signal as sig
import scipy.stats as stats
from lmfit import Minimizer, Parameters, report_fit

# local files
import set_scenario as s_scen
import surv_full
import get_thresholds2 as gt  # brentq-based bisection replacement; see get_thresholds2.py
import subject_data
import calc_ecap as c_ecap
from common_params import *  # import common values across all models
from inverse_fit_io import load_scenario_or_subject_data, save_scenario_results, save_summary_results
from inverse_fit_initial_guess import estimate_initial_guesses

# User adjustable parameters
fit_mode = 'combined'  # Which variable(s) to fit? Alternatives are 'combined', 'rpos' or 'survival'
if_plot = True  # Whether to plot output at end
unsupervised = False  # If True, makes & saves summary plots but does not display them or wait for user input
if_plot_guess_contours = False  # Option to plot initial guesses for parameters given to the fitting algorithm
fit_tol = 0.1  # Fit tolerance for subject fits
use_minimizer = True  # If true, uses the lmfit wrapper around scipy optimize. Otherwise, vanilla scipy.minimize
if_save_npy = True


def objectivefunc_lmfit_all(par, sigvals, sim_params, f_par, e_field, thr_goals):
    # Repack parameters into arrays
    nel = len(sim_params['electrodes']['zpos'])
    vals = par.valuesdict()
    show_retval = False  # helpful to track progress of the fitting process

    sim_params['electrodes']['rpos'] = np.zeros(nel)


    if not tp_extend:
        # Note: rpos[0]/rpos[-1] and tempsurv/tempsurv2/surv_full below don't depend on the loop
        # variable 'i' -- only the final values (after all rpos entries are set) are ever used, so
        # they're computed once after the loop instead of being recomputed (and surv_full re-run)
        # on every one of the nel-2 iterations.
        for i in range(0, nel - 2):
            varname = 'v_%i' % i
            myvalue = vals[varname]
            sim_params['electrodes']['rpos'][i + 1] = myvalue
        sim_params['electrodes']['rpos'][0] = sim_params['electrodes']['rpos'][1]
        sim_params['electrodes']['rpos'][-1] = sim_params['electrodes']['rpos'][-2]

        tempsurv = np.zeros(nel - 2)
        tempsurv2 = np.zeros(nel)
        for j, loopval in enumerate(range(nel - 2, 2 * (nel - 2))):
            varname = 'v_%i' % (j + nel - 2)
            myvalue = vals[varname]
            tempsurv[j] = myvalue
        tempsurv2[1:nel - 1] = tempsurv
        tempsurv2[0] = tempsurv2[1]
        tempsurv2[-1] = tempsurv2[-2]
        sim_params['neurons']['nsurvival'] = surv_full.surv_full(sim_params['electrodes']['zpos'],
                                                                 tempsurv2, sim_params['grid']['z'])

    else:
        for i in range(0, nel):
            varname = 'v_%i' % i
            myvalue = vals[varname]
            sim_params['electrodes']['rpos'][i] = myvalue

        tempsurv = np.zeros(nel)
        for i, loopval in enumerate(range(nel, 2 * nel)):
            varname = 'v_%i' % (i + nel)
            myvalue = vals[varname]
            tempsurv[i] = myvalue

            tempsurv[0] = tempsurv[1]
            tempsurv[-1] = tempsurv[-2]
        sim_params['neurons']['nsurvival'] = surv_full.surv_full(sim_params['electrodes']['zpos'],
                                                                 tempsurv, sim_params['grid']['z'])

    # Call for monopolar then tripolar
    sim_params['channel']['sigma'] = sigvals[0]
    sim_params['channel']['behavior'] = 0
    weights = np.zeros(len(sim_params['run_info']['fit_measures']))
    weights[0] = sim_params['run_info']['fit_measures']['mp']
    weights[1] = sim_params['run_info']['fit_measures']['tp']
    weights[2] = sim_params['run_info']['fit_measures']['mp_mcl']
    weights[3] = sim_params['run_info']['fit_measures']['ecap']

    if weights[0] > 0.0:
        sim_params['channel']['sigma'] = sigvals[0]   # 0 if looking at M levels
        sim_params['channel']['behavior'] = 0
        thresh_mp = gt.get_thresholds(e_field, f_par, sim_params)
        mp_diff = np.subtract(thresh_mp[0], thr_goals['thrmp_db'])
        mp_err = np.nanmean(np.abs(mp_diff))
    else:
        mp_diff = np.zeros(NELEC)
        mp_err = 0.0

    if weights[1] > 0.0:
        sim_params['channel']['sigma'] = sigvals[1]
        sim_params['channel']['behavior'] = 0
        thresh_tp = gt.get_thresholds(e_field, f_par, sim_params)
        tp_diff = np.subtract(thresh_tp[0][1:nel - 1], thr_goals['thrtp_db'][1:nel - 1])
        tp_err = np.nanmean(np.abs(tp_diff))
    else:
        tp_diff = np.zeros(NELEC)
        tp_err = 0.0

    if weights[2] > 0.0:
        # mp_mcl stuff here
        sim_params['channel']['sigma'] = sigvals[0]  # 0 if looking at M levels
        sim_params['channel']['behavior'] = 1
        mp_mcl_db = gt.get_thresholds(e_field, f_par, sim_params)
        mp_mcl_diff = np.subtract(mp_mcl_db[0], thr_goals['mp_mcl_db'])
        mcl_err = np.nanmean(np.abs(np.subtract(mp_mcl_db[0], thr_goals['mp_mcl_db'])))
    else:
        mp_mcl_diff = np.zeros(NELEC)
        mcl_err = 0.0

    if weights[3] > 0.0:
        raise ValueError("ECAP fitting not implemented yet")

        ## ECAP fitting not ready
        # ec_thresh = 1000  # neurons
        # nstim = 500
        # # need to reset sigma to 0.0
        # sim_params['channel']['sigma'] = sigvals[0]
        # ecap = c_ecap.calc_ecap(act_vals, f_par, sim_params, ec_thresh, thr_sim_db[:, i], nstim)
        # ecap_diff = np.subtract(ecap, thr_goals['ecap'])
        # ecap_err = np.nanmean(np.abs(np.subtract(ecap, thr_goals['ecap'])))
    else:
        ecap_diff = np.zeros(NELEC)
        ecap_err = 0.0

    # Calculate errors
    mean_error = np.sum(np.multiply(weights, [mp_err, tp_err, mcl_err, ecap_err]))

    tempzero = np.zeros(1)
    if tp_extend:
        retval = np.concatenate((mp_diff*weights[0], tempzero, tp_diff*weights[1], tempzero,
                                 mp_mcl_diff*weights[2], ecap_diff*weights[3]))
    else:
        retval = np.concatenate((mp_diff[1:-1]*weights[0], tp_diff*weights[1],
                                 mp_mcl_diff*weights[2], ecap_diff*weights[3]))
    # Returns a vector of errors, with the first and last of the tripolar errors set to zero
    # because they can't be calculated
    if show_retval:  # helpful for debugging
        scen = sim_params['run_info']['scenario']
        print('subj/scen: ', scen, '; tempsurv[4] = ', '%.3f' % float(tempsurv[4]), '; rpos[4]= ',
              '%.3f' % sim_params['electrodes']['rpos'][4],
              '; Mean abs error (dB) = ', '%.3f' % mean_error, '; Max error (dB) = ',
              '%.3f' % np.nanmax(np.abs(retval)))
    return retval


def build_lmfit_parameters(fit_mode, scen, thresh_summary_full, mono_thr, tripol_thr, sig_vals,
                           use_fwd_model, subject, survvals):
    """Build the lmfit Parameters object (with bounds) for one scenario/subject, for whichever
    fit_mode is active. All three original branches ('combined', 'rpos', 'survival') are kept here
    since all three are in active use.

    Returns (par, fitrposvals, fitsurvvals, initvec). For 'combined', fitrposvals/fitsurvvals hold
    the contour-intersection initial guesses (see inverse_fit_initial_guess.py); for 'rpos'/
    'survival' they're left as zeros, exactly as in the original -- those two modes don't use a
    per-electrode contour-based initial guess.
    """
    par = Parameters()
    fitrposvals = np.zeros(NELEC)
    fitsurvvals = np.zeros(NELEC)
    initvec = []
    lb, ub = 0.0, 0.0

    if fit_mode == 'combined':  # Optimize survival and rpos to match MP and TP thresholds
        fitrposvals, fitsurvvals = estimate_initial_guesses(thresh_summary_full, scen, mono_thr, tripol_thr,
                                                            sig_vals, if_plot_guess_contours, use_fwd_model)

        if not tp_extend:
            initvec = np.append(fitrposvals[1:-1], fitsurvvals[1:-1])
            for i, val in enumerate(initvec):  # place values in to the par object
                if i < NELEC - 2:
                    lb, ub = -0.95, 0.95  # lower and upper bounds for position
                elif NELEC - 2 <= i < 2 * (NELEC - 2):
                    lb, ub = 0.0, 1.0  # density
                else:  # TODO leftover from tryng to fit external resistivity. Can remove
                    lb, ub = 125, 2500

                par.add('v_%i' % i, value=initvec[i], min=lb, max=ub)
        else:
            initvec = np.append(fitrposvals, fitsurvvals)
            for i, val in enumerate(initvec):  # place values in to the par object
                if i < NELEC:
                    lb, ub = -0.95, 0.95  # lower and upper bounds for position
                elif NELEC <= i < 2 * NELEC:
                    lb, ub = 0.0, 1.0  # density

                par.add('v_%i' % i, value=initvec[i], min=lb, max=ub)

    elif fit_mode == 'rpos':  # fit rpos only; hold survival fixed as the values loaded from the scenario
        if use_minimizer:
            initvec = np.append(fitrposvals[1:-1], fitsurvvals[1:-1])
            for i, val in enumerate(initvec):  # place values in to the par object
                if i < NELEC - 2:
                    lb, ub = -0.95, 0.95  # lower and upper bounds for position
                elif NELEC - 2 <= i < 2 * (NELEC - 2):
                    lb, ub = 0.05, 0.95  # density
                else:  # TODO leftover from tryng to fit external resistivity. Can remove
                    lb, ub = 125, 2500

                par.add('v_%i' % i, value=initvec[i], min=lb, max=ub)
        else:
            initvec = np.append(np.zeros(NELEC), survvals)
            for i, val in enumerate(initvec):
                if i < 16:
                    par.add('v_%i' % i, value=initvec[i], min=-0.85, max=0.85)  # lower and upper bounds
                else:
                    par.add('v_%i' % i, value=initvec[i], vary=False)

    elif fit_mode == 'survival':
        pos_slack = 0.05  # amount by which we allow rpos to move
        if use_fwd_model:
            rposvals = electrodes['rpos']
        else:
            rposvals = subject_data.subj_ct_data(subject)
        radius = 1.0

        if not tp_extend:
            initvec = np.append(rposvals[1:-1], (np.ones(NELEC - 2) * 0.5))
            if use_minimizer:
                for i, val in enumerate(initvec):  # place values in to the par object
                    if i < NELEC - 2:
                        lb = initvec[i] - pos_slack  # lower and upper bounds for position
                        ub = initvec[i] + pos_slack
                        if lb <= -1 * radius + .05:
                            lb = -1 * radius + .05
                    elif NELEC - 2 <= i < 2 * (NELEC - 2):
                        lb, ub = 0.1, 0.9  # density

                    par.add('v_%i' % i, value=initvec[i], min=lb, max=ub)
            else:
                for i, val in enumerate(initvec):
                    if i < 16:
                        lb = rposvals[i] - pos_slack  # lower and upper bounds
                        ub = rposvals[i] + pos_slack
                        if lb <= -1 * radius:
                            lb = -1 * radius + .05
                        if ub <= -1 * radius + .95:
                            ub = -1 * radius + .95
                    else:
                        lb, ub = 0.1, 1.0
                    par.add('v_%i' % i, value=initvec[i], min=lb, max=ub)
        else:
            initvec = np.append(rposvals, (np.ones(NELEC) * 0.5))
            if use_minimizer:
                for i, val in enumerate(initvec):  # place values in to the par object
                    if i < NELEC:
                        lb = initvec[i] - pos_slack  # lower and upper bounds for position
                        ub = initvec[i] + pos_slack
                        if lb <= -1 * radius:
                            lb = -1 * radius + .05
                        if ub <= -1 * radius + .95:
                            ub = -1 * radius + .95
                    else:
                        lb, ub = 0.0, 1.0  # density

                    par.add('v_%i' % i, value=initvec[i], min=lb, max=ub)
            else:
                for i, val in enumerate(initvec):
                    if i < 16:
                        lb = rposvals[i] - pos_slack  # lower and upper bounds
                        ub = rposvals[i] + pos_slack
                        if lb <= -1 * radius:
                            lb = -1 * radius + .05
                        if ub <= -1 * radius + .95:
                            ub = -1 * radius + .95
                    else:
                        lb, ub = 0.1, 1.0
                    par.add('v_%i' % i, value=initvec[i], min=lb, max=ub)
    else:  # none of the planned fit_modes
        raise Exception('Incorrect fit_mode ', fit_mode, ' provided.')

    return par, fitrposvals, fitsurvvals, initvec


def run_fit_and_extract(par, sig_vals, fp, act_vals, thr_data):
    """Run the lmfit Minimizer for one scenario/subject, extract the fitted rpos/survival vectors,
    and simulate MP/TP thresholds from the fit. Returns (fitrposvals, fitsurvvals, thrsimmp, thrsimtp).
    """
    minner = Minimizer(objectivefunc_lmfit_all, par, nan_policy='omit',
                       fcn_args=(sig_vals, sim_params, fp, act_vals, thr_data))
    result = minner.minimize(method='least_squares', ftol=fit_tol, diff_step=0.1)

    fitrposvals = np.zeros(NELEC)
    fitsurvvals = np.zeros(NELEC)
    if not tp_extend:  # store the results in the right place
        for i in range(NELEC - 2):
            vname = 'v_%i' % i
            fitrposvals[i + 1] = result.params[vname]
            vname = 'v_%i' % (i + NELEC - 2)
            fitsurvvals[i + 1] = result.params[vname]

        fitrposvals[0] = fitrposvals[1]
        fitrposvals[-1] = fitrposvals[-2]
        fitsurvvals[0] = fitsurvvals[1]
        fitsurvvals[-1] = fitsurvvals[-2]
    else:
        for i in range(NELEC):
            vname = 'v_%i' % i
            fitrposvals[i] = result.params[vname]
            vname = 'v_%i' % (i + NELEC)
            fitsurvvals[i] = result.params[vname]

    sim_params['electrodes']['rpos'] = fitrposvals

    if use_minimizer:
        report_fit(result)
        result.params.pretty_print()
    else:
        print('fitrposvals: ', fitrposvals)
        print('fitsurvvals: ', fitsurvvals)

    sim_params['electrodes']['rpos'] = fitrposvals
    sim_params['neurons']['nsurvival'] = surv_full.surv_full(sim_params['electrodes']['zpos'], fitsurvvals,
                                                             sim_params['grid']['z'])

    sim_params['channel']['sigma'] = sig_vals[0]
    sim_params['channel']['behavior'] = 0
    thrsimmp = gt.get_thresholds(act_vals, fp, sim_params)
    sim_params['channel']['sigma'] = sig_vals[1]
    sim_params['channel']['behavior'] = 0  # 1 for M levels
    thrsimtp = gt.get_thresholds(act_vals, fp, sim_params)

    return fitrposvals, fitsurvvals, thrsimmp, thrsimtp


def compute_fit_errors(scen, scenario, subject, use_fwd_model, thr_data, thrsimmp, thrsimtp,
                       fitrposvals, fitsurvvals, thresh_err_summary, thresh_summary_full,
                       rpos_summary, rpos_err_summary, surv_err_summary, dist_corr, dist_corr_p):
    """Compute per-scenario threshold/position/survival error metrics, mutating the summary
    accumulator arrays/lists in place (same style as the original). Returns a dict of the values
    save_scenario_results() needs to write the per-scenario CSV/npy files."""
    thrsim = [thrsimmp[0]], [thrsimtp[0]]
    thrtargs = [[thr_data['thrmp_db']], [thr_data['thrtp_db']]]

    mean_mp_thr_err = np.nanmean(np.abs(np.array(thrsim[0]) - np.array(thrtargs[0])))
    mean_tp_thr_err = np.nanmean(np.abs(np.array(thrsim[1]) - np.array(thrtargs[1])))

    thresh_err_summary[scen, 0] = mean_mp_thr_err
    thresh_err_summary[scen, 1] = mean_tp_thr_err
    thresh_summary_full[scen, 1, 0, :] = np.array(thrsim[0])
    thresh_summary_full[scen, 1, 1, :] = np.array(thrsim[1])

    ct_vals = None
    survvals = None
    rposvals = None

    if use_fwd_model:
        [survvals, rposvals, espace] = s_scen.set_scenario(scenario, NELEC)
        if not tp_extend:
            rposerrs = np.subtract(rposvals[1:-1], fitrposvals[1:-1])
            survivalerrs = np.abs(np.subtract(survvals[1:-1], fitsurvvals[1:-1]))
        else:
            rposerrs = np.subtract(rposvals, fitrposvals)
            survivalerrs = np.abs(np.subtract(survvals, fitsurvvals))

        rpos_summary.append([rposvals, fitrposvals])
        rpos_err_metric = np.mean(np.abs(rposerrs))
        rpos_err_summary[scen] = rpos_err_metric
        surv_err_summary[scen] = np.mean(np.abs(survivalerrs))

        if not tp_extend:
            distvals = 1 - rposvals[1:-1]
            if np.std(distvals) == 0.0:
                distvals = distvals + np.random.normal(0.0, 1e-20, size=len(distvals))  # to prevent a nan from stats.pearsonr
            [dist_corr[scen], dist_corr_p[scen]] = stats.pearsonr(distvals, 1 - fitrposvals[1:-1])
        else:
            [dist_corr[scen], dist_corr_p[scen]] = stats.pearsonr(1 - rposvals, 1 - fitrposvals)

    else:  # for subjects
        ct_vals = subject_data.subj_ct_data(subject)
        survivalerrs = np.empty(NELEC)

        if np.any(ct_vals):
            if not tp_extend:
                rposerrs = np.abs(np.subtract(fitrposvals[1:-1], ct_vals[1:-1]))
                rpos_summary.append([fitrposvals[1:-1], ct_vals[1:-1]])
                rpos_err_metric = np.mean(np.abs(rposerrs))
                rpos_err_summary[scen] = rpos_err_metric
                [dist_corr[scen], dist_corr_p[scen]] = stats.pearsonr(1 - ct_vals[1:-1], 1 - fitrposvals[1:-1])
            else:
                rposerrs = np.abs(np.subtract(fitrposvals, ct_vals))
                rpos_err_metric = np.mean(np.abs(rposerrs))
                rpos_summary.append([fitrposvals, ct_vals])
                rpos_err_summary[scen] = rpos_err_metric
                [dist_corr[scen], dist_corr_p[scen]] = stats.pearsonr(1 - ct_vals, 1 - fitrposvals)
        else:
            rpos_summary.append(fitrposvals)
            rposerrs = np.empty(NELEC)
            rpos_err_metric = np.nan
            rpos_err_summary[scen] = np.nan

    return {
        'thrsim': thrsim, 'thrtargs': thrtargs, 'ct_vals': ct_vals, 'survvals': survvals,
        'rposvals': rposvals, 'rposerrs': rposerrs, 'survivalerrs': survivalerrs,
        'rpos_err_metric': rpos_err_metric,
    }


# --- Worker-process state and functions -------------------------------------------------------
# These run inside each worker process (started by ProcessPoolExecutor). '_worker_fp' and
# '_worker_act_vals' are populated once per worker by '_init_worker', instead of being re-loaded
# for every scenario that worker processes.

_worker_fp = None
_worker_act_vals = None


def _init_worker(fieldtable_path, neuron_overrides=None):
    """Pool initializer: runs once in each worker process. Loads the field table locally, and
    (for mode == 'survey') applies the same NEURONS overrides the main process would otherwise
    have set before the loop -- workers get a fresh import of common_params.py (spawn), so any
    global mutation the main process makes *after* import but before the pool starts would
    otherwise never reach them.
    """
    global _worker_fp, _worker_act_vals
    with open(fieldtable_path, 'rb') as combined_data:
        data = pickle.load(combined_data)
    fp = data[0]
    fp['zEval'] = np.array(fp['zEval'])
    _worker_fp = fp
    _worker_act_vals = data[2]
    sim_params['grid']['table'] = _worker_act_vals
    if neuron_overrides:
        NEURONS.update(neuron_overrides)


def _process_scenario(scen, scenario, fit_mode_local, if_plot_local, if_save_npy_local, unsupervised_local):
    """Worker task: fully process one scenario/subject -- load its data, build fit parameters, run
    the optimizer, compute error metrics, and save its own output files. Scenarios are independent
    of each other (each only touches its own row of the summary arrays and its own output files),
    so this is the per-scenario equivalent of the original serial for-loop body.

    Returns (scen, thresh_err_row, thresh_summary_full_row, rpos_summary_entry, rpos_err_val,
    surv_err_val, dist_corr_val, dist_corr_p_val) for the caller to place into the shared summary
    arrays at index scen.
    """
    fp = _worker_fp
    act_vals = _worker_act_vals

    # Single-scenario-sized local accumulator; only index 0 is ever touched.
    local_thresh_summary_full = np.zeros((1, 2, 2, NELEC))

    (use_fwd_model, sig_vals, espace, thr_data, subject,
     mono_thr, tripol_thr, overall_offset_db) = load_scenario_or_subject_data(0, scenario, local_thresh_summary_full)

    ELEC_MIDPOINT = GRID['z'][-1] / 2.0  # electrode array midpoint
    ARRAY_BASE = -(np.arange(NELEC - 1, -1, -1) * espace)
    array_mid = (ARRAY_BASE[0] + ARRAY_BASE[-1]) / 2.0
    electrodes['zpos'] = (ELEC_MIDPOINT - array_mid) + ARRAY_BASE

    survvals = np.empty(NELEC)
    survvals[:] = np.nan
    ct_data = {'stdiameter': [], 'scala': [], 'elecdist': [], 'espace': espace, 'type': [], 'insrt_base': [],
               'insert_apex': []}
    radius = 1.0
    ct_data['stdiameter'] = radius * 2.0 * (np.zeros(NELEC) + 1.0)

    if np.isnan(thr_data['thrtp_sigma']) or thr_data['thrtp_sigma'] < 0.75 or thr_data['thrtp_sigma'] > 1:
        # The original called exit() here, which would stop the whole (serial) script. A worker
        # process can't do that for the others, so raise instead -- fut.result() re-raises this in
        # the main process, which stops result collection the same way exit() would have.
        raise ValueError(f"The sigma value for the TP configuration is invalid for scenario '{scenario}'.")

    fradvec = (ct_data['stdiameter'] / 2)  # smooth the radius data!!
    fradvec = sig.filtfilt(np.hanning(5) / sum(np.hanning(5)), 1, fradvec)
    sim_params['cochlea']['radius'] = fradvec
    avec = np.arange(0, 1.005, 0.01)  # create the neuron count to neuron spikes transformation
    rlvec = NEURONS['coef'] * (np.power(avec, 2.0)) + (1 - NEURONS['coef']) * avec
    rlvec = NEURONS['neur_per_clust'] * np.power(rlvec, NEURONS['power'])
    rlvltable = np.stack((avec, rlvec))
    sim_params['neurons']['rlvl'] = rlvltable

    # Construct the sim_params structure
    sim_params['cochlea'] = COCHLEA
    sim_params['electrodes'] = electrodes
    sim_params['channel'] = CHANNEL
    sim_params['channel']['sigma'] = 0.0

    par, fitrposvals, fitsurvvals, initvec = build_lmfit_parameters(
        fit_mode_local, 0, local_thresh_summary_full, mono_thr, tripol_thr, sig_vals, use_fwd_model,
        subject, survvals)

    # fix for a problem but needs a better fix
    thr_data['thrmp_db'] = local_thresh_summary_full[0, 0, 0, :]

    fitrposvals, fitsurvvals, thrsimmp, thrsimtp = run_fit_and_extract(par, sig_vals, fp, act_vals, thr_data)

    local_thresh_err_summary = np.zeros((1, 2))
    local_rpos_err_summary = np.zeros(1)
    local_surv_err_summary = np.zeros(1)
    local_dist_corr = np.zeros(1)
    local_dist_corr_p = np.zeros(1)
    local_rpos_summary = []

    fit_results = compute_fit_errors(0, scenario, subject, use_fwd_model, thr_data, thrsimmp, thrsimtp,
                                     fitrposvals, fitsurvvals, local_thresh_err_summary, local_thresh_summary_full,
                                     local_rpos_summary, local_rpos_err_summary, local_surv_err_summary,
                                     local_dist_corr, local_dist_corr_p)

    save_scenario_results(scen, scenario, subject, use_fwd_model, fit_mode_local, thr_data, thrsimmp, thrsimtp,
                          fitrposvals, fitsurvvals, sig_vals, initvec, overall_offset_db, fit_results,
                          if_plot_local, if_save_npy_local, unsupervised_local)

    return (scen, local_thresh_err_summary[0], local_thresh_summary_full[0], local_rpos_summary[0],
            local_rpos_err_summary[0], local_surv_err_summary[0], local_dist_corr[0], local_dist_corr_p[0])


def inverse_model_multi(mode):  # Start this script
    neuron_overrides = None
    if mode == 'main':
        pass  # proceed as normal
    elif mode == 'survey':
        # load survey params
        import csv
        param_file = 'surv_params.txt'
        tempdata = np.zeros(4)  # 4 values
        if os.path.exists(param_file):
            with open(param_file, newline='') as csvfile:
                datareader = csv.reader(csvfile, delimiter=',')
                for i, row in enumerate(datareader):
                    tempdata[i] = row[0]

        NEURONS['act_stdrel'] = tempdata[1]
        NEURONS['act_ctr'] = tempdata[2]
        NEURONS['thrtarg'] = 100
        espace = tempdata[3]
        # Workers get a fresh import of common_params.py (spawn), so these overrides need to be
        # replayed inside each worker via _init_worker rather than relying on this mutation of the
        # main process's own NEURONS dict.
        neuron_overrides = {'act_stdrel': NEURONS['act_stdrel'], 'act_ctr': NEURONS['act_ctr'],
                            'thrtarg': NEURONS['thrtarg']}
    else:  # should not happen
        print('fwd_model called with unrecognized mode: ', mode)
        exit()

    if not os.path.isdir(INV_OUT_PRFIX):  # if the inverse output directory doesn't exist, make it
        os.mkdir(INV_OUT_PRFIX)
        os.mkdir(INVOUTPUTDIR)
    else:
        if not os.path.isdir(INVOUTPUTDIR):
            os.mkdir(INVOUTPUTDIR)

    pr = cProfile.Profile()
    pr.enable()  # Start the profiler
    # Note: this only profiles the main process below (task submission/collection), not the actual
    # per-scenario fitting, which now runs in worker processes and won't show up here.

    num_scen = len(scenarios)
    # Set up array for summary values
    thresh_err_summary = np.zeros((num_scen, 2))
    thresh_summary_full = np.zeros((num_scen, 2, 2, NELEC))  # indices: scenario, (target, fit), sigma, electrodes
    rpos_summary = [None] * num_scen
    rpos_err_summary = np.zeros(num_scen)
    surv_err_summary = np.zeros(num_scen)
    dist_corr = np.zeros(num_scen)
    dist_corr_p = np.zeros(num_scen)

    # Scenarios are fully independent of each other (each only touches its own row of the summary
    # arrays and its own output files), so the loop is parallelized across processes rather than
    # run serially, the same way fwd_model5_2d.py and voltage_calc.py parallelize their own
    # independent-iteration loops.
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print('Parallelizing scenario loop across ', n_workers, ' worker processes')

    with cf.ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 initargs=(FIELDTABLE, neuron_overrides)) as executor:
        futures = [executor.submit(_process_scenario, scen, scenarios[scen], fit_mode, if_plot,
                                    if_save_npy, unsupervised)
                   for scen in range(num_scen)]

        n_done = 0
        for fut in cf.as_completed(futures):
            (scen, te_row, tsf_row, rpos_entry, rpos_err_val, surv_err_val,
             dcorr_val, dcorr_p_val) = fut.result()
            thresh_err_summary[scen, :] = te_row
            thresh_summary_full[scen, :, :, :] = tsf_row
            rpos_summary[scen] = rpos_entry
            rpos_err_summary[scen] = rpos_err_val
            surv_err_summary[scen] = surv_err_val
            dist_corr[scen] = dcorr_val
            dist_corr_p[scen] = dcorr_p_val
            n_done += 1
            print('scenario ', n_done, ' of ', num_scen, ' complete (', scenarios[scen], ')')

    # Now we are done with the loop on scenarios/subjects
    pr.disable()  # stop the profiler
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(10)
    print(s.getvalue())

    print('Completed inverse model on ', num_scen, ' scenarios in total time of: %.2f' % ps.total_tt, ' s.')

    save_summary_results(scenarios, thresh_err_summary, rpos_err_summary, surv_err_summary,
                         dist_corr, dist_corr_p, thresh_summary_full, rpos_summary, if_save_npy)


if __name__ == '__main__':
    inverse_model_multi(mode='main')  # main or 'survey'
