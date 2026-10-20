# Forward model 2D version -- parallelized

# This 2D version uses a single electrode and scans through an arbitrary set of values for survival values and
# electrode position
# It is critical to note that the parameter values do not to be tied to the # of electrodes in an implant
# David J. Perkel 7 March 2019
# Translated to python 8 March 2020
# Reorganizing the code to be more modular and to allow sharing of code
# with inverse model, for simplicity and reproducibility
# This version runs through a set of 'scenarios', each with a specific set
# of electrode radial positions and neuron survival values. For each, it
# calculates predicted threshold for monopolar and partial tripolar
# stimulation, typically sigma = 0 and 0.9.
# Re-written 12 August 2021 to simplify data structures into dicts
#
# fwd_model5_2d.py -- based on FwdModel4_2D.py, with the (survVals x rposVals) sweep parallelized
# across processes. The sweep is embarrassingly parallel (each (surv, rpos, sigma) combination is
# independent), but the original code relied on mutating one shared, module-level 'simParams' dict
# in place, which is not safe to hand to multiple workers. Here, each worker task gets its own
# private copy of sim_params (built fresh from a template) and processes one full row of the sweep
# (all rposVals x all_sigs for a single survival value), returning just that row's results. The
# field table is small, so it's loaded once per worker process via a pool initializer rather than
# re-loaded or re-pickled for every task.

import surv_full
import get_thresholds2 as gt  # brentq-based bisection replacement; see get_thresholds2.py
import datetime
import os
import csv
import copy
import concurrent.futures as cf
import matplotlib.pyplot as plt
import pickle
import calc_ecap as c_ecap
from common_params import *
import figs4_5_2D_contour
import fig3_neuron_activation


# --- Worker-process state and functions -------------------------------------------------------
# These run inside each worker process (started by ProcessPoolExecutor). '_worker_fp' and
# '_worker_act_vals' are populated once per worker by '_init_worker', instead of being re-loaded
# or re-pickled for every task.

_worker_fp = None
_worker_act_vals = None


def _init_worker(fieldtable_path):
    """Pool initializer: runs once in each worker process. Loads the field table locally so it
    doesn't need to be pickled and sent along with every single task."""
    global _worker_fp, _worker_act_vals
    with open(fieldtable_path, 'rb') as combined_data:
        data = pickle.load(combined_data)
    fp = data[0]
    fp['zEval'] = np.array(fp['zEval'])
    _worker_fp = fp
    _worker_act_vals = data[2]


def _process_surv_row(j, surv, rpos_vals, all_sig_vals, na_sig, ndim, mcl_idx, all_measures_local,
                       ec_thresh, n_stim, sim_params_template):
    """Worker task: compute one full row (fixed survival value, all radial positions and sigma
    values) of the 2D sweep.

    A private deep copy of sim_params_template is made here so this task never reads or writes
    any state shared with the main process or other workers -- required for this to be safe under
    multiprocessing (the original serial code mutated one shared global dict in place instead).
    """
    sim_params = copy.deepcopy(sim_params_template)
    sim_params['grid']['table'] = _worker_act_vals

    nz = len(sim_params['grid']['z'])
    nel = len(sim_params['electrodes']['zpos'])
    nrpos = len(rpos_vals)

    thr_row = np.full((nrpos, na_sig), np.nan)
    neuronact_row = np.full((nrpos, ndim, 1, nz), np.nan)
    m_level_row = np.full(nrpos, np.nan)
    ecap_row = np.zeros((nrpos, n_stim, 2))

    # Get survival values for all neuron clusters from the single survival value for this row
    sim_params['neurons']['nsurvival'] = surv_full.surv_full(
        sim_params['electrodes']['zpos'], np.ones(nel) * surv, sim_params['grid']['z'])

    for k, rpos in enumerate(rpos_vals):
        sim_params['electrodes']['rpos'] = rpos

        if 'threshold' in all_measures_local:
            # Calculate thresholds for all sigma values in all_sig_vals from common_params
            sim_params['channel']['behavior'] = 0
            for i in range(na_sig):
                sim_params['channel']['sigma'] = all_sig_vals[i]
                tempvals = gt.get_thresholds(_worker_act_vals, _worker_fp, sim_params)
                thr_row[k, i] = tempvals[0].item()
                nexttemp2 = tempvals[1]
                if np.max(nexttemp2) == 0:
                    print("flat activation")
                neuronact_row[k, i, :, :] = nexttemp2

        if 'mp_mcl' in all_measures_local:
            sim_params['channel']['sigma'] = 0
            sim_params['channel']['behavior'] = 1  # search for m_level, not threshold
            tempvals = gt.get_thresholds(_worker_act_vals, _worker_fp, sim_params)
            m_level_row[k] = tempvals[0].item()
            nexttemp2 = tempvals[1]
            if np.max(nexttemp2) == 0:
                print("flat activation")
            # Note: the original serial version (FwdModel4_2D.py) wrote this into neuronact[j, k, i, :, :],
            # reusing 'i' left over from the 'threshold' loop above and so overwriting that sigma's
            # slot. mp_mcl gets its own dedicated slot here instead (mcl_idx, computed from 'ndim').
            neuronact_row[k, mcl_idx, :, :] = nexttemp2

        if 'ecap' in all_measures_local:
            # calculate ecap growth curve for stim intensities within the range defined in calc_ecap
            # (the 'thr_sim_db' argument is currently unused inside calc_ecap beyond commented-out
            # lines; passed here as this row's own threshold values rather than the original's
            # unrelated thr_sim_db[:, i] slice)
            ecap_temp = c_ecap.calc_ecap(_worker_act_vals, _worker_fp, sim_params, ec_thresh,
                                         thr_row[k, :], n_stim)
            ecap_row[k, :, :] = ecap_temp

    return j, thr_row, neuronact_row, m_level_row, ecap_row


# --- Main entry point --------------------------------------------------------------------------

def fwd_model_2d(mode):

    n_stim = 4000  # number of stim levels for ecap calculation
    ec_thresh = 300  # neurons; ecap growth curve threshold target
    # TODO:  now loops on espace values. Looping can now be removed from the param_survey script
    if mode == 'main':
        espace_range = [0.85, 1.1]

        pass  # proceed as normal
    elif mode == 'survey':
        # load survey params
        param_file = 'surv_params.txt'
        tempdata = np.zeros(4)  # 4 values
        if os.path.exists(param_file):
            with open(param_file, newline='') as csvfile:
                datareader = csv.reader(csvfile, delimiter=',')
                csvfile.seek(0)
                for i, row in enumerate(datareader):
                    # Do the parsing
                    tempdata[i] = row[0]

        #  res_ext = tempdata[0]  # this variable is not used in this context.
        NEURONS['act_stdrel'] = tempdata[1]
        NEURONS['thrtarg'] = 100
        NEURONS['act_ctr'] = tempdata[2]
        espace_range = tempdata[3]
    else:  # should not happen
        print('fwd_model called with unrecognized mode: ', mode)
        exit()

    # We depend on a voltage and activation tables calculated using voltage_calc.py and saved as a .dat file
    with open(FIELDTABLE, 'rb') as combined_data:
        data = pickle.load(combined_data)
        combined_data.close()

    # Unpack data from the field data file
    fp = data[0]
    #  vVals = data[1]  # Voltage values are loaded from the file but are not needed
    act_vals = data[2]
    # Convert zEval to a numpy array
    fp['zEval'] = np.array(fp['zEval'])

    GRID['table'] = act_vals

    COCHLEA['res1'] = fp['resInt'] * np.ones(NELEC)  # Note these valus do not match thos of Goldwyn et al., 2010
    COCHLEA['res2'] = fp['resExt'] * np.ones(NELEC)  # resistivities are in Ohm*cm (conversion to Ohm*mm occurs later)
    GRID['r'] = fp['rspace']  # only 1 of the 3 cylindrical dimensions can be a vector (for CYLINDER3D_MAKEPROFILE)

    # The lines below are not used, but represent infrastructure for nonlinear loudness growth curves or
    # ecap amplitude growth
    # coef = 0  # convex: -1 | 0.4, 0.9 ; linear: 0 | 1; concave: +1 | 1.0, 1.8
    # POW = 1
    # SIDELOBE = 1
    ################################

    if_plot = False  # Whether to plot the results

    # surv_grid_vals, rpos_grid_vals, n_surv and n_rpos now live in common_params.py

    # set up filename
    descrip = 'surv_%.2f' % np.min(surv_grid_vals) + '_%.2f' % np.max(surv_grid_vals) + "_rpos_%.2f" %\
              np.min(rpos_grid_vals) + '_%.2f' % np.max(rpos_grid_vals) + hires

    if not os.path.isdir(FWDOUTPUTDIR):
        os.mkdir(FWDOUTPUTDIR)

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print('Parallelizing sweep across ', n_workers, ' worker processes')

    # One pool, reused across all espace values below -- avoids re-spawning worker processes
    # (and re-running the initializer) once per espace value.
    with cf.ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 initargs=(FIELDTABLE,)) as executor:

        for this_es in espace_range:
            if this_es == 0.85:
                e_txt = '085'
            elif this_es == 1.1:
                e_txt = '110'
            else:
                e_txt = 'xxx'
            es_text = '_espace_' + e_txt

            outfile = FWDOUTPUTDIR + 'FwdModelOutput_' + descrip + es_text + '.csv'

            # Additional setup
            COCHLEA['timestamp'] = datetime.datetime.now()
            electrodes['timestamp'] = datetime.datetime.now()
            electrodes['rpos'] = np.zeros(NELEC)
            ELEC_MIDPOINT = GRID['z'][-1] / 2.0  # electrode array midpoint
            array_base = -(np.arange(NELEC - 1, -1, -1) * this_es)
            array_mid = (array_base[0] + array_base[-1]) / 2.0
            electrodes['zpos'] = (ELEC_MIDPOINT - array_mid) + array_base

            CHANNEL['timestamp'] = datetime.datetime.now()
            COCHLEA['radius'] = np.ones(NELEC) * fp['cylRadius']  # rpos is in mm, so be sure they fit inside the radius

            # Construct the simParams structure
            sim_params['cochlea'] = COCHLEA
            sim_params['electrodes'] = electrodes
            sim_params['channel'] = CHANNEL
            sim_params['grid'] = GRID

            # Example of neural activation at threshold (variants of Goldwyn paper) ; using choices for channel, etc.,
            # made above. Keep in mind that the activation sensitivity will be FIXED, as will the number of neurons required
            # for threshold. Therefore, the final current to achieve theshold will vary according to the simple minimization
            # routine. Also note this works much faster when the field calculations are performed with a look-up table.
            avec = np.arange(0, 1.01, .01)  # create the neuron count to neuron spikes transformation
            rlvec = NEURONS['coef'] * (avec ** 2) + (1 - NEURONS['coef']) * avec
            rlvec = NEURONS['neur_per_clust'] * (rlvec ** NEURONS['power'])
            rlvltable = np.stack((avec, rlvec))  # start with the e-field(s) created above, but remove the current scaling

            n_sig = len(sigma_vals)  # sigmaVals and all_sigs come from common_params
            na_sig = len(all_sigs)
            # n_behavior = len(behaviorVals)
            #  how many entries in the neuronact 3rd dimension? One each for mp & tp threshold plus 1 for mp_mcl
            ndim = 0
            for entry in all_measures:
                if entry == 'threshold':
                    ndim += na_sig
                if entry == 'mp_mcl':
                    ndim += 1
            mcl_idx = na_sig if 'threshold' in all_measures else 0  # dedicated neuronact slot for mp_mcl

            thr_sim_db = np.empty((n_surv, n_rpos, na_sig))  # Array for threshold data for different elecs and sigma values
            thr_sim_db[:] = np.nan
            neuronact = np.empty((n_surv, n_rpos, ndim, 1, len(GRID['z'])))  # Only 1 electrode in this model
            neuronact[:] = np.nan

            # Get survival values for all neuron clusters from the 16 values at electrode positions.
            sim_params['neurons']['rlvl'] = rlvltable
            m_level_db = np.empty((n_surv, n_rpos))  # Array for threshold data for different stim elecs and sigma values
            m_level_db[:] = np.nan
            ecap_all = np.zeros((n_surv, n_rpos, n_stim, 2))

            # Sanity check. Could add other sanity checks here
            # if any(simParams.grid.r < 1):
            # raise('Ending script. One or more evaluation points are inside cylinder; not appropriate for neural activation.')

            # Build a template sim_params for this espace value. Each worker task deep-copies this
            # for itself, so mutating it inside a worker never touches this (or any other worker's)
            # state.
            sim_params_template = copy.deepcopy(sim_params)

            # Determine threshold for each value of sigma, for every survival value and radial position.
            # Each worker task handles one full row (fixed survival value, looping over rpos_grid_vals and
            # all_sigs internally) -- this matches the original serial loop structure, just split across
            # processes, and keeps the number of tasks (nSurv) reasonably matched to available cores.
            futures = [executor.submit(_process_surv_row, j, surv, rpos_grid_vals, all_sigs, na_sig, ndim,
                                        mcl_idx, all_measures, ec_thresh, n_stim, sim_params_template)
                       for j, surv in enumerate(surv_grid_vals)]

            n_done = 0
            for fut in cf.as_completed(futures):
                j, thr_row, neuronact_row, m_level_row, ecap_row = fut.result()
                thr_sim_db[j, :, :] = thr_row
                neuronact[j, :, :, :, :] = neuronact_row
                m_level_db[j, :] = m_level_row
                ecap_all[j, :, :, :] = ecap_row
                n_done += 1
                print('surv row ', n_done, ' of ', n_surv, ' complete (j=', j, ')')

            # Write results to a CSV file
            if 'threshold' in all_measures:
                with open(outfile, mode='w') as data_file:
                    data_writer = csv.writer(data_file, delimiter=',', quotechar='"')

                    for sig_idx, sig in enumerate(all_sigs):
                        header1 = ['Thresholds for sig: ', sig, '; rpos values in columns']
                        header2 = ['Survival ']
                        for i, rpos in enumerate(rpos_grid_vals):
                            header2 += str(rpos) + ', '

                            data_writer.writerow(header1)
                            data_writer.writerow(header2)
                            for row, s_val in enumerate(surv_grid_vals):
                                data_writer.writerow([row, s_val, thr_sim_db[row, i, :]])

                    if 'mp_mcl' in all_measures:
                        header1 = 'Most comfortable levels: rpos values in columns; sigma = 0.0'
                        data_writer.writerow(header1)
                        data_writer.writerow(header2)
                        for row, s_val in enumerate(surv_grid_vals):
                            data_writer.writerow([row, s_val, m_level_db[row]])

            data_file.close()

            if 'threshold' in all_measures:
                # loop on sigma values
                for i in range(0, na_sig):  # number of sigma values to test
                    sigval = all_sigs[i]
                    sigtext = f"{sigval:.1f}".replace(".", "_")
                    filename = FWDOUTPUTDIR + '2D_' + sigtext + '_' + STD_TEXT + es_text + '.csv'
                    np.savetxt(filename, thr_sim_db[:, :, i], delimiter=',')
            if 'mp_mcl' in all_measures:
                np.savetxt(FWDOUTPUTDIR + 'MP_mcl_2D_' + STD_TEXT + es_text + '.csv', m_level_db[:, :], delimiter=',')

            spname = FWDOUTPUTDIR + 'simParams' + descrip + es_text + '.pickle'
            with open(spname, 'wb') as f:
                pickle.dump(sim_params, f, pickle.HIGHEST_PROTOCOL)
                f.close()
            print('saved: ', spname)
            # Note that this is saving only the last simParams structure from the loops on sigma and in get_thresholds.

            # display min and max threshold values
            print('min MP thr:  ', np.min(thr_sim_db[:, :, 0]))
            print('max MP thr:  ', np.max(thr_sim_db[:, :, 0]))
            print('min TP thr:  ', np.min(thr_sim_db[:, :, 1]))
            print('max TP thr:  ', np.max(thr_sim_db[:, :, 1]))

            # Save neuron activation data into a binary file
            np.savez(FWDOUTPUTDIR + 'neuronact_' + STD_TEXT + es_text, surv_grid_vals, rpos_grid_vals, neuronact)

            #  Save ecap data to a binary file
            np.savez(FWDOUTPUTDIR + 'ecap_' + STD_TEXT + es_text, surv_grid_vals, rpos_grid_vals, ecap_all)

    # Plot the results
    if if_plot:
        figs4_5_2D_contour.fig_2D_contour()
        fig3_neuron_activation.fig3_neuron_activation()
        plt.show()


if __name__ == '__main__':
    fwd_model_2d('main')  # mode is 'main' or 'survey'
