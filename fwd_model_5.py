# Forward Model version 5. 20 Feb 2026

import pickle
import datetime
import os
import csv
import matplotlib.pyplot as plt
#  These are local to this project
import set_scenario as s_scen
import surv_full
import get_thresholds2 as gt
import calc_ecap as c_ecap
from common_params import *


def fwd_model_5(mode):

    if mode == 'main':
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
                    # Load each row
                    tempdata[i] = row[0]

        # parse values loaded from the table
        NEURONS['act_stdrel'] = tempdata[1]
        NEURONS['act_ctr'] = tempdata[2]
        NEURONS['thrtarg'] = 100
        espace = tempdata[3]  # should be overridden by scenario/subject
    else:  # should not happen
        print('fwd_model called with unrecognized mode: ', mode)
        exit()

    # We depend on voltage and activation tables calculated using
    # voltage_calc.py and saved as a .dat file. the file is specified in common_params.py
    with open(FIELDTABLE, 'rb') as combined_data:
        data = pickle.load(combined_data)
        combined_data.close()

    fp = data[0]
    fp['zEval'] = np.array(fp['zEval'])
    #  v_vals = data[1] #  voltage values not used
    act_vals = data[2]
    GRID['table'] = act_vals

    COCHLEA['res1'] = fp['resInt'] * np.ones(NELEC)  # Note these values do not match those of Goldwyn et al., 2010
    COCHLEA['res2'] = fp['resExt'] * np.ones(NELEC)  # resistivities are in Ohm*cm (conversion to Ohm*mm occurs later)
    GRID['r'] = fp['rspace']  # only 1 of the 3 cylindrical dimensions can be a vector (for CYLINDER3D_MAKEPROFILE)

    if_plot = True  # Whether to plot the results
    # n_sig = len(sigmaVals)
    n_sig = len(all_sigs)

    for scenario in scenarios:
        # if this scenario is a subject, set use_forward_model to be false
        first_let = scenario[0]  # first letter of the scenario
        if (first_let == 'A' or first_let == 'S') and scenario[1:3].isnumeric():
            print('This scenario, ', scenario, ' appears to be for a subject, not a forward model scenario. Skipping.')
            continue

        [surv_vals, electrodes['rpos'], espace] = s_scen.set_scenario(scenario, NELEC)
        elec_midpoint = GRID['z'][-1]/2.0  # electrode array midpoint
        array_base = -(np.arange(NELEC - 1, -1, -1) * espace)
        array_mid = (array_base[0] + array_base[-1])/2.0
        electrodes['zpos'] = (elec_midpoint - array_mid) + array_base

        if not os.path.isdir(FWDOUTPUTDIR):
            os.makedirs(FWDOUTPUTDIR)

        # Additional setup
        RUN_INFO['scenario'] = scenario
        RUN_INFO['run_time'] = datetime.datetime.now()
        COCHLEA['radius'] = np.ones(NELEC) * fp['cylRadius']  # note that '.rpos' is in mm, must fit inside the radius

        # Construct the simParams structure
        sim_params['cochlea'] = COCHLEA
        sim_params['electrodes'] = electrodes
        sim_params['channel'] = CHANNEL
        sim_params['grid'] = GRID
        sim_params['run_info'] = RUN_INFO

        # Example of neural activation at threshold (variants of Goldwyn paper) ; using choices for channel, etc.
        # made above.
        # Activation sensitivity is FIXED, as will the number of neurons required for threshold.
        # Therefore, the final current to achieve threshold will vary according to the simple minimization routine.
        avec = np.arange(0, 1.01, .01)  # create the neuron count to neuron spikes transformation
        rlvec = NEURONS['coef'] * (avec ** 2) + (1 - NEURONS['coef']) * avec
        rlvec = NEURONS['neur_per_clust'] * (rlvec ** NEURONS['power'])
        rlvltable = np.stack((avec, rlvec))  # start with the e-field(s) created above, but remove the current scaling
        NEURONS['rlvl'] = rlvltable

        thr_sim_db = np.empty((NELEC, n_sig))  # Array for threshold data for different stim elecs and sigma values
        thr_sim_db[:] = np.nan

        # Get survival values for all 330 clusters from the 16 values at electrode positions.
        NEURONS['nsurvival'] = surv_full.surv_full(sim_params['electrodes']['zpos'], surv_vals, sim_params['grid']['z'])
        sim_params['neurons'] = NEURONS
        #  Sanity check. Could add other sanity checks here
        if sim_params['grid']['r'] < 1:
            raise ('Ending script. One or more evaluation points are inside cylinder;\
             not appropriate for neural activation.')

        if 'threshold' in all_measures:
            # Determine threshold for each value of sigma
            for i in range(0, n_sig):  # number of sigma values to test
                sim_params['channel']['behavior'] = 0  # Regular threshold search
                sim_params['channel']['sigma'] = all_sigs[i]
                [thr_sim_db[:, i], neuron_vals] = gt.get_thresholds(act_vals, fp, sim_params)

            # Write a csv file
            outfile = FWDOUTPUTDIR + 'FwdModelOutput_thresh_' + scenario + '.csv'
            with open(outfile, mode='w') as data_file:
                data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
                for row in range(0, NELEC):
                    # TODO Need to have all values of sigma saved on this line. can do brute-force for
                    # 0, 1 or 2 or 3 values of sigma. Or there must be a pythonic way
                    if n_sig == 1:
                        data_writer.writerow([row, surv_vals[row], electrodes['rpos'][row], thr_sim_db[row, 0]])
                    elif n_sig == 2:
                        data_writer.writerow([row, surv_vals[row], electrodes['rpos'][row], thr_sim_db[row, 0],
                                          thr_sim_db[row, 1]])
                    elif n_sig == 3:
                        data_writer.writerow([row, surv_vals[row], electrodes['rpos'][row], thr_sim_db[row, 0],
                                              thr_sim_db[row, 1], thr_sim_db[row, 2]])
                    elif n_sig == 4:
                        data_writer.writerow([row, surv_vals[row], electrodes['rpos'][row], thr_sim_db[row, 0],
                                              thr_sim_db[row, 1], thr_sim_db[row, 2], thr_sim_db[row, 3]])
                    else:
                        data_writer.writerow([row, surv_vals[row], electrodes['rpos'][row]])

            data_file.close()

        if 'mp_mcl' in all_measures:  # for M levels
            m_level_db = np.empty(NELEC)  # Array for threshold data for different stim elecs and sigma values
            m_level_db[:] = np.nan
            # for i in range(0, n_behavior):  # number of sigma values to test
            sim_params['channel']['sigma'] = 0  # sigmaVals[i]
            sim_params['channel']['behavior'] = 1  # search for m_level, not threshold
            [m_level_db, neuron_vals] = gt.get_thresholds(act_vals, fp, sim_params)

            # Write a csv file
            outfile = FWDOUTPUTDIR + 'FwdModelOutput_mp_mcl_' + scenario + '.csv'
            with open(outfile, mode='w') as data_file:
                data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
                for row in range(0, NELEC):
                    data_writer.writerow([row, surv_vals[row], electrodes['rpos'][row], m_level_db[row]])
            data_file.close()

        if 'ecap' in all_measures:
            ec_thresh = 1000  # neurons
            nstim = 500
            # need to reset sigma to 0.0
            sim_params['channel']['sigma'] = 0.0
            ecap = c_ecap.calc_ecap(act_vals, fp, sim_params, ec_thresh, thr_sim_db[:, i], nstim)

            # Save ecap
            np.save(FWDOUTPUTDIR + 'ecap_' + scenario, ecap)
            # Now save ecap results as CSV
            ecap_csv_name = FWDOUTPUTDIR + 'Dist1_5_high_density.csv'
            with open(ecap_csv_name, mode='w') as data_file:
                data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
                data_writer.writerow(ecap[7, :, 0])  # 7 is for middle electrode
                data_writer.writerow(ecap[7, :, 1])  # 7 is for middle electrode
            data_file.close()

        # Save simParams
        spname = FWDOUTPUTDIR + 'simParams' + scenario
        with open(spname + '.pickle', 'wb') as f:
            pickle.dump(sim_params, f, pickle.HIGHEST_PROTOCOL)
        print('saved: ', spname + '.pickle')
        # Note that this is saving only the last simParams structure from the loops on sigma and in get_thresholds.

        # Plot the results, if desired
        ## TODO -- generlaize -- make plots for all types of measures
        if if_plot:
            fig1, ax1 = plt.subplots()
            ax1.plot(np.arange(0, NELEC) + 1, thr_sim_db, marker='o')
            title_text = 'Threshold ' + scenario
            ax1.set(xlabel='Electrode number', ylabel='Threshold (dB)', title=title_text)
            plt.show()

    # Save PDF, if desired
    #        legend([simSurv, targetSurv], 'sim', 'target', 'Location', 'north');
    #        print('-dpdf', '-painters', '-bestfit', 'epsFig.pdf');
    #        movefile('epsFig.pdf', [FWDOUTPUTDIR scenario '_thresh.pdf']);


if __name__ == '__main__':
    fwd_model_5('main')  # alternatives are 'main' and "survey'
