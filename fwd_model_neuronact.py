# Forward Model Neuron Activation. 8 October 2026 David J. Perkel
# Calculate # of neurons activated and activation profile for a scenario or subject's electrode distance
# and survival data for a given set of stimuli.


import pickle
import datetime
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

#  These are local to this project
import set_scenario as s_scen
import surv_full
import thr_function as thr_f
import cylinder3d_makeprofile as c3dm
import calc_ecap as c_ecap
from common_params import *
from subject_data_xlsx import subj_thr_data_xlsx
# from obsolete.paper_figures import use_fwd_model

fit_mode = 'combined'

def fwd_model_neuron_act():

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
    # Example of neural activation at threshold (variants of Goldwyn paper) ; using choices for channel, etc.
    # made above.
    # Activation sensitivity is FIXED, as will the number of neurons required for threshold.
    # Therefore, the final current to achieve threshold will vary according to the simple minimization routine.
    avec = np.arange(0, 1.01, .01)  # create the neuron count to neuron spikes transformation
    rlvec = NEURONS['coef'] * (avec ** 2) + (1 - NEURONS['coef']) * avec
    rlvec = NEURONS['neur_per_clust'] * (rlvec ** NEURONS['power'])
    rlvltable = np.stack((avec, rlvec))  # start with the e-field(s) created above, but remove the current scaling
    NEURONS['rlvl'] = rlvltable


    # Construct the simParams structure
    sim_params['electrodes'] = electrodes
    sim_params['channel'] = CHANNEL
    sim_params['grid'] = GRID


    if_plot = True  # Whether to plot the results
    n_sig = len(sigma_vals)
    use_fwd_model = False
    # n_sig = len(all_sigs)

    for scenario in scenarios:
        # Additional setup
        RUN_INFO['scenario'] = scenario
        RUN_INFO['run_time'] = datetime.datetime.now()
        sim_params['run_info'] = RUN_INFO
        COCHLEA['radius'] = np.ones(NELEC) * fp['cylRadius']  # note that '.rpos' is in mm, must fit inside the radius
        sim_params['cochlea'] = COCHLEA

        # if this scenario is a subject, set use_forward_model to be false
        first_let = scenario[0]  # first letter of the scenario
        # TODO note that subsequent code is for A031 only.
        if (first_let == 'A' or first_let == 'S') and scenario[1:3].isnumeric():
            use_fwd_model = False
            print('This scenario, ', scenario, ' appears to be for a subject.')

            # Load electrode distance and neuronal density profile

            # Columns N ('Windowed SGN count) and Q ('dist') of A031_SGN_Data_v2.xlsx; empty cells become NaN
            # custom code for A031
            espace = 1.1
            df = pd.read_excel('A031_SGN_Data_v2.xlsx', usecols='O,Q,R,S', header=0)
            # sgn_1to1 = df.iloc[:, 0].to_numpy(dtype=float)
            sgn_window = df.iloc[:, 0].to_numpy(dtype=float)

            dist = df.iloc[:, 1].to_numpy(dtype=float)
            A031_thr_mp = df.iloc[:, 2].to_numpy(dtype=float)
            A031_thr_tp = df.iloc[:, 3].to_numpy(dtype=float)
            data_filename = INVOUTPUTDIR + scenario + '_fitResults_' + fit_mode + '.npy'
            results = np.load(data_filename, allow_pickle=True)
            n_core = 10 if use_fwd_model else 11
            overall_offset_db = results[n_core] if len(results) > n_core else np.nan
            # sig_vals is the [monopolar, tripolar] sigma pair this run was actually fit with. Use it for the
            # title -- common_params.sigma_vals is only the global default and doesn't reflect the sigmas
            # chosen per subject (e.g. from subject_data_xlsx.py).

            [sig_vals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals],
                 _, rpos_err_metric, _, ct_vals] = results[:n_core]

            # Load stimulus levels for threshold (MP or TP) and/or mcl. Since this is from a subject,
            # get this from the Excel sheet
            # retval = subj_thr_data_xlsx(scenario)
            thr_data = {'thrmp_db': A031_thr_mp, 'thrmp': np.power(10, (A031_thr_mp/20.0)),
                        'thrtp_db': A031_thr_tp, 'thrtp': np.power(10, (A031_thr_tp/20)), 'thrtp_sigma': TP_SIGMA}
            # sig_vals = retval[2]
            # espace = retval[3]
            elec_midpoint = GRID['z'][-1]/2.0  # electrode array midpoint
            array_base = -(np.arange(NELEC - 1, -1, -1) * espace)
            array_mid = (array_base[0] + array_base[-1])/2.0
            electrodes['zpos'] = (elec_midpoint - array_mid) + array_base
            # Get survival values for all 330 clusters from the 16 values at electrode positions.
            NEURONS['nsurvival'] = surv_full.surv_full(sim_params['electrodes']['zpos'], sgn_window[0:16]/200,
                                                       sim_params['grid']['z'])
            sim_params['neurons'] = NEURONS

            # Calculate neuronal activation profile and total

        else:  # set up for an artificial scenario
            use_fwd_model = True



# Now do the main calculation of neuronal activation for each electrode for the stim levels
        # Additional setup
        RUN_INFO['scenario'] = scenario
        RUN_INFO['run_time'] = datetime.datetime.now()
        COCHLEA['radius'] = np.ones(NELEC) * fp['cylRadius']  # note that '.rpos' is in mm, must fit inside the radius

        # Construct the simParams structure
        sim_params['cochlea'] = COCHLEA
        sim_params['electrodes'] = electrodes
        sim_params['electrodes']['rpos'] = 1-dist
        sim_params['channel'] = CHANNEL
        sim_params['grid'] = GRID
        sim_params['run_info'] = RUN_INFO

        ncounts = np.zeros((n_sig, NELEC))
        neural_profile = np.zeros((n_sig, NELEC, n_z))
        active_profile = np.zeros((n_sig, NELEC, n_z))

        # prepare the plot
        fig, axs = plt.subplots(1, 1)
        text = 'Sigma vals: ' + str(sigma_vals[0]) + ' and ' + str(sigma_vals[1])
        fig.suptitle(scenario + text)
        axs.set_ylabel('Neurons activated')

        for i in range(NELEC):
            for idx, this_sig in enumerate(sigma_vals):
                sim_params['channel']['sigma'] = this_sig
                if sigma_vals[0] > 0.0:
                    if i == 0 or i == NELEC - 1:
                        pass
                    else:
                        # get neuron count and neuronal activation for this electrode
                        sim_params['channel']['number'] = i
                        aprofile = c3dm.cylinder3d_makeprofile(act_vals, fp, sim_params)
                        e_field = abs(aprofile)  # uV^2/mm^2
                        ncounts[idx, i], neural_profile[idx, i, :], active_profile[idx, i, :] =(
                            thr_f.thr_function(thr_data['thrmp'][i], e_field, sim_params))
                        if idx == 0:
                            axs.plot(i, ncounts[idx, i], 'o', color='green')
                        else:
                            axs.plot(i, ncounts[idx, i], 'o', color='red')

                else:
                    # get neuron count and neuronal activation for this electrode
                    sim_params['channel']['number'] = i
                    aprofile = c3dm.cylinder3d_makeprofile(act_vals, fp, sim_params)
                    # This is the biophysically correct behavior, to include sidelobes
                    e_field = abs(aprofile)  # uV^2/mm^2
                    ncounts[idx, i], neural_profile[idx, i, :], active_profile[idx, i, :] =(
                        thr_f.thr_function(thr_data['thrmp'][i], e_field, sim_params))


        plt.show()

        # Write a csv file
        # outfile = FWDOUTPUTDIR + 'FwdModelOutput_thresh_' + scenario + '.csv'
        # with open(outfile, mode='w') as data_file:
        #     data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
        #     for row in range(0, NELEC):
        #         # TODO Need to have all values of sigma saved on this line. can do brute-force for
        #         # 0, 1 or 2 or 3 values of sigma. Or there must be a pythonic way
        #         if n_sig == 1:
        #             data_writer.writerow([row, surv_vals[row], electrodes['rpos'][row], thr_sim_db[row, 0]])
        #         elif n_sig == 2:
        #             data_writer.writerow([row, surv_vals[row], electrodes['rpos'][row], thr_sim_db[row, 0],
        #                               thr_sim_db[row, 1]])
        #         elif n_sig == 3:
        #             data_writer.writerow([row, surv_vals[row], electrodes['rpos'][row], thr_sim_db[row, 0],
        #                                   thr_sim_db[row, 1], thr_sim_db[row, 2]])
        #         elif n_sig == 4:
        #             data_writer.writerow([row, surv_vals[row], electrodes['rpos'][row], thr_sim_db[row, 0],
        #                                   thr_sim_db[row, 1], thr_sim_db[row, 2], thr_sim_db[row, 3]])
        #         else:
        #             data_writer.writerow([row, surv_vals[row], electrodes['rpos'][row]])
        #
        # data_file.close()



            # save the data



        # Save simParams
        spname = FWDOUTPUTDIR + 'simParams' + scenario
        with open(spname + '.pickle', 'wb') as f:
            pickle.dump(sim_params, f, pickle.HIGHEST_PROTOCOL)
        print('saved: ', spname + '.pickle')
        # Note that this is saving only the last simParams structure from the loops on sigma and in get_thresholds.

        # Plot the results, if desired
        ## TODO -- generalize -- make plots for all types of measures
        # if if_plot:
        #     fig1, ax1 = plt.subplots()
        #     ax1.plot(np.arange(0, NELEC) + 1, thr_sim_db, marker='o')
        #     title_text = 'Threshold ' + scenario
        #     ax1.set(xlabel='Electrode number', ylabel='Threshold (dB)', title=title_text)
        #     plt.show()



if __name__ == '__main__':
    fwd_model_neuron_act()
