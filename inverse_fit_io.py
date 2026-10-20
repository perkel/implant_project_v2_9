# inverse_fit_io.py
# I/O helpers for inverse_model_multi2.py: loading 2D threshold-map CSVs, loading per-scenario/
# subject threshold data, and saving per-scenario and summary fit results.
#
# Split out of InverseModelMulti.py (which is left untouched) to shrink that file's ~800-line
# inverse_model_multi() down to its core fitting logic. Behavior matches the original inline code,
# with two exceptions noted inline below: a genuine NameError bug in the (currently unreachable)
# ecap-loading branch is fixed, and a couple of no-op `with open(...) as f: ... f.close()` calls are
# simplified to plain `with` blocks (the explicit close() was always redundant).

import csv
import numpy as np

import set_scenario as s_scen
import load_fwd_csv_data as lcsv
from subject_data_xlsx import subj_thr_data_xlsx
from common_params import *


def is_scenario(scen):  # test whether this is an artificial scenario or subject
    # if this scenario is a subject, set use_forward_model to be false
    if (scen[0] == 'A' or scen[0] == 'S') and scen[1:3].isnumeric():
        return False  # it's a subject
    else:
        return True


def load_2d_grid_csv(datafile):
    """Load one 2D grid CSV file (as written by fwd_model5_2d.py) into a numpy array."""
    with open(datafile, newline='') as csvfile:
        rows = list(csv.reader(csvfile, delimiter=','))
    return np.array(rows, dtype=float)


def load_2d_threshold_maps(sig_vals, es_text):
    """Load whichever 2D threshold/measure maps are enabled in fit_measures for this electrode
    spacing. Returns a dict with keys among 'mono_thr', 'tripol_thr', 'mp_mcl_thr', 'ecap_thr' --
    entries for disabled measures are omitted, exactly as the original code left the corresponding
    local variable undefined when its fit_measures weight was 0.

    Note: mp_mcl_thr and ecap_thr are loaded here (as the original did) but currently have no
    downstream consumer anywhere in inverse_model_multi -- only mono_thr/tripol_thr are read again
    after loading (for the subject-offset correction and the initial-guess contours).
    """
    maps = {}
    if fit_measures['mp'] > 0.0:  # 'mp' is the lower of the two sigma values (e.g. [0.5, 0.9])
        sigtext = f"{sig_vals[0]:.1f}".replace(".", "_")
        datafile = FWDOUTPUTDIR + '2D_' + sigtext + '_' + STD_TEXT + es_text + '.csv'
        maps['mono_thr'] = load_2d_grid_csv(datafile)

    if fit_measures['tp'] > 0.0:
        sigtext = f"{sig_vals[1]:.1f}".replace(".", "_")
        datafile = FWDOUTPUTDIR + '2D_' + sigtext + '_' + STD_TEXT + es_text + '.csv'
        maps['tripol_thr'] = load_2d_grid_csv(datafile)

    if fit_measures['mp_mcl'] > 0.0:
        datafile = FWDOUTPUTDIR + "MP_mcl_2D_" + STD_TEXT + es_text + ".csv"
        maps['mp_mcl_thr'] = load_2d_grid_csv(datafile)

    if fit_measures['ecap'] > 0.0:
        datafile = FWDOUTPUTDIR + "ecap_2D_" + STD_TEXT + es_text + ".csv"
        # Bug fix vs. the original: it parsed rows into an undefined 'ecap_thr' variable here
        # (it had allocated 'mp_mcl_thr' again by copy-paste instead), guaranteeing a NameError if
        # this branch ever ran. fit_measures['ecap'] defaults to 0.0, so it never has.
        maps['ecap_thr'] = load_2d_grid_csv(datafile)

    return maps


def load_scenario_or_subject_data(scen, scenario, thresh_summary_full):
    """Load survival/rpos/espace/sigma values for one scenario or subject, then finalize thr_data:
    for a synthetic scenario, load the forward model's own per-scenario threshold CSV; for a real
    subject, apply a monopolar/tripolar offset correction against the 2D maps. Mutates
    thresh_summary_full[scen, 0, :, :] in place, and (for scenarios) electrodes['rpos'] in place,
    exactly as the original inline code did.

    Returns: use_fwd_model, sig_vals, espace, thr_data, subject, mono_thr, tripol_thr,
    overall_offset_db (np.nan for scenarios, where no offset correction is applied).
    """
    use_fwd_model = is_scenario(scenario)
    sim_params['run_info']['scenario'] = scenario
    subject = None
    thr_data = None

    if use_fwd_model:
        surv_vals, electrodes['rpos'], espace = s_scen.set_scenario(scenario, NELEC)
        sig_vals = sigma_vals
    else:
        subject = scenario
        # Thresholds come from the master workbook (subject_data_xlsx.py): the two most recent
        # measurements at different sigmas, lower sigma = monopolar, higher = tripolar.
        retval = subj_thr_data_xlsx(subject)
        thr_data = {'thrmp_db': retval[0], 'thrmp': [], 'thrtp_db': retval[1], 'thrtp': [], 'thrtp_sigma': TP_SIGMA}
        sig_vals = retval[2]
        espace = retval[3]

    if espace == 0.85:
        e_txt = '085'
    elif espace == 1.1:
        e_txt = '110'
    else:
        e_txt = 'xxx'
    es_text = '_espace_' + e_txt

    maps = load_2d_threshold_maps(sig_vals, es_text)
    mono_thr = maps.get('mono_thr')
    tripol_thr = maps.get('tripol_thr')

    overall_offset_db = np.nan
    if use_fwd_model:
        if fit_measures['mp'] > 0.0:  # get mp data
            csv_file = FWDOUTPUTDIR + 'FwdModelOutput_thresh_' + scenario + '.csv'
            [thr_data, ct_data] = lcsv.load_fwd_csv_data(csv_file, 'mp')
            subject = []
            thresh_summary_full[scen, 0, 0, :] = thr_data['thrmp_db']

        if fit_measures['tp'] > 0.0:  # get tp data
            csv_file = FWDOUTPUTDIR + 'FwdModelOutput_thresh_' + scenario + '.csv'
            [thr_data, ct_data] = lcsv.load_fwd_csv_data(csv_file, 'tp')
            subject = []
            thresh_summary_full[scen, 0, 1, :] = thr_data['thrtp_db']

        if fit_measures['mp_mcl'] > 0.0:
            csv_file = FWDOUTPUTDIR + 'FwdModelOutput_mp_mcl_' + scenario + '.csv'
            [thr_data_mcl, ct_data] = lcsv.load_fwd_csv_data(csv_file, 'mp_mcl')
            thr_data['mp_mcl_db'] = np.array(thr_data_mcl['mp_mcl_db'])

        if fit_measures['ecap'] > 0.0:
            csv_file = FWDOUTPUTDIR + 'FwdModelOutput_mp_mcp_' + scenario + '.csv'
            [thr_data_ecap, ct_data] = lcsv.load_fwd_csv_data(csv_file, 'ecap')
            # TODO need to incorporate

    else:  # use threshold data from a subject
        thr_data['thrtp_db'] = np.insert(thr_data['thrtp_db'], 0, np.nan)  # put NaNs at ends of array
        thr_data['thrtp_db'] = np.append(thr_data['thrtp_db'], np.nan)

        # Calculate offset to get closer to what the model can produce
        mp_offset_db = np.nanmean(thr_data['thrmp_db']) - np.nanmean(mono_thr)
        tp_offset_db = np.nanmean(thr_data['thrtp_db']) - np.nanmean(tripol_thr)
        overall_offset_db = np.mean([mp_offset_db, tp_offset_db])
        thr_data['thrmp_db'] -= overall_offset_db
        thr_data['thrtp_db'] -= overall_offset_db
        thresh_summary_full[scen, 0, 0, :] = thr_data['thrmp_db']
        thresh_summary_full[scen, 0, 1, :] = thr_data['thrtp_db']

    return use_fwd_model, sig_vals, espace, thr_data, subject, mono_thr, tripol_thr, overall_offset_db


def save_scenario_results(scen, scenario, subject, use_fwd_model, fit_mode, thr_data, thrsimmp, thrsimtp,
                          fitrposvals, fitsurvvals, sig_vals, initvec, overall_offset_db, fit_results,
                          if_plot, if_save_npy, unsupervised):
    """Write the per-scenario CSV and .npy result files, and (if if_plot) trigger the summary plot.
    fit_results is the dict returned by inverse_model_multi2.compute_fit_errors()."""
    import plot_inverse_results  # imported here to avoid a module-load-time circular import

    thrsim = fit_results['thrsim']
    thrtargs = fit_results['thrtargs']
    ct_vals = fit_results['ct_vals']
    survvals = fit_results['survvals']
    rposvals = fit_results['rposvals']
    rposerrs = fit_results['rposerrs']
    survivalerrs = fit_results['survivalerrs']
    rpos_err_metric = fit_results['rpos_err_metric']

    if use_fwd_model:
        save_file_name = INVOUTPUTDIR + scenario + '_fitResults_' + fit_mode + '.csv'
    else:
        save_file_name = INVOUTPUTDIR + subject + '_fitResults_' + fit_mode + '.csv'

    with open(save_file_name, mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
        if use_fwd_model:
            header = ['Electrode', 'ThreshMP', 'ThreshTP', 'Fitted ThreshMP', 'Fitted ThreshTP',
                      'Rposition', 'RpositionFit', 'Survival', 'Fitted Survival', 'RposError', 'SurvError',
                      'offset (NAN)']
        else:
            header = ['Electrode', 'ThreshMP', 'ThreshTP', 'Fitted ThreshMP', 'Fitted ThreshTP',
                      'CT_position', 'RpositionFit', 'Survival (Nan)', 'Fitted survival', 'RposError',
                      'SurvError (Nan)', 'offset']
        data_writer.writerow(header)

        for row in range(NELEC):
            t1 = row
            t2 = thr_data['thrmp_db'][row]
            t3 = thr_data['thrtp_db'][row]
            t4 = thrsimmp[0][row]
            t5 = thrsimtp[0][row]
            if use_fwd_model:
                t6 = rposvals[row]
            else:
                t6 = ct_vals[row] if np.any(ct_vals) else np.nan
            t7 = fitrposvals[row]
            t8 = survvals[row] if use_fwd_model else np.nan
            t9 = fitsurvvals[row]
            t10 = np.nan if row in (0, NELEC - 1) else rposerrs[row - 1]  # with tp_extend, only has 14 values
            if use_fwd_model:
                # Note: the original only reassigned t11/t12 to nan at the two end rows and left
                # t11 (interior rows) as survivalerrs[row-1]; t12 was never reassigned for interior
                # rows, so it kept whatever value it last held (nan, from row 0) -- i.e. t12 was
                # always nan here in practice. Written directly as nan below for the same result.
                t11 = np.nan if row in (0, NELEC - 1) else survivalerrs[row - 1]
                t12 = np.nan
            else:
                t11 = np.nan
                t12 = overall_offset_db

            data_writer.writerow([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12])

        thr_err_mp = np.abs(np.subtract(thrsim[0][0][:], thrtargs[0][0][:]))
        ovrl_mean_mp_err = np.mean(thr_err_mp)
        thr_err_tp = np.abs(np.subtract(thrsim[1][0][:], thrtargs[1][0][:]))
        ovrl_mean_tp_err = np.mean(thr_err_tp)

        data_writer.writerow(" ")  # blank line
        data_writer.writerow(['Mean: ', ' ', 'MP_err: ', ovrl_mean_mp_err, ' TP_err: ', ovrl_mean_tp_err])

    # Save values in npy format. overall_offset_db (dB subtracted from the measured subject thresholds
    # before fitting; nan for scenarios) is always the LAST element so earlier positions are unchanged.
    save_file_name = INVOUTPUTDIR + scenario + '_fitResults_' + fit_mode + '.npy'
    if use_fwd_model:
        np.save(save_file_name,
                np.array([sig_vals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals],
                          rposerrs, rpos_err_metric, survivalerrs, overall_offset_db], dtype=object))
    else:
        np.save(save_file_name,
                np.array([sig_vals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals],
                          rposerrs, rpos_err_metric, survivalerrs, ct_vals, overall_offset_db], dtype=object))

    np.save(INVOUTPUTDIR + scenario + '_simParams_inv', sim_params)

    if if_plot:
        txt_string = scenario if use_fwd_model else subject
        plot_inverse_results.plot_inverse_results(use_fwd_model, txt_string, unsupervised, fit_mode)
    if not if_save_npy:
        import os
        os.remove(save_file_name)
        os.remove(INVOUTPUTDIR + scenario + '_simParams_inv.npy')


def save_summary_results(scenarios, thresh_err_summary, rpos_err_summary, surv_err_summary,
                         dist_corr, dist_corr_p, thresh_summary_full, rpos_summary, if_save_npy):
    """Write the across-scenario summary CSV (and, if if_save_npy, the summary .npy)."""
    num_scen = len(scenarios)
    print('saving summary data in CSV form')
    summary_file_name = INVOUTPUTDIR + 'summary_inverse_fit_results.csv'
    with open(summary_file_name, mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
        header = ['Scenario/subject', 'Mean MP Threshold Error', 'Mean TP Threshold Error', 'Position Error',
                  'Density error', 'Dist correlation', 'Dist corr p']
        data_writer.writerow(header)
        avg_rpos_err = 0.0
        for row in range(num_scen):
            if is_scenario(scenarios[row]):
                data_writer.writerow([scenarios[row], '%.4f' % float(thresh_err_summary[row, 0]),
                                      '%.4f' % float(thresh_err_summary[row, 1]),
                                      '%.4f' % float(rpos_err_summary[row]),
                                      '%.4f' % float(surv_err_summary[row]),
                                      '%.4f' % float(dist_corr[row]),
                                      '%.5f' % float(dist_corr_p[row])])
            else:
                data_writer.writerow([scenarios[row], '%.4f' % float(thresh_err_summary[row, 0]),
                                      '%.4f' % float(thresh_err_summary[row, 1]),
                                      '%.4f' % float(rpos_err_summary[row]), '%.4f' % np.nan,
                                      '%.4f' % float(dist_corr[row]), '%.5f' % float(dist_corr_p[row])])

            avg_rpos_err += rpos_err_summary[row]

        avg_rpos_err /= num_scen  # convert from sum to average

        data_writer.writerow(' ')
        text = 'Summary. Mean rpos errorr: ' + '%.3f' % avg_rpos_err
        data_writer.writerow([text])

    if if_save_npy:
        summary_file_name = INVOUTPUTDIR + 'summary_inverse_fit_results.npy'
        np.save(summary_file_name, np.array([scenarios, [thresh_summary_full], rpos_summary], dtype=object))

    print('Done with saving summary files')
