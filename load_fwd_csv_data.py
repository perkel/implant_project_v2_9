# load_fwd_csv_data.py
#  Utility function to load threshold, CT, and potentially other data from a
#  spreadsheet.
#  Translated to Python 3 by David Perkel December 2020

import csv
import numpy as np
from common_params import *


def _sigma_column(target_sigma, tol=1e-9):
    """Column index (into the per-row values fwd_model_5.py writes, starting at column 3) that
    holds the threshold for 'target_sigma'. fwd_model_5.py always sweeps every value in all_sigs
    (not just sigma_vals) and writes them in that order, so the right column depends on where
    target_sigma falls within all_sigs -- it is NOT a fixed index. Using a fixed column (as this
    function used to) silently reads the wrong sigma's threshold whenever sigma_vals doesn't
    happen to match all_sigs[0]/all_sigs[-1], e.g. after sigma_vals[0] was changed from 0.0 to 0.5
    without fwd_model_5.py's column layout changing to match.
    """
    matches = [k for k, sv in enumerate(all_sigs) if abs(sv - target_sigma) < tol]
    if not matches:
        raise ValueError(f"sigma value {target_sigma} (from sigma_vals) is not one of the swept "
                         f"all_sigs values {all_sigs} -- fwd_model_5.py never computed a threshold "
                         f"for it, so there's no column in the CSV to read it from.")
    return 3 + matches[0]


def load_fwd_csv_data(loadfile, mode):

    thr_data = {'thrmp_db': np.zeros(NELEC), 'thrmp': np.zeros(NELEC), 'thrtp_db': np.zeros(NELEC),
                'thrtp': np.zeros(NELEC), 'thrtp_sigma': TP_SIGMA, 'mp_mcl_db': np.zeros(NELEC)}
    ct_data = {'stdiameter': [], 'scala': [], 'elecdist': [], 'espace': 1.1, 'type': [], 'insrt_base': [],
               'insert_apex': []}

    radius = 1.0

    # Load the data
    if mode == 'mp':
        # file format: columns 0-2 are row/survival/rpos, then one column per all_sigs entry in
        # order (see fwd_model_5.py). thr_mp_col/thr_tp_col pick out whichever of those columns
        # actually correspond to sigma_vals[0]/sigma_vals[1] -- not necessarily columns 3 and 5.
        thr_mp_col = _sigma_column(sigma_vals[0])
        thisrow = 0
        with open(loadfile, mode='r') as data_file:
            data_reader = csv.reader(data_file, delimiter=',', quotechar='"')
            for row in data_reader:
                if row[thr_mp_col] == 'nan':
                    thr_data['thrmp_db'][thisrow] = np.nan
                else:
                    aaa = row[thr_mp_col]
                    bbb = np.fromstring(aaa, sep=' ')
                    # ccc = float(bbb)
                    thr_data['thrmp_db'][thisrow] = float(bbb[0])

                thr_data['thrmp'][thisrow] = np.power(10.0, (thr_data['thrmp_db'][thisrow] / 20))

                thisrow += 1


    if mode == 'tp':
        # file format: columns 0-2 are row/survival/rpos, then one column per all_sigs entry in
        # order (see fwd_model_5.py). thr_mp_col/thr_tp_col pick out whichever of those columns
        # actually correspond to sigma_vals[0]/sigma_vals[1] -- not necessarily columns 3 and 5.

        thr_mp_col = _sigma_column(sigma_vals[0])
        thr_tp_col = _sigma_column(sigma_vals[1])
        thisrow = 0
        with open(loadfile, mode='r') as data_file:
            data_reader = csv.reader(data_file, delimiter=',', quotechar='"')
            for row in data_reader:
                if row[thr_tp_col] == 'nan':
                    thr_data['thrtp_db'][thisrow] = np.nan
                else:
                    aaa = row[thr_tp_col]
                    bbb = np.fromstring(aaa, sep=' ')
                    thr_data['thrtp_db'][thisrow] = float(bbb[0])

                thr_data['thrtp'][thisrow] = np.power(10.0, (thr_data['thrtp_db'][-1] / 20))
                thisrow += 1

    elif mode == 'mp_mcl':
        mp_mcl_col = 3
        thisrow = 0
        with open(loadfile, mode='r') as data_file:
            data_reader = csv.reader(data_file, delimiter=',', quotechar='"')
            for row in data_reader:
                if row[mp_mcl_col] == 'nan':
                    thr_data['mp_mcl_db'].append(np.nan)
                else:
                    aaa = row[mp_mcl_col]
                    bbb = np.fromstring(aaa, sep=' ')
                    thr_data['mp_mcl_db'][thisrow] = float(bbb[0])
                thisrow += 1
    elif mode == 'ecap':
        pass

    n_elec = len(thr_data['thrmp_db'])
    ct_data['stdiameter'] = radius * 2.0 * (np.zeros(n_elec) + 1.0)
    return [thr_data, ct_data]
