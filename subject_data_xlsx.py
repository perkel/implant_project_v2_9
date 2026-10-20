# subject_data_xlsx.py
#
# Reads the master subject threshold workbook (A_subject_data_<date>.xlsx) and returns the same
# structure as subject_data.subj_thr_data():
#
#     [thrmp_db, thrtp_db, sig_vals, espace]
#
#   thrmp_db : np.ndarray, 16 values (electrodes 1-16; the end electrodes duplicate their neighbours,
#              as in subject_data.py) -- threshold (dB) at the LOWER of the two sigmas, sig_vals[0]
#   thrtp_db : np.ndarray, 14 values (electrodes 2-15) -- threshold (dB) at the HIGHER sigma, sig_vals[1]
#   sig_vals : [lower sigma (monopolar), higher sigma (tripolar)]
#   espace   : electrode spacing (mm). Not stored in the workbook, so it is a parameter.
#
# Workbook layout: one tab per subject. Each column is one measurement run. Rows: Session, Date
# (yyyymmdd), SQP (e.g. 'SQP090' = sigma 0.9), Run1, Run2, RunPair, then E1..E16 threshold values (dB);
# E1/E16 are empty.
#
# Which data are used. A run is complete only if it has a value for every electrode E2-E15 (E1/E16
# are empty in every run in the workbook, so E2-E15 is all there is; the monopolar array's end
# electrodes are filled by duplicating E2/E15). Incomplete runs are ignored entirely, so a newer run
# with a missing electrode never hides an older complete one. For each sigma, the latest date having
# a complete run is used, regardless of how it compares in recency with other sigmas. The pair is
# chosen by priority, taking the first that the subject has complete data for:
#     1. sigma (0.0, 0.9)      2. sigma (0.5, 0.9)      (PREFERRED_PAIRS)
#     3. otherwise: if there is a complete 0.9 run, 0.9 plus the most recent complete lower sigma;
#        if there is none, the two most recent complete sigmas. (Date ties go to the higher sigma.)
# The lower sigma is always monopolar and the higher tripolar. If a sigma has several complete runs
# on its chosen date, they are averaged per electrode.
# Requires: pandas, openpyxl

import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from common_params import all_sigs, TP_SIGMA

XLSX_FILE = Path(__file__).parent / 'A_subject_data_2026_09_23.xlsx'
# Electrode spacing (mm) isn't stored in the workbook. This mirrors subject_data.py: 0.85 for
# every A subject except A022L/A022R (1.1). Add a subject here to override the default.
DEFAULT_ESPACE = 0.85
ESPACE_BY_SUBJECT = {'A022L': 1.1, 'A022R': 1.1}

# Sigma pairs (monopolar, tripolar) in order of preference; the latest complete data at each sigma is
# used, even when it is older than that of another sigma.
PREFERRED_PAIRS = [(0.0, 0.9), (0.5, 0.9)]

_FIRST_E, _LAST_E = 2, 15  # electrodes with threshold data (tripolar needs both neighbours)


@lru_cache(maxsize=None)
def _read_workbook(xlsx_path):
    return pd.read_excel(xlsx_path, sheet_name=None, header=None)


def xlsx_subject_names(xlsx_path=XLSX_FILE):
    """Names of all subject tabs in the workbook."""
    return list(_read_workbook(str(xlsx_path)))


def _sqp_to_sigma(label):
    """'SQP090' -> 0.9, 'SQP000' -> 0.0, 'SQP050' -> 0.5"""
    return int(str(label).upper().replace('SQP', '')) / 100.0


def _tab_to_frame(raw):
    """Turn one raw tab into (meta, values): meta has one row per run column (date, sigma);
    values is a DataFrame indexed by electrode number (2-15), one column per run."""
    labels = raw[0].astype(str).str.strip()
    row = {lab: i for i, lab in enumerate(labels)}
    meta = pd.DataFrame({
        'date': [int(v) for v in raw.iloc[row['Date'], 1:]],  # yyyymmdd, so numeric order == chronological
        'sigma': [_sqp_to_sigma(v) for v in raw.iloc[row['SQP'], 1:]],
    })
    e_rows = [row[f'E{e}'] for e in range(_FIRST_E, _LAST_E + 1)]
    values = raw.iloc[e_rows, 1:].apply(pd.to_numeric, errors='coerce')
    values.index = range(_FIRST_E, _LAST_E + 1)
    values.columns = range(values.shape[1])
    meta['complete'] = ~values.isna().any().to_numpy()  # a value for every electrode E2-E15
    return meta, values


def _latest_date_per_sigma(meta, allowed_sigmas):
    """{sigma: most recent date with a complete run}, restricted to allowed_sigmas (None = no restriction)."""
    m = meta[meta['complete']]
    if allowed_sigmas is not None:
        # np.array(..., dtype=bool) so an empty m gives an empty row mask (a bare [] would select columns)
        m = m[np.array([any(np.isclose(s, a) for a in allowed_sigmas) for s in m['sigma']], dtype=bool)]
    return m.groupby(m['sigma'].round(3))['date'].max().to_dict()


def _fallback_pair(subj, latest):
    """Pair used when no PREFERRED_PAIRS entry is available: 0.9 plus the most recent lower sigma if
    the subject has a complete 0.9 run, else the two most recent sigmas. Returns (chosen, passed_over)."""
    priority = round(TP_SIGMA, 3)
    if priority in latest:
        candidates = {sg: d for sg, d in latest.items() if sg < priority}
        if not candidates:
            raise ValueError(f'{subj}: has a complete run at sigma {priority} but none at a lower sigma from '
                             f'{all_sigs}. Pass sig_vals=[low, high] to override.')
        by_recency = sorted(candidates.items(), key=lambda kv: (-kv[1], -kv[0]))  # newest first; ties -> higher
        return [(priority, latest[priority]), by_recency[0]], by_recency[1:]
    by_recency = sorted(latest.items(), key=lambda kv: (-kv[1], -kv[0]))
    return by_recency[:2], by_recency[2:]


def select_measurements(subj, xlsx_path=XLSX_FILE, sig_vals=None):
    """The two measurements used for 'subj', as [(sigma, date), (sigma, date)] ordered low sigma first.

    By default: the first of PREFERRED_PAIRS ((0.0, 0.9), then (0.5, 0.9)) that the subject has complete
    data for, each at its latest complete date; otherwise see _fallback_pair. Only sigmas in common_params.all_sigs are considered (the 2D
    forward model has no threshold maps for others). If sig_vals=[a, b] is given, those two sigmas
    are used instead (each at its latest complete date).
    """
    tabs = _read_workbook(str(xlsx_path))
    if subj not in tabs:
        raise ValueError(f'Threshold data are not available for subject {subj} in {Path(xlsx_path).name}')
    meta, _ = _tab_to_frame(tabs[subj])

    if sig_vals is not None:
        latest = _latest_date_per_sigma(meta, None)
        missing = [s for s in sig_vals if round(s, 3) not in latest]
        if missing:
            raise ValueError(f'{subj}: no complete run at sigma {missing} (workbook has complete runs at '
                             f'{sorted(latest)})')
        for s in sig_vals:
            if s not in all_sigs:
                warnings.warn(f'{subj}: sigma {s} is not in common_params.all_sigs; the 2D forward model has '
                              f'no threshold maps for it.')
        chosen = [(round(s, 3), latest[round(s, 3)]) for s in sig_vals]
    else:
        latest = _latest_date_per_sigma(meta, all_sigs)
        if len(latest) < 2:
            raise ValueError(f'{subj}: need measurements at two different sigmas from {all_sigs}, but only found '
                             f'{sorted(latest) or "none"} (workbook has complete runs at sigma '
                             f'{sorted(set(meta.loc[meta["complete"], "sigma"].round(3)))}). '
                             f'Pass sig_vals=[low, high] to override.')
        pair = next((pr for pr in PREFERRED_PAIRS if all(round(sg, 3) in latest for sg in pr)), None)
        if pair is not None:
            chosen = [(round(sg, 3), latest[round(sg, 3)]) for sg in pair]
        else:
            chosen, passed_over = _fallback_pair(subj, latest)
            pick = chosen[1]  # the 'second' sigma, whose selection can hinge on a date tie
            tied = sorted([pick[0]] + [sg for sg, d in passed_over if d == pick[1]])
            if len(tied) > 1:
                warnings.warn(f'{subj}: sigmas {tied} were all measured on {pick[1]}; the tie was broken '
                              f'in favour of the higher sigma ({pick[0]}). Pass sig_vals to choose.')
    return sorted(chosen)  # lower sigma first


def _mean_thresholds(meta, values, sigma, date):
    """Per-electrode mean over the complete runs at 'sigma' on 'date' (no NaNs possible)."""
    cols = meta.index[np.isclose(meta['sigma'], sigma) & (meta['date'] == date) & meta['complete']]
    return values[list(cols)].to_numpy().mean(axis=1)


def subj_thr_data_xlsx(subj, xlsx_path=XLSX_FILE, sig_vals=None, espace=None):
    """Threshold data for subject/tab 'subj', in the format returned by subject_data.subj_thr_data().

    Uses the two most recent complete measurements at different sigmas (see select_measurements); the lower
    sigma is returned as monopolar (thrmp_db) and the higher as tripolar (thrtp_db).
    sig_vals : optional [sigma_a, sigma_b] to force the sigmas used (each at its latest complete date).
    espace   : electrode spacing (mm) to return; by default ESPACE_BY_SUBJECT[subj], else DEFAULT_ESPACE
               (the workbook doesn't store it).
    """
    if espace is None:
        espace = ESPACE_BY_SUBJECT.get(subj, DEFAULT_ESPACE)
    chosen = select_measurements(subj, xlsx_path, sig_vals)
    meta, values = _tab_to_frame(_read_workbook(str(xlsx_path))[subj])
    (lo_sigma, lo_date), (hi_sigma, hi_date) = chosen
    mp = _mean_thresholds(meta, values, lo_sigma, lo_date)
    tp = _mean_thresholds(meta, values, hi_sigma, hi_date)
    thrmp_db = np.concatenate(([mp[0]], mp, [mp[-1]]))  # electrodes 1..16, ends duplicated as in subject_data.py
    return [thrmp_db, tp, [lo_sigma, hi_sigma], espace]


if __name__ == '__main__':
    for name in xlsx_subject_names():
        try:
            (s1, d1), (s2, d2) = select_measurements(name)
            print(f'{name:10} MP sigma {s1:.1f} ({d1})   TP sigma {s2:.1f} ({d2})')
        except ValueError as err:
            print(f'{name:10} -> {str(err)[:110]}')
