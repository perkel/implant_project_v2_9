#  plot_inverse_results.py
#  This script takes the latest (from common_params.py) run and plots the results

import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as stats
from itertools import combinations
from common_params import *  # import common values across all models
import subject_data

def select_electrode_conditions(distance, survival, n_deactivate=4, n_focus=7, max_adjacent=2,
                                    no_focus=(0, 15), deact_thr=0.1, focus_thr=1.0):
    """Electrode selections for three stimulation conditions, from per-electrode fitted distance and
    neuronal survival (16 values each, electrode order).

      A: focus up to n_focus electrodes with the greatest distance; nothing deactivated. If focus_thr
         is given, only electrodes with distance strictly greater than focus_thr can be focused (so
         fewer than n_focus, possibly none, may be chosen).
      B: deactivate up to n_deactivate electrodes with the lowest total survival, with no more than
         max_adjacent of the deactivated electrodes consecutive; no focusing. If deact_thr is given,
         only electrodes with survival strictly below deact_thr can be deactivated (so fewer than
         n_deactivate, possibly none, may be chosen).
      C: deactivate the same electrodes as B, and focus up to n_focus remaining (active) electrodes
         with the greatest distance (same focus_thr rule as A).

    NaN survival -> never deactivated; NaN distance -> never focused. 'Up to': if no valid set of
    n_deactivate exists (NaNs / strict adjacency limit), the largest valid set is used.
    Returns {'A': {'deactivate': idx, 'focus': idx}, 'B': {...}, 'C': {...}} with sorted 0-based
    index arrays (add 1 for electrode numbers)."""
    distance = np.asarray(distance, dtype=float)
    survival = np.asarray(survival, dtype=float)

    def longest_run(idx):
        run = best = 1
        for a, b in zip(idx[:-1], idx[1:]):
            run = run + 1 if b == a + 1 else 1
            best = max(best, run)
        return best


    def top_distance(eligible):
        # never focus on NaN-distance electrodes or on the end electrodes in no_focus
        excluded = np.union1d(np.flatnonzero(np.isnan(distance)), no_focus)
        if focus_thr is not None:  # only electrodes farther than focus_thr are focus candidates
            with np.errstate(invalid='ignore'):
                excluded = np.union1d(excluded, np.flatnonzero(~(distance > focus_thr)))
        eligible = np.setdiff1d(eligible, excluded)
        return np.sort(eligible[np.argsort(distance[eligible])[::-1][:n_focus]])

    # B: lowest-survival set under the adjacency limit (exhaustive search -> true optimum)
    with np.errstate(invalid='ignore'):  # NaN < deact_thr is False, so NaN is never a candidate
        can_deactivate = ~np.isnan(survival) if deact_thr is None else survival < deact_thr
    candidates = np.flatnonzero(can_deactivate)
    deactivate = np.array([], dtype=int)
    for k in range(min(n_deactivate, len(candidates)), 0, -1):
        valid = [c for c in combinations(candidates, k) if longest_run(c) <= max_adjacent]
        if valid:
            deactivate = np.array(sorted(min(valid, key=lambda c: survival[list(c)].sum())))
            break

    none = np.array([], dtype=int)
    all_el = np.arange(len(distance))
    return {'A': {'deactivate': none, 'focus': top_distance(all_el)},
            'B': {'deactivate': deactivate, 'focus': none},
            'C': {'deactivate': deactivate, 'focus': top_distance(np.setdiff1d(all_el, deactivate))}}



def plot_inverse_results(use_fwd_model, this_case, unsupervised,fit_mode):

    # unsupervised: if True, don't show plots
    # Key constants to set
    save_fig = True
    plot_guess = False
    remove_offset = True
    # if True, removes offset value applied to subject data to put values in the range of the 2D map
    plot_ct_uncertainty = False
    ct_vals = []  # to get rid of a warning
    plot_scatter = False
    tp_extend = True
    if_plot_program_recs = True
    deact_thr = 0.4  # only electrodes with fitted survival < deact_thr can be deactivated; None = no limit
    focus_thr = 1.0  # only electrodes with fitted distance (1 - rpos, mm) > focus_thr can be focused; None = no limit

    # Open file and load data
    # The npy file holds a list of results; overall_offset_db (dB subtracted from measured subject
    # thresholds before fitting; nan for scenarios) is the last element. Files written before it was
    # added are one element shorter, so fall back to nan for those.
    if not use_fwd_model:
        ct_vals = subject_data.subj_ct_data(this_case)
    data_filename = INVOUTPUTDIR + this_case + '_fitResults_' + fit_mode+'.npy'
    results = np.load(data_filename, allow_pickle=True)
    n_core = 10 if use_fwd_model else 11
    overall_offset_db = results[n_core] if len(results) > n_core else np.nan
    # sig_vals is the [monopolar, tripolar] sigma pair this run was actually fit with. Use it for the
    # title -- common_params.sigma_vals is only the global default and doesn't reflect the sigmas
    # chosen per subject (e.g. from subject_data_xlsx.py).
    if use_fwd_model:
        [sig_vals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals],
         _, rpos_err_metric, survivalerrs] = results[:n_core]
    else:
        [sig_vals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals],
         _, rpos_err_metric, _, ct_vals] = results[:n_core]

    # Make plots
    xvals = np.arange(0, NELEC) + 1
    l_e = NELEC - 1  # last electrode to plot

    # All on one plot
    figrows = 3
    figcols = 1
    fig_consol, axs = plt.subplots(figrows, figcols)
    fig_consol.set_figheight(9)
    fig_consol.set_figwidth(7.5)

    if remove_offset:
        offset = overall_offset_db
    else:
        offset = 0.0

    if not tp_extend:
        axs[0].plot(xvals[1:l_e] + 0.1, thrsim[0][0][1:l_e] + offset, marker='o', color='lightblue', label='fit MP')
        axs[0].plot(xvals[1:l_e] - 0.1, thrtargs[0][0][1:l_e] + offset, marker='o', color='blue', label='measured MP')
        axs[0].plot(xvals[1:l_e] + 0.1, thrsim[1][0][1:l_e] + offset, marker='o', color='pink', label='fit TP')
        axs[0].plot(xvals[1:l_e] - 0.1, thrtargs[1][0][1:l_e] + offset, marker='o', color='red', label='measured TP')
    else:

        axs[0].plot(xvals + 0.1, thrsim[0][0], marker='o', color='lightblue', label='fit MP')
        axs[0].plot(xvals - 0.1, thrtargs[0][0], marker='o', color='blue', label='measured MP')
        axs[0].plot(xvals[1:l_e] + 0.1, thrsim[1][0][1:l_e], marker='o', color='pink', label='fit TP')
        axs[0].plot(xvals[1:l_e] - 0.1, thrtargs[1][0][1:l_e], marker='o', color='red', label='measured TP')

    mean_thr_err = (np.nanmean(np.abs(np.array(thrsim[0]) - np.array(thrtargs[0]))) +
                    np.nanmean(np.abs(np.array(thrsim[1]) - np.array(thrtargs[1])))) / 2.0

    yl = 'Threshold (dB)'
    if use_fwd_model:
        title_text = 'Known scenario thresholds: ' + this_case + '; mean thr error (dB): ' + '%.2f' % mean_thr_err
    else:
        title_text = ('Subject thresholds: ' + this_case + ' ' + RE_TEXT + ' ' + RI_TEXT + ' ' + TARG_TEXT +
                      '; mean thr error (dB): ' + '%.2f' % mean_thr_err + '\nMP: sigma = %.1f' % sig_vals[0] +
                      ' TP: sigma = %.1f' % sig_vals[1])
    axs[0].set(xlabel='Electrode number', ylabel=yl, title=title_text)
    axs[0].set_xlim(0, 17)
    axs[0].legend(loc='upper right', ncol=2)

    if use_fwd_model:
        [dist_corr, dist_corr_p] = stats.pearsonr(1 - rposvals, 1 - fitrposvals)
    else:
        if np.any(ct_vals):
            [dist_corr, dist_corr_p] = stats.pearsonr(1 - ct_vals, 1 - fitrposvals)
    title_text = 'Fit and actual positions; mean position error (mm): ' + '%.2f' % rpos_err_metric
    if not tp_extend:

        axs[1].plot(xvals[1:l_e] + 0.1, 1 - fitrposvals[1:l_e], marker='o', color='gray', label='fit')
        if use_fwd_model:
            axs[1].plot(xvals[1:l_e] - 0.1, 1 - rposvals[1:l_e], marker='o', color='black', label='actual')
        else:
            if np.any(ct_vals):
                axs[1].plot(xvals[1:l_e] - 0.1, 1 - ct_vals[1:l_e], marker='o', color='black', label='CT estimate')
                if plot_ct_uncertainty:
                    axs[1].fill_between(xvals[1:l_e] - 0.1, (1 - ct_vals[1:l_e]) - ct_uncertainty, (1 - ct_vals[1:l_e]) + ct_uncertainty,
                                        color='black', alpha=0.1)
    else:
        axs[1].plot(xvals + 0.1, 1 - fitrposvals, marker='o', color='gray', label='fit')
        if use_fwd_model:
            axs[1].plot(xvals - 0.1, 1 - rposvals, marker='o', color='black', label='actual')
        else:
            if np.any(ct_vals):
                axs[1].plot(xvals - 0.1, 1 - ct_vals, marker='o', color='black', label='CT estimate')
                if plot_ct_uncertainty:
                    axs[1].fill_between(xvals - 0.1, (1 - ct_vals) - ct_uncertainty, (1 - ct_vals) + ct_uncertainty,
                                        color='black', alpha=0.1)

   # if use_fwd_model:
    #    axs[1].plot(xvals - 0.1, 1 - rposvals, marker='o', color='black', label='actual')
   # else:
    #    if np.any(ct_vals):
     #       axs[1].plot(xvals - 0.1, 1 - ct_vals, marker='o', color='black', label='CT estimate')
      #      if plot_ct_uncertainty:
       #         axs[1].fill_between(xvals - 0.1, (1 - ct_vals) - ct_uncertainty, (1 - ct_vals) + ct_uncertainty,
        #                            color='black', alpha=0.1)

    if plot_guess:
        axs[1].plot(xvals, 1 - initvec[0:NELEC], marker='o', color='purple', label='initial guess')

    axs[1].set(xlabel='Electrode number', ylabel='Electrode distance (mm)', title=title_text)
    axs[1].set_xlim(0, 17)
    axs[1].set_ylim(-0.5, 2)
    axs[1].legend()

    # Now using a horizontal line rather than a dot to show that survival value is for a range of space.
    title_text = 'Fit survival values'
    if not tp_extend:
        if use_fwd_model:
            # axs[2].plot(xvals[1:l_e], fitsurvvals[1:l_e], marker='o', color='red', label='fit')
            for el in range(NELEC-2):
                axs[2].plot([xvals[el+1] - 0.45, xvals[el+1] + 0.45], [fitsurvvals[el+1], fitsurvvals[el+1]],
                            color='black', label='fit')
            axs[2].plot(xvals[1:l_e], survvals[1:l_e], marker='o', color='green', label='desired')
            if plot_guess:
                axs[2].plot(xvals[1:l_e], initvec[NELEC-2:], marker='o', color='purple', label='initial guess')

        else:
            # axs[2].plot(xvals[1:l_e], fitsurvvals[1:l_e], marker='o', color='black', label='modeled')
            for el in range(NELEC - 2):
                if el == 0:
                    axs[2].plot([xvals[el+1] - 0.45, xvals[el+1] + 0.45], [fitsurvvals[el], fitsurvvals[el]],
                            color='black', label='fit')
                else:
                    axs[2].plot([xvals[el+1] - 0.45, xvals[el+1] + 0.45], [fitsurvvals[el], fitsurvvals[el]],
                            color='black')


    else:

        if use_fwd_model:
            # axs[2].plot(xvals, fitsurvvals, marker='o', color='red', label='fit')
            for el in range(NELEC):
                axs[2].plot([xvals[el] - 0.45, xvals[el] + 0.45], [survvals[el], survvals[el]], marker='o', color='green', label='desired')
            if plot_guess:
                axs[2].plot(xvals, initvec[NELEC:], marker='o', color='purple', label='initial guess')

        else:
            for el in range(NELEC):
                axs[2].plot([xvals[el]-0.45, xvals[el]+0.45], [fitsurvvals[el], fitsurvvals[el]], color='black')

    axs[2].set(xlabel='Electrode number', ylabel='Fractional neuronal density', title=title_text)
    axs[2].set_xlim(0, 17)
    axs[2].set_ylim(-0.02, 1.02)
    axs[2].legend()
    fig_consol.tight_layout()

    # -- could add plots of error (difference between desired/measured and fitted values)


    if not use_fwd_model:
        if if_plot_program_recs:
            # find and indicate electrodes selected for deactivation
            # deact_recs = select_low_survival(fitsurvvals) + 1  # + 1 because the first 1 available is #1
            # deactivate, focus_comb, focus_only = select_electrodes(fitsurvvals[1:NELEC-1], 1-fitrposvals[1:NELEC-1])

            conditions = select_electrode_conditions(1-fitrposvals, fitsurvvals, deact_thr=deact_thr,
                                                     focus_thr=focus_thr)
            if focus_thr is not None:  # show the focus threshold on the distance panel
                axs[1].axhline(focus_thr, color='blue', linestyle='--', linewidth=1, alpha=0.5,
                               label='focus threshold')
                axs[1].legend()

            deactivate = conditions['B']['deactivate']  # deactivate only
            focus_only = conditions['A']['focus']  # focus only
            focus_comb = conditions['C']['focus']  # combined focusing and eactivation
            print('deactivate recs: ', deactivate[:])
            print('focus recs: ', focus_comb[:])
            print('focus only: ', focus_only[:])
            for idx, elec in enumerate(deactivate):
                # Define the ellipse parameters
                # (center_x, center_y), width, height, rotation_angle (in degrees)
                print('idx, elec: ', idx, elec)
                ellipse = Ellipse(xy=(xvals[elec], fitsurvvals[elec]), width=1.2, height=0.1, linewidth=3, angle=0,
                                  edgecolor='red', facecolor='None', alpha=0.5)

                # 3. Add the patch to the axes
                axs[2].add_patch(ellipse)
            # focus_comb
            for idx, elec in enumerate(focus_comb):
                # axs[1].arrow(elec+1, 0, 0, 0.15, width=0.2, head_width=0.1, facecolor='blue')
                axs[1].annotate('', xy=(xvals[elec], 0.3), xytext=(xvals[elec], 0),
                                arrowprops=dict(arrowstyle="->", color="blue", lw=3))

            # focus only the top 7
            # myf_only = np.argsort(1-fitrposvals[1:NELEC-1], descending=True)[:7] + 1
            for idx, elec in enumerate(focus_only):
                axs[1].plot(xvals[elec]+0.1, 1-fitrposvals[elec], marker='o', color='blue')


            # Save CSV file
            rows = [
                ['Subject:' + this_case],  # a text label line
                ['Electrodes suggested for deactivation'],
                list(deactivate+1),  # numeric row (any length)
                ['Focus threshold distance (mm): ' + str(focus_thr)],
                ['Electrodes selected for focusing (with no deactivation'],
                list(focus_only+1),
                ['Electrodes selected for focusing (with others deactivated'],
                list(focus_comb+1)
            ]

            with open('electrode_selection_' + this_case + '.csv', 'w', newline='') as f:
                csv.writer(f).writerows(rows)

    if save_fig:
        save_file_name = INVOUTPUTDIR + this_case + '_fitResultsFig_' + fit_mode + '.png'
        fig_consol.savefig(save_file_name)

    # test correlation figure
    # if not use_fwd_model:
    #     if np.any(ct_vals):
    #         fig2, ax2 = plt.subplots()
    #         ct_dist = 1 - ct_vals[1:l_e]
    #         fit_dist = 1 - fitrposvals[1:l_e]
    #         fig2 = plt.plot(ct_dist, fit_dist, 'o')
    #         ax2.set(xlabel='CT distance (mm)', ylabel='fit distance (mm)')
    #         ax2.set_xlim([0, 2])
    #         ax2.set_ylim([0, 2])
    #         fit_corr = np.corrcoef(ct_dist, fit_dist)
    if not unsupervised:
        plt.show()

    if plot_scatter:
        # second plot, scatter of electrode position and neuronal survival
        fig3, axs3 = plt.subplots(1, 1)
        axs3.plot(1 - fitrposvals, fitsurvvals, 'o', color='green')
        axs3.set_xlabel('fit distance (mm)')
        axs3.set_ylabel('fit density')
        axs3.set_xlim([0, 2])
        axs3.set_ylim([0, 1])
        # now plot initial guesses
        if not tp_extend:
            axs3.plot(1 - initvec[0:NELEC-2], initvec[NELEC-2:], 'o', color='red')
            for i in range(NELEC-2):
                axs3.plot([1 - fitrposvals[i], 1 - initvec[i]], [fitsurvvals[i], initvec[NELEC-2 + i]], color='black')

        else:
            axs3.plot(1 - initvec[0:NELEC], initvec[NELEC:], 'o', color='red')
            for i in range(NELEC):
                axs3.plot([1 - fitrposvals[i], 1 - initvec[i]], [fitsurvvals[i], initvec[NELEC + i]], color='black')

        title_text = this_case + '  Mean position error (mm): ' + '%.2f' % rpos_err_metric
        fig3.suptitle(title_text, fontsize=14)
        axs3.text(1.75, 0.95, 'guess', color='red')
        axs3.text(1.75, 0.9, 'fit', color='green')
        save_file_name = INVOUTPUTDIR + this_case + '_fitResults_scatter.png'
        fig3.savefig(save_file_name, format='png')
        if not unsupervised:
            plt.show()

if __name__ == '__main__':
    use_fwd_model = False
    # case = scenarios[0]
    case = 'A039L'
    # espace = 0.85
    # case = 'RampRposSGradual80'
    # case = 'Gradual80R00'

    unsupervised = False
    fit_mode='combined'
    plot_inverse_results(use_fwd_model, case, unsupervised,fit_mode)
