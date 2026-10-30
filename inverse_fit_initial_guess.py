# inverse_fit_initial_guess.py
# Initial-guess estimation for the 'combined' fit mode in inverse_model_multi2.py: for each
# electrode, find a starting (rpos, survival) guess by intersecting the monopolar and tripolar
# threshold contours from the 2D forward-model maps.
#
# Split out of InverseModelMulti.py (which is left untouched) as a self-contained sub-algorithm --
# behavior is identical to the original inline code.

import matplotlib.pyplot as plt
from contourpy import contour_generator
from scipy.interpolate import RegularGridInterpolator
from common_params import *


def closest_midpoint(x1, y1, x2, y2, rpos_scale=0.5):
    """Midpoint of the closest pair of points between two curves, in (rpos, survival) space. Used
    as the initial guess when the MP and TP contours don't cross.

    rpos differences are scaled by rpos_scale (0.5, as in the original find_closest) because rpos
    spans about twice the range of survival (-0.95..0.95 vs 0.04..0.96), so this roughly balances
    the two axes. NaN points (the gaps between contour pieces from join_segments) are ignored.
    Returns (rpos, survival)."""
    x1, y1, x2, y2 = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (x1, y1, x2, y2))
    # distance between every point on curve 1 (rows) and every point on curve 2 (columns)
    d = np.sqrt(((x1[:, None] - x2[None, :]) * rpos_scale) ** 2 + (y1[:, None] - y2[None, :]) ** 2)
    if np.all(np.isnan(d)):
        raise ValueError('closest_midpoint: no valid points on one of the contours')
    i, j = np.unravel_index(np.nanargmin(d), d.shape)
    return (x1[i] + x2[j]) / 2.0, (y1[i] + y2[j]) / 2.0


def contour_lines(z_map, level):
    """Contour lines of a 2D threshold map at 'level', computed with contourpy (the library matplotlib
    itself uses for ax.contour), so no figure is created. Returns a list of (N, 2) arrays of
    (rpos, survival) points -- the same as matplotlib's ContourSet.allsegs[0]. Empty if 'level' is
    outside the range of the map. Rows 0-1 of the map (lowest survival values) are skipped, as before."""
    gen = contour_generator(x=rpos_grid_vals, y=surv_grid_vals[2:], z=z_map[2:, :])
    return [seg for seg in gen.lines(level) if len(seg) > 1]


def tp_crossings_on_mp(mp_segs, tp_interp, tptarg):
    """Points where the TP threshold contour crosses the MP threshold contour, found without
    intersecting two curves: walk along each piece of the MP contour, evaluate (TP map - TP target)
    at every point, and wherever that changes sign, linearly interpolate the crossing point within
    that step. Returns (rpos, survival) arrays of the crossings (empty if there are none)."""
    xs, ys = [], []
    for seg in mp_segs:
        d = tp_interp(seg[:, ::-1]) - tptarg  # TP error at each MP-contour point; interpolator wants (surv, rpos)
        k = np.flatnonzero((d[:-1] <= 0) != (d[1:] <= 0))  # steps where the sign changes
        t = d[k] / (d[k] - d[k + 1])  # fraction of the way along each step where d == 0
        pts = seg[k] + t[:, None] * (seg[k + 1] - seg[k])
        pts = pts[np.all(np.isfinite(pts), axis=1)]  # ignore steps that touch NaN map values
        xs.append(pts[:, 0])
        ys.append(pts[:, 1])
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def join_segments(segs):
    """Join contour pieces into single x and y arrays, separated by NaN (used by closest_midpoint when
    the contours don't cross)."""
    nan_row = np.full((1, 2), np.nan)
    joined = np.concatenate([part for seg in segs for part in (seg, nan_row)][:-1])
    return joined[:, 0], joined[:, 1]


def estimate_initial_guesses(thresh_summary_full, scen, mono_thr, tripol_thr, sig_vals,
                             if_plot_guess_contours, use_fwd_model):
    """For fit_mode == 'combined': for each electrode (except the two end ones, which have no
    tripolar target), find an initial (rpos, survival) guess where the monopolar and tripolar
    threshold contours from the 2D forward-model maps cross (found by evaluating the tripolar map
    along the monopolar contour; see tp_crossings_on_mp). Falls back to a closest-point
    heuristic when the contours don't intersect, and to the mean of all intersections when there's
    more than one. If a target threshold is outside the range of its 2D map (so there is no contour
    at all), the previous electrode's guess is used (or rpos 0.0, survival 0.5 for the first one).
    Returns (fitrposvals, fitsurvvals), both length NELEC, with the end electrodes copied from their
    neighbors.

    Contours are computed with contourpy, not matplotlib, so no figures are created unless
    if_plot_guess_contours is True. All pieces of each contour are used, not just the first.
    """
    nsols = np.zeros(NELEC)  # number of solutions found from the 2D maps
    fitrposvals = np.zeros(NELEC)
    fitsurvvals = np.zeros(NELEC)
    titletext = ''
    # TP map as a function of (survival, rpos), for evaluating it at points along the MP contour
    tp_interp = RegularGridInterpolator((surv_grid_vals[2:], rpos_grid_vals), tripol_thr[2:, :],
                                        bounds_error=False, fill_value=None)

    for i in range(1, NELEC - 1):  # Fit params for each electrode (except ends, where there is no tripol value)
        mptarg = thresh_summary_full[scen, 0, 0, :][i]
        tptarg = thresh_summary_full[scen, 0, 1, :][i]
        titletext = ("sigvals: " + f"{sig_vals[0]: .1f}" + ' and ' + f"{sig_vals[1]: .1f}" +
                     " thr: " + f"{mptarg: .1f}" + ' and ' + f"{tptarg: .1f}")

        mp_segs = contour_lines(mono_thr, mptarg)
        tp_segs = contour_lines(tripol_thr, tptarg)

        if if_plot_guess_contours:
            fig4, ax5 = plt.subplots()
            for seg in mp_segs:
                ax5.plot(seg[:, 0], seg[:, 1], color='green')
            for seg in tp_segs:
                ax5.plot(seg[:, 0], seg[:, 1], color='red')
            ax5.set_xlabel('Rpos (mm)')
            ax5.set_ylabel('Survival fraction')
            ax5.set_title(titletext)

        if not mp_segs or not tp_segs:
            # A target threshold lies outside the range of its 2D map, so there is no contour to
            # intersect (the old matplotlib-based code raised an IndexError here).
            which = ' and '.join(name for name, segs in (('MP', mp_segs), ('TP', tp_segs)) if not segs)
            if i > 1:
                rp_guess, sv_guess = fitrposvals[i - 1], fitsurvvals[i - 1]
            else:
                rp_guess, sv_guess = 0.0, 0.5
            print('electrode: ', i, ';  no ', which, ' contour (target outside the 2D map: MP ', mptarg,
                  ', TP ', tptarg, '). Using (position, survival): ', rp_guess, sv_guess)
        else:
            mpx, mpy = join_segments(mp_segs)
            tpx, tpy = join_segments(tp_segs)

            x, y = tp_crossings_on_mp(mp_segs, tp_interp, tptarg)  # where the two contours cross

            # How many intersections? 0, 1 or more?  If single intersection use those values
            nsols[i] = len(x)
            if nsols[i] == 0:
                # no solution. This shouldn't happen with known scenarios,
                # since the forward model calculated threshold.
                if use_fwd_model:
                    print('electrode: ', i, ';  no solution, but this is a known scenario')
                    print('min thr in 2D: ', np.min(mono_thr[2:, :]), ' and ', np.min(tripol_thr[2:, :]))
                    print('max thr in 2D: ', np.max(mono_thr[2:, :]), ' and ', np.max(tripol_thr[2:, :]))
                    print(' and goals are: ', mptarg, ' and ', tptarg)

                rp_guess, sv_guess = closest_midpoint(mpx, mpy, tpx, tpy)
                print('electrode: ', i, ';  no solutions. Using midpoint of closest contour points ',
                      '(position, survival): ', rp_guess, sv_guess)

            elif nsols[i] == 1:  # unique solution
                print('electrode: ', i, ';  one solution: ', x, y)
                rp_guess = x
                sv_guess = y

            else:  # multiple solutions
                print('electrode: ', i, ';  ', nsols[i], ' solutions: ', x, ' and: ', y)
                # figure out which is best
                print("which is best? use average")
                rp_guess = np.mean(x)
                sv_guess = np.mean(y)

        if if_plot_guess_contours:
            ax5.plot(rp_guess, sv_guess, 'x')
            plt.show()

            # additional plot with distance rather than rpos
            fig5, ax6 = plt.subplots()
            for seg in mp_segs:
                ax6.plot(1 - seg[:, 0], seg[:, 1], 'k')
            for seg in tp_segs:
                ax6.plot(1 - seg[:, 0], seg[:, 1], 'r')
            ax6.set_xlabel("Electrode distance (mm)")
            ax6.set_ylabel("Survival probability (%)")
            ax6.set_title(titletext)
            plt.show()

        if hasattr(rp_guess, '__len__'):
            fitrposvals[i] = rp_guess[0]
        else:
            fitrposvals[i] = rp_guess
        if hasattr(sv_guess, '__len__'):
            fitsurvvals[i] = sv_guess[0]
        else:
            fitsurvvals[i] = sv_guess

    # fix values for first and last electrodes as identical to their neighbors
    fitrposvals[0] = fitrposvals[1]
    fitrposvals[-1] = fitrposvals[-2]
    fitsurvvals[0] = fitsurvvals[1]
    fitsurvvals[-1] = fitsurvvals[-2]

    return fitrposvals, fitsurvvals
