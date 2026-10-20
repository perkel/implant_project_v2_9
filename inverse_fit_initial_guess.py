# inverse_fit_initial_guess.py
# Initial-guess estimation for the 'combined' fit mode in inverse_model_multi2.py: for each
# electrode, find a starting (rpos, survival) guess by intersecting the monopolar and tripolar
# threshold contours from the 2D forward-model maps.
#
# Split out of InverseModelMulti.py (which is left untouched) as a self-contained sub-algorithm --
# behavior is identical to the original inline code.

import matplotlib.pyplot as plt
import intersection as intsec
from common_params import *


def find_closest(x1, y1, x2, y2):  # returns indices of the point on each curve that are closest
    # Brute force (pretty ugly, but hopefully a rare case)
    # TODO Should test for case where x1 or x2 is a scalar
    n1 = len(x1)
    n2 = len(x2)

    min_dist = 5.0
    min_idx = [0, 0]
    for ii in range(n1):
        for jj in range(n2):
            dist = np.sqrt((((x1[ii] - x2[jj]) / 2) ** 2) + ((y1[ii] - y2[jj]) ** 2))
            if dist < min_dist:
                min_idx = [ii, jj]
                min_dist = dist

    return min_idx


def estimate_initial_guesses(thresh_summary_full, scen, mono_thr, tripol_thr, sig_vals,
                             if_plot_guess_contours, use_fwd_model):
    """For fit_mode == 'combined': for each electrode (except the two end ones, which have no
    tripolar target), find an initial (rpos, survival) guess by intersecting the monopolar and
    tripolar threshold contours from the 2D forward-model maps. Falls back to a closest-point
    heuristic (overridden by a fixed test value -- see the TODO below, carried over unchanged from
    the original) when the contours don't intersect, and to the mean of all intersections when
    there's more than one. Returns (fitrposvals, fitsurvvals), both length NELEC, with the end
    electrodes copied from their neighbors.
    """
    nsols = np.zeros(NELEC)  # number of solutions found from the 2D maps
    fitrposvals = np.zeros(NELEC)
    fitsurvvals = np.zeros(NELEC)

    for i in range(1, NELEC - 1):  # Fit params for each electrode (except ends, where there is no tripol value)
        mptarg = thresh_summary_full[scen, 0, 0, :][i]
        tptarg = thresh_summary_full[scen, 0, 1, :][i]

        fig4, ax5 = plt.subplots()
        # ax5 stays the real Axes throughout (previously it was reassigned to the ContourSet
        # returned by plt.contour(), then accessed back via '.axes' -- confusing, and the source of
        # a PyCharm warning since Artist.axes is typed Optional[Axes]). The two ContourSets are kept
        # in their own variables instead. This also drops three redundant calls that were all
        # setting the same x/y labels on what is a single Axes object.
        cs_mp = ax5.contour(rpos_grid_vals, surv_grid_vals[2:], mono_thr[2:, :], [mptarg], colors='green')
        cs_tp = ax5.contour(rpos_grid_vals, surv_grid_vals[2:], tripol_thr[2:, :], [tptarg], colors='red')
        ax5.set_xlabel('Rpos (mm)')
        ax5.set_ylabel('Survival fraction')
        mpcontour = cs_mp.allsegs[0]
        tpcontour = cs_tp.allsegs[0]
        if not if_plot_guess_contours:
            plt.close(fig4)

        # Each contour segment from allsegs is already an (N, 2) array of (x, y) points, so the
        # x/y columns can be sliced out directly instead of copied one point at a time.
        mpx = mpcontour[0][:, 0]
        mpy = mpcontour[0][:, 1]
        tpx = tpcontour[0][:, 0]
        tpy = tpcontour[0][:, 1]

        x, y = intsec.intersection(mpx, mpy, tpx, tpy)  # find intersection(s)

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

            mp_idx, tp_idx = find_closest(mpx, mpy, tpx, tpy)

            rp_guess = (mpx[mp_idx] + tpx[tp_idx]) / 2.0
            sv_guess = (mpy[mp_idx] + tpy[tp_idx]) / 2.0
            ax_guess = plt.plot(rp_guess, sv_guess, 'x')
            titletext = ("sigvals: " + f"{sig_vals[0]: .1f}" + ' and ' + f"{sig_vals[1]: .1f}" +
                        " thr: " + f"{mptarg: .1f}" + ' and ' + f"{tptarg: .1f}")
            plt.title(titletext)

            print('electrode: ', i, ";  no solutions. Closest: ", mp_idx, ' and ', tp_idx,
                  ' , OVERRIDING to: (position, survival): ', rp_guess, sv_guess)

            print('electrode: ', i, ';  no solutions. OVERRIDING to: (position, survival): ', rp_guess, sv_guess)

        elif nsols[i] == 1:  # unique solution
            print('electrode: ', i, ';  one solution: ', x, y)
            rp_guess = x
            sv_guess = y
            ax_guess = plt.plot(rp_guess, sv_guess, 'x')

        else:  # multiple solutions
            print('electrode: ', i, ';  ', nsols[i], ' solutions: ', x, ' and: ', y)
            which_sols = np.zeros((4, int(nsols[i])))  # array for solutions and best fit
            for sol in range(int(nsols[i])):  # Try all potential solutions; keep best
                rp_guess = x[sol]
                sv_guess = y[sol]

                rposweight = 2
                survweight = 0.0
                if i > 1:  # calculate distance from previous coords
                    which_sols[3, sol] = np.sqrt(rposweight * (rp_guess - fitrposvals[i - 1]) ** 2 +
                                                 survweight * (sv_guess - fitsurvvals[i - 1]) ** 2)

            # figure out which is best
            print("which is best? use average")
            rp_guess = np.mean(x)
            sv_guess = np.mean(y)

            ax_guess = plt.plot(rp_guess, sv_guess, 'x')

        if if_plot_guess_contours:
            plt.show()

            # additional plot with distance rather than rpos
            plt.plot(1 - mpx, mpy, 'k')
            plt.plot(1 - tpx, tpy, 'r')
            plt.xlabel("Electrode distance (mm)")
            plt.ylabel("Survival probability (%)")
            titletext = ("sigvals: " + f"{sig_vals[0]: .1f}" + ' and ' + f"{sig_vals[1]: .1f}" +
                        " thr: " + f"{mptarg: .1f}" + ' and ' + f"{tptarg: .1f}")
            plt.title(titletext)
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
