# Script to make figures 4 (3D plot) & 5 (example contours
# for Perkel, Goldwyn & Arenberg (2023) implant model paper on forward and inverse models

# Import required packages
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv
from scipy import interpolate
import intersection as intsec
from common_params import *
import shapely as shap


# noinspection PyUnusedLocal
def fig_2D_contour():
    # this is an option to search systematically for unique or multiple solutions for a given set of
    # monopolar and tripolar threshold values
    map_unique_solutions = False
    filled_contours = False  # Fill the contour plots? Otherwise make contour lines
    # Set default figure values
    mpl.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 2

    # Declare variables  TOTO should get these from the saved or some common param
    surv_vals = np.arange(0.04, 0.97, 0.02)
    nsurv = len(surv_vals)
    rpos_vals = np.arange(-0.95, 0.96, 0.02)
    nrpos = len(rpos_vals)

    espace = 1.1
    if espace == 0.85:
        e_txt = '085'
    elif espace == 1.1:
        e_txt = '110'
    else:
        e_txt = 'xxx'
    es_text = '_espace_' + e_txt

    # Load monopolar data
    # Rename fwd and inverse output directories
    new_dir_suffix = 'RE%d' % R_EXT + '_'+'RI%d' % R_INT +'std_%.1f' % ACT_STDREL + '_thr_%d' % THRTARG
    # offset = len(cp.FWD_OUT_PRFIX)
    FWD_OUT_PRFIX = 'FWD_OUTPUT/'
    FWDOUTPUTDIR = FWD_OUT_PRFIX + new_dir_suffix

    datafile = FWDOUTPUTDIR + "/Monopolar_2D_" + STD_TEXT + es_text + ".csv"
    file = open(datafile)
    numlines = len(file.readlines())
    file.close()

    with open(datafile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        ncol = len(next(datareader))
        csvfile.seek(0)
        mono_thr = np.empty([numlines, ncol])
        for i, row in enumerate(datareader):
            # Do the parsing
            mono_thr[i, :] = row

    print("average monopolar threshold: ", np.nanmean(mono_thr))

    # Load tripolar data
    datafile = FWDOUTPUTDIR + "/Tripolar_09_2D_" + STD_TEXT + es_text + ".csv"
    file = open(datafile)
    numlines = len(file.readlines())
    file.close()

    with open(datafile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        ncol = len(next(datareader))
        csvfile.seek(0)
        tripol_thr = np.empty([numlines, ncol])
        for i, row in enumerate(datareader):
            # Do the parsing
            tripol_thr[i, :] = row

    print('Data retrieved from directory: ', FWDOUTPUTDIR)

    # Measure min/max/mean differences between monopolar and tripolar
    thr_diff = tripol_thr - mono_thr
    mean_diff = np.mean(thr_diff[:])
    min_diff = np.min(thr_diff[:])
    max_diff = np.max(thr_diff[:])
    print('Min/max/mean differences: ', min_diff, ' , ', max_diff, ' , ', mean_diff)

    # # set up 2D interpolation
    rp_curt = rpos_vals[0:-2]
    xnew = np.linspace(rpos_vals[1], rpos_vals[-1], 50)
    ynew = np.linspace(surv_vals[1], surv_vals[-1], 50)
    np.meshgrid(xnew, ynew)

    f_interp = interpolate.RectBivariateSpline(rp_curt, surv_vals, mono_thr[:, 0:-2].transpose())
    znew_mp = f_interp(xnew, ynew)
    m_min = np.min(znew_mp)
    m_max = np.max(znew_mp)

    # f_interp = interpolate.interp2d(rp_curt, surv_vals, tripol_thr[:, 0:-2])
    f_interp = interpolate.RectBivariateSpline(rp_curt, surv_vals, tripol_thr[:, 0:-2].transpose())
    znew_tp = f_interp(xnew, ynew)
    t_min = np.min(znew_tp)
    t_max = np.max(znew_tp)

    # all_min = np.min([t_min, m_min])  # if we want to do this automatically
    # all_max = np.max([t_max, m_max])
    # rounding manually for figure
    all_min = 30.0
    all_max = 90.0
    n_levels = 6
    labels = ['P1', 'P2', 'P3', 'P4']

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.tight_layout(pad=3, w_pad=2, h_pad=2.0)
    plt.figtext(0.03, 0.93, 'A', fontsize=20, weight='bold')
    plt.figtext(0.51, 0.93, 'B', fontsize=20, weight='bold')
    plt.figtext(0.03, 0.49, 'C', fontsize=20, weight='bold')
    plt.figtext(0.51, 0.49, 'D', fontsize=20, weight='bold')

    if filled_contours:
        cs3 = axs[0, 0].contourf(1 - rpos_vals, surv_vals, mono_thr,
                                 np.arange(all_min, all_max, (all_max - all_min) / n_levels),
                                 cmap='viridis', extend='both')
        low_rpos_val = -0.5  # these are the P1-4 points
        high_rpos_val = 0.5
        low_surv_val = 0.4
        high_surv_val = 0.8
        axs[0, 0].set_xlabel('Electrode distance (mm)')
        axs[0, 0].set_ylabel('Fractional neuronal density')
        axs[0, 0].set_title('Monopolar', fontsize=12)
        lab_shift = 0.025
        axs[0, 0].text(1 - high_rpos_val + lab_shift, high_surv_val, labels[0], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 0].text(1 - low_rpos_val + lab_shift, high_surv_val, labels[1], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 0].text(1 - high_rpos_val + lab_shift, low_surv_val, labels[2], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 0].text(1 - low_rpos_val + lab_shift, low_surv_val, labels[3], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 0].plot([1 - high_rpos_val], [high_surv_val], 'sk', markersize=20)
        axs[0, 0].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [high_surv_val, high_surv_val], color='blue')
        axs[0, 0].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [low_surv_val, low_surv_val], color='blue',
                       linestyle='dashed')
        axs[0, 0].plot([1 - low_rpos_val, 1 - low_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='black',
                       linestyle='dashed')
        axs[0, 0].plot([1 - high_rpos_val, 1 - high_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='black')
        axs[0, 0].set_xticks([0.4, 0.8, 1.5, 1.6])



        cs4 = axs[0, 1].contourf(1 - rpos_vals, surv_vals, tripol_thr,
                                 np.arange(all_min, all_max, (all_max - all_min) / n_levels),
                                 cmap='viridis', extend='both')
        axs[0, 1].set_title('Tripolar', fontsize=12)
        axs[0, 1].set_xlabel('Electrode distance (mm)')
        axs[0, 1].text(1 - high_rpos_val + lab_shift, high_surv_val, labels[0], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 1].text(1 - low_rpos_val + lab_shift, high_surv_val, labels[1], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 1].text(1 - high_rpos_val + lab_shift, low_surv_val, labels[2], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 1].text(1 - low_rpos_val + lab_shift, low_surv_val, labels[3], horizontalalignment='left',
                       verticalalignment='bottom')
        axs[0, 1].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [high_surv_val, high_surv_val], color='red')
        axs[0, 1].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [low_surv_val, low_surv_val], color='red',
                       linestyle='dashed')
        axs[0, 1].plot([1 - low_rpos_val, 1 - low_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='gray',
                       linestyle='dashed')
        axs[0, 1].plot([1 - high_rpos_val, 1 - high_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='gray')
        axs[0, 1].set_xticks([0.4, 0.8, 1.2, 1.6])
    else:
        cs3 = axs[0, 0].contour(1 - rpos_vals, surv_vals, mono_thr,
                                np.arange(all_min, all_max, (all_max - all_min) / n_levels), colors='k')

        # This custom formatter removes trailing zeros, e.g. "1.0" becomes "1", and
        # then adds a percent sign.
        def fmt(x):
            s = f"{x:.1f}"
            if s.endswith("0"):
                s = f"{x:.0f}"
            return rf"{s} dB" if plt.rcParams["text.usetex"] else f"{s} dB"

        manual_locations = [(0, 0.9), (0.3, 0.6), (0.8, 0.3), (1.0, 0.2)]
        axs[0, 0].clabel(cs3, cs3.levels, inline=True, fmt=fmt, fontsize=10, manual=manual_locations)
        low_rpos_val = -0.5
        high_rpos_val = 0.5
        low_surv_val = 0.4
        high_surv_val = 0.8
        axs[0, 0].set_xlabel('Electrode distance (mm)')
        axs[0, 0].set_ylabel('Fractional neuronal density')
        axs[0, 0].set_title('Monopolar', fontsize=12)
        lab_shift = 0.025
        axs[0, 0].text(1 - high_rpos_val - 0.15, high_surv_val - 0.06, labels[0], horizontalalignment='left',
                       verticalalignment='bottom', fontweight='bold')
        axs[0, 0].text(1 - low_rpos_val + lab_shift, high_surv_val - 0.06, labels[1], horizontalalignment='left',
                       verticalalignment='bottom', fontweight='bold')
        axs[0, 0].text(1 - high_rpos_val - 0.15, low_surv_val, labels[2], horizontalalignment='left',
                       verticalalignment='bottom', fontweight='bold')
        axs[0, 0].text(1 - low_rpos_val + lab_shift, low_surv_val, labels[3], horizontalalignment='left',
                       verticalalignment='bottom', fontweight='bold')
        axs[0, 0].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [high_surv_val, high_surv_val], color='blue')
        axs[0, 0].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [low_surv_val, low_surv_val], color='blue',
                       linestyle='dashed')
        axs[0, 0].plot([1 - low_rpos_val, 1 - low_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='black',
                       linestyle='dashed')
        axs[0, 0].plot([1 - high_rpos_val, 1 - high_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='black')
        axs[0, 0].set_xticks([0.4, 0.8, 1.2, 1.6])


        pos_1=np.array([0.476232742, 0.476232742, 0.575021851, 0.656158433, 0.745848506, 0.812965907, 0.856014243, 0.968904478, 1.047277507, 1.103322552, 1.19391566, 1.294313055, 1.366825349, 1.432090617, 1.51687377, 1.51687377])
        surv_1=np.array([0.81395243, 0.81395243, 0.526326022, 0.429863864, 0.473558241, 0.553300381, 0.691186001, 0.986029868, 0.758011284, 0.434596852, 0.518933325, 0.477039127, 0.545469618, 0.721835387, 0.819748833, 0.819748833])
        axs[0, 0].plot(pos_1,surv_1, 'o')
        all_min = 30.0
        all_max = 90.0
        n_levels = 6
        cs4 = axs[0, 1].contour(1 - rpos_vals, surv_vals, tripol_thr,
                                np.arange(all_min, all_max, (all_max - all_min) / n_levels),
                                extend='both', colors='k')
        manual_locations = [(0, 0.9), (0.3, 0.3), (0.7, 0.7), (1.2, 0.2), (1.2, 0.5), (1.3, 0.02)]

        axs[0, 1].clabel(cs4, cs4.levels, inline=True, fmt=fmt, fontsize=10, manual=manual_locations)
        axs[0, 1].set_title('Tripolar', fontsize=12)
        axs[0, 1].set_xlabel('Electrode distance (mm)')
        axs[0, 1].text(1 - high_rpos_val + 0.03, high_surv_val - 0.06, labels[0], horizontalalignment='left',
                       verticalalignment='bottom', fontweight='bold')
        axs[0, 1].text(1 - low_rpos_val + lab_shift, high_surv_val - 0.06, labels[1], horizontalalignment='left',
                       verticalalignment='bottom', fontweight='bold')
        axs[0, 1].text(1 - high_rpos_val + 0.06, low_surv_val, labels[2], horizontalalignment='left',
                       verticalalignment='bottom', fontweight='bold')
        axs[0, 1].text(1 - low_rpos_val + lab_shift, low_surv_val, labels[3], horizontalalignment='left',
                       verticalalignment='bottom', fontweight='bold')
        axs[0, 1].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [high_surv_val, high_surv_val], color='red')
        axs[0, 1].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [low_surv_val, low_surv_val], color='red',
                       linestyle='dashed')
        axs[0, 1].plot([1 - low_rpos_val, 1 - low_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='gray',
                       linestyle='dashed')
        axs[0, 1].plot([1 - high_rpos_val, 1 - high_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='gray')
        axs[0, 1].set_xticks([0.4, 0.8, 1.2, 1.6])
        #example values from uniform scenerio
       # pos_1=np.array([0.476232742, 0.476232742, 0.575021851, 0.656158433, 0.745848506, 0.812965907, 0.856014243, 0.968904478, 1.047277507, 1.103322552, 1.19391566, 1.294313055, 1.366825349, 1.432090617, 1.51687377, 1.51687377])
       # surv_1=np.array([0.81395243, 0.81395243, 0.526326022, 0.429863864, 0.473558241, 0.553300381, 0.691186001, 0.986029868, 0.758011284, 0.434596852, 0.518933325, 0.477039127, 0.545469618, 0.721835387, 0.819748833, 0.819748833])
       # axs[0, 1].plot(pos_1,surv_1, 'o')
#        pos_1=np.array([1.105082129, 1.105082129, 1.078369686, 0.894760897, 1.042783396, 1.050013736, 1.139597599, 1.226986237, 1.271727668, 1.200529353, 1.068024186, 0.7044402, 1.00113993, 1.153762216, 1.214020265, 1.214020265])
  #      surv_1=np.array([0.635876938, 0.635876938, 0.626996472, 0.245047452, 0.586458575, 0.676018414, 0.811287604, 0.830225343, 0.734071391, 0.998805076, 0.334925774, 0.000260672, 0.318277833, 0.758215672, 0.789004392, 0.789004392])
  #      axs[0, 1].plot(pos_1,surv_1, 'o')

    #  calculate  indices from target survival and rpos values
    low_rpos_idx = np.argmin(np.abs(rpos_vals - low_rpos_val))
    high_rpos_idx = np.argmin(np.abs(rpos_vals - high_rpos_val))
    low_surv_idx = np.argmin(np.abs(surv_vals - low_surv_val))
    high_surv_idx = np.argmin(np.abs(surv_vals - high_surv_val))

    axs[1, 0].plot(1 - rpos_vals, mono_thr[high_surv_idx, :], color='blue', linestyle='solid')
    axs[1, 0].plot(1 - rpos_vals, tripol_thr[high_surv_idx, :], color='red', linestyle='solid')
    axs[1, 0].plot(1 - rpos_vals, mono_thr[low_surv_idx, :], color='blue', linestyle='dashed')
    axs[1, 0].plot(1 - rpos_vals, tripol_thr[low_surv_idx, :], color='red', linestyle='dashed')
    axs[1, 0].axes.set_xlabel('Electrode distance (mm)')
    axs[1, 0].axes.set_ylabel('Threshold (dB)')
    axs[1, 0].axes.set_xlim([0.1, 1.9])
    axs[1, 0].axes.set_ylim([30, 80])
    axs[1, 0].set_xticks([0.4, 0.8, 1.2, 1.6])

    axs[1, 1].plot(surv_vals, mono_thr[:, high_rpos_idx], color='black', linestyle='solid')
    axs[1, 1].plot(surv_vals, tripol_thr[:, high_rpos_idx], color='gray', linestyle='solid')
    axs[1, 1].plot(surv_vals, mono_thr[:, low_rpos_idx], color='black', linestyle='dashed')
    axs[1, 1].plot(surv_vals, tripol_thr[:, low_rpos_idx], color='gray', linestyle='dashed')
    axs[1, 1].axes.set_xlabel('Fractional neuronal density')
    axs[1, 1].axes.set_xlim([0.1, 0.9])
    axs[1, 1].axes.set_ylim([30, 80])

    filename = 'Fig4_2D_contour' + es_text + '.pdf'
    # plt.savefig('Fig4_2D_contour.eps', format='eps')
    plt.savefig(filename, format='pdf')

    figrows = 2
    figcols = 2
    fig2, axs2 = plt.subplots(figrows, figcols)
    idx_surv = 0
    idx_rpos = 0
    the_ax = 0
    for i in range(4):
        if i == 0:
            idx_surv = high_surv_idx
            idx_rpos = high_rpos_idx
            the_ax = axs2[0, 0]
        elif i == 1:
            idx_surv = high_surv_idx
            idx_rpos = low_rpos_idx
            the_ax = axs2[0, 1]
        elif i == 2:
            idx_surv = low_surv_idx
            idx_rpos = high_rpos_idx
            the_ax = axs2[1, 0]
        elif i == 3:
            idx_surv = low_surv_idx
            idx_rpos = low_rpos_idx
            the_ax = axs2[1, 1]

        this_mp_thr = [mono_thr[idx_surv, idx_rpos]]
        print('this_mp_thr: ', i, ' and ', this_mp_thr)
        cont_mp = the_ax.contour(1 - rpos_vals, surv_vals, mono_thr, this_mp_thr, colors='blue')
        if i == 2 or i == 3:
            the_ax.axes.set_xlabel('Electrode distance (mm)', fontsize=14)

        if i == 0:
            the_ax.axes.set_ylabel('Fractional neuronal density', fontsize=14)
            the_ax.yaxis.set_label_coords(-0.2, -0.08)

        this_tp_thr = [tripol_thr[idx_surv, idx_rpos]]
        print('this_tp_thr: ', i, ' and ', this_tp_thr)
        cont_tp = the_ax.contour(1 - rpos_vals, surv_vals, tripol_thr, this_tp_thr, colors='red')
        mpcontour = cont_mp.allsegs[0]
        tpcontour = cont_tp.allsegs[0]
        nmp = len(mpcontour[0])
        ntp = len(tpcontour[0])
        mpx = np.zeros(nmp)
        mpy = np.zeros(nmp)
        tpx = np.zeros(ntp)
        tpy = np.zeros(ntp)

        for j in range(0, nmp):  # Should be able to do this without for loops
            mpx[j] = mpcontour[0][j][0]
            mpy[j] = mpcontour[0][j][1]

        for j in range(0, ntp):
            tpx[j] = tpcontour[0][j][0]
            tpy[j] = tpcontour[0][j][1]

        x, y = intsec.intersection(mpx, mpy, tpx, tpy)  # find intersection(s)
        if i > 0 and len(x) > 0:
            the_ax.plot(x[-1], y[-1], 'x', color='black', markersize='10', mew=2.5)
        the_ax.plot(x, y, 'x', color='black', markersize='10', mew=2.5)
        the_ax.set_xlim([0, 1.9])
        the_ax.text(0.1, 0.8, labels[i], fontsize=16)

    plt.savefig('Fig5_contour_examples.pdf', format='pdf')

    if map_unique_solutions:
        n_sols = np.zeros((nrpos, nsurv), dtype=int)
        # Map whether there is only a single solutions giving these values (or 0 or > 1)
        for sidx, surv in enumerate(surv_vals):
            # print('sidx = ', sidx)
            print('Approximately ', 100 * (sidx * nrpos) / (nrpos * nsurv), ' % done.')
            for ridx, rpos in enumerate(rpos_vals):
                # print('ridx is ', ridx)

                fig, ax1 = plt.subplots()
                ax1 = plt.contour(rpos_vals, surv_vals, mono_thr, [mono_thr[sidx, ridx]], colors='green')
                ax2 = plt.contour(rpos_vals, surv_vals, tripol_thr, [tripol_thr[sidx, ridx]], colors='red')
                mpcontour = ax1.allsegs[0]
                tpcontour = ax2.allsegs[0]
                # if ifPlotContours == False:
                #    plt.close(fig)

                nmp = len(mpcontour[0])
                ntp = len(tpcontour[0])
                mpx = np.zeros(nmp)
                mpy = np.zeros(nmp)
                tpx = np.zeros(ntp)
                tpy = np.zeros(ntp)

                for j in range(0, nmp):  # Should be able to do this without for loops
                    mpx[j] = mpcontour[0][j]
                    mpy[j] = mpcontour[0][j][1]

                for j in range(0, ntp):
                    tpx[j] = tpcontour[0][j][0]
                    tpy[j] = tpcontour[0][j][1]

                # try finding intersection using Shapely class
                if len(mpcontour[0]) == 1 or len(tpcontour[0]) == 1:
                    n_sols[ridx, sidx] = 0
                else:
                    ('len contours: ', len(mpcontour[0]), len(tpcontour[0]))
                    line1 = shap.LineString(mpcontour[0])
                    line2 = shap.LineString(tpcontour[0])
                    x = shap.intersection(line1, line2)

                    # How many intersections? 0, 1 or more?  If single intersection use those values
                    if x.geom_type == 'Point':  # test for multiple identical values

                        n_sols[ridx, sidx] = 1

                    elif x.geom_type == 'MultiPoint':
                        print('Multipoint at sidx = ', sidx, ' and ridx: ', ridx, ' : ', x)
                        n_sols[ridx, sidx] = 2
                    else:
                        n_sols[ridx, sidx] = 0

        solmap_file = FWDOUTPUTDIR + 'solution_map' + STD_TEXT
        np.savez(solmap_file, surv_vals, rpos_vals, n_sols)
        fig_nsols, ax_nsols = plt.subplots()
        ax_nsols.contour(rpos_vals, surv_vals, n_sols, [1, 2])

    figname = 'Fig5_contour_examples.pdf'
    plt.savefig(figname, format='pdf', pad_inches=0.1)
    figname = 'Fig5_contour_examples.png'
    plt.savefig(figname, format='png', pad_inches=0.1)

    plt.show()



if __name__ == '__main__':
    fig_2D_contour()
