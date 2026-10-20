# Import critical packages
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

from common_params import *
import pickle
import csv
from matplotlib.font_manager import findfont, FontProperties
import matplotlib as mpl


# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def fig3_neuron_activation():

    mpl.rcParams['font.family'] = 'Arial'

    espace = 1.1
    if espace == 0.85:
        e_txt = '085'
    elif espace == 1.1:
        e_txt = '110'
    elif espace ==2.2:
        e_txt='220'
    else:
        e_txt = 'xxx'
    es_text = '_espace_' + e_txt

    font = findfont(FontProperties(family=['Arial']))
    # Rename fwd and inverse output directories
    new_dir_suffix = 'RE%d' % R_EXT + '_' + 'RI%d' % R_INT+ 'std_%.1f' % ACT_STDREL + '_thr_%d' % THRTARG
    # offset = len(cp.FWD_OUT_PRFIX)
    FWD_OUT_PRFIX = 'FWD_OUTPUT/'
    FWDOUTPUTDIR = FWD_OUT_PRFIX + new_dir_suffix
    activation_file = FWDOUTPUTDIR + '/neuronact_' + STD_TEXT + es_text + '.npz'
    print("Activation file: ", activation_file)
    data = np.load(activation_file, allow_pickle=True)
    surv_vals = data['arr_0']
    rpos_vals = data['arr_1']
    neuronact = data['arr_2']

    # can be useful for some debugging or draft figures
    hires = '_hi_res'
   # descrip = "surv_" + str(np.min(surv_vals)) + "_" + str(np.max(surv_vals)) + "_rpos_" + \
    #          str(np.min(rpos_vals)) + "_" + str(np.max(rpos_vals)) + hires
#
#
 #
    descrip = (
        f"surv_{np.min(surv_vals):.2f}_{np.max(surv_vals):.2f}_rpos_"
        f"{np.min(rpos_vals):.2f}_{np.max(rpos_vals):.2f}{hires}"
    )

    params_file = f"{FWDOUTPUTDIR}/simParams{descrip}{es_text}.pickle"
    params_file = FWDOUTPUTDIR + '/simParams' + descrip + es_text + '.pickle'
    # sp = np.load(params_file + '.npy', allow_pickle=True)
    with open(params_file, 'rb') as f:
        sp = pickle.load(f)
        f.close()

    posvals = np.arange(0, 33, 0.01)
    posvals -= posvals[-1]/2.0

    # Load monopolar data
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

    # Make plots
    survidxs = []
    rposidxs = []
    plt_surv_vals = [0.8]  # desired survival values to be plotted
    nsurv = len(plt_surv_vals)
    plt_rpos_vals = [-0.5, 0.0, 0.5]  # desired rpos values to be plotted
    nrpos = len(plt_rpos_vals)
    fig, axs = plt.subplots(nrpos, nsurv, sharex=True, sharey=True, figsize=(5, 8))
    plt.figtext(0.015, 0.89, 'A', fontsize=20, weight='bold')
    plt.figtext(0.015, 0.62, 'B', fontsize=20, weight='bold')
    plt.figtext(0.015, 0.34, 'C', fontsize=20, weight='bold')

    for i, val in enumerate(plt_surv_vals):
        theidx = np.argmin(np.abs(surv_vals - val))
        survidxs.append(theidx)

    for i, val in enumerate(plt_rpos_vals):
        theidx = np.argmin(np.abs(rpos_vals - val))
        rposidxs.append(theidx)

    # Set labels and plot
    titletext = []
    for i, ax in enumerate(axs.flat):
        row = int(i / nsurv)
        col = int(np.mod(i, nsurv))
        # if row == 0:
        #     titletext = 'surv = %.2f' % surv_vals[survidxs[col]]
        #     ax.set_title(titletext)

        n_p_c = sp['neurons']['neur_per_clust']
        mono_prof = neuronact[survidxs[col], rposidxs[row], 0, 0, :]/n_p_c
        tripol_prof = neuronact[survidxs[col], rposidxs[row], 1, 0, :]/n_p_c
        ax.plot(posvals, mono_prof, '.',
                color='blue', linewidth=0.5)
        ax.plot(posvals, tripol_prof, '.',
                color='red', linewidth=0.5)
        ax.set(xlabel='Longitudinal distance (mm)')
        if row == 1 and col == 0:
            ax.set(ylabel='Fractional neuronal activation')
        # place threshold values here
        xlimit = [-4, 4]
        ax.set_xlim((xlimit[0], xlimit[1]))
       # ax.set_ylim((0.0, 0.08))
        #ax.set_yticks([0, 0.04, 0.08])
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.label_outer()
        if col == (nsurv - 1):
            mytext = 'Distance = ' + str(1 - plt_rpos_vals[row]) + ' mm'
            #ax.text(4.0, 0.075, mytext, horizontalalignment='right')
            #m_thr_text = "Monopolar thr.: %.2f dB" % mono_thr[survidxs[col], rposidxs[row]]
            #t_thr_text = "Tripoloar thr.: %.2f dB" % tripol_thr[survidxs[col], rposidxs[row]]
            #ax.text(4.0, 0.068, m_thr_text, horizontalalignment='right')
            #ax.text(4.0, 0.061, t_thr_text, horizontalalignment = 'right')
            ax.text(4.0, 0.4, mytext, horizontalalignment='right')
            m_thr_text = "Monopolar thr.: %.2f dB" % mono_thr[survidxs[col], rposidxs[row]]
            t_thr_text = "Tripoloar thr.: %.2f dB" % tripol_thr[survidxs[col], rposidxs[row]]
            ax.text(4.0, 0.3, m_thr_text, horizontalalignment='right')
            ax.text(4.0, 0.2, t_thr_text, horizontalalignment='right')

            print('surv ', titletext, ' dist ', str(1 - plt_rpos_vals[row]), ' Thr: ', mono_thr[survidxs[col],
                        rposidxs[row]], ' and ', tripol_thr[survidxs[col], rposidxs[row]])

        # find width at half height
        first_idx = np.argwhere(mono_prof > (np.max(mono_prof)/2))
        first_pt = posvals[first_idx[0]]
        width_half_height = 2*np.abs(first_pt)
        # find stddev of best-fit Gaussian
        p0 = [0.05, 0.0, 1.0]
        coeff, var_matrix = opt.curve_fit(gauss, posvals, mono_prof, p0=p0)
        print('width_half_height', width_half_height)
        print('coeffs: ', coeff)

        resvec = gauss(posvals, *coeff)
        ax.plot(posvals, resvec, color='green')


    # Save figure
    fig_filename = 'fig3_neuronact.pdf'
    plt.savefig(fig_filename, format='pdf')
    fig_filename = 'fig3_neuronact.png'
    plt.savefig(fig_filename, format='png')

    plt.show()


if __name__ == '__main__':

    fig3_neuron_activation()
