#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:46:54 2020
new version in 1.4 branch 19 May 2024. No substantive changes to this file

Sept 2026: Many changes using Claude to parallelize and stabilize the Bessel function calculations

@author: perkel
"""

# implant voltage calculations from Goldwyn et al, 2010, denovo.
#
# voltage_calc2.py -- based on voltage_calc.py, with the innermost loop over z values replaced by
# a single scipy.integrate.quad_vec() call per (electrode, y) pair instead of one integ.quad() call
# per (electrode, y, z) triple.
#
# For a fixed electrode position and y-offset, the expensive part of the integrand (the Bessel-order
# sum in integ_func) doesn't depend on z at all -- only the final cos(x*z) factor does. The original
# per-z quad() loop redundantly re-runs the same expensive adaptive integration once per z value even
# though most of the work is identical across all of them. quad_vec() integrates a vector-valued
# integrand (one value per z) using one shared adaptive x-grid, computing the expensive part once per
# sampled x and reusing it for every z -- benchmarked at ~12x faster than the per-z loop.
#
# This was tried once before and abandoned: quad_vec's shared adaptive search, needing to satisfy
# convergence for all z values from one grid, samples much closer to the lower integration bound
# (down to ~1e-12) than any single-z quad() call needs on its own, and at those very small x values,
# high-order (n up to 46) modified Bessel functions in goldwyn_beta's denominator hit an IEEE
# inf*0 = NaN (I_n(x) underflows to exactly 0 while K_n(x)/K_n'(x) overflow to +-inf as x -> 0, for
# high n). That instability has since been fixed at the source (see the note in goldwyn_beta below) --
# with the fix in place, quad_vec now samples all the way down to the true lower bound with zero NaN,
# so this version is safe to use.

import cProfile
import io
import os
import pickle
import pstats
import time
import concurrent.futures as cf
from datetime import datetime
from pstats import SortKey
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ
import scipy.special as spec
import remove_nans as r_nans


def goldwyn_beta(eps, k2, rs, re, n):
    # 'n' may be a scalar or an array of orders -- scipy.special's Bessel functions broadcast
    # over the order argument, so passing the full array of orders at once (see integ_func_over_z
    # below) computes all of them in one vectorized call instead of one Python-level call per order.
    #
    # Numerical note: for small k2 (the quadrature variable can get very close to the lower
    # integration bound, ~1e-12) combined with high order n, I_n(krs) underflows to exactly 0.0
    # while K_n(krs)/K_n'(krs) overflow to +-inf (small-argument Bessel asymptotics: I_n(z) ~
    # (z/2)^n/n! -> 0, K_n(z) ~ (n-1)!/2 * (2/z)^n -> inf as z -> 0, and both blow up/vanish faster
    # for larger n). That makes 'denom' below hit an IEEE inf*0 = nan for the highest orders in
    # exactly this regime -- confirmed to happen in real runs (not just a theoretical edge case),
    # e.g. order 46 at krs ~ 8.3e-6. Where I_n(krs) has genuinely underflowed to 0.0, the true
    # physical contribution of that order is astronomically small, so we force it to exactly 0
    # instead of letting the inf*0 corruption through -- verified to reproduce the correct,
    # smoothly-varying result (checked against neighboring z values) with zero effect anywhere
    # the computation was already stable. This same fix is what makes quad_vec's much deeper
    # excursion toward x=0 (see module docstring) safe: it also samples down near the true
    # 1e-12 lower bound, where even more orders underflow than in the per-z quad() case, but each
    # is caught and zeroed the same way.
    krs = k2 * rs
    with np.errstate(invalid='ignore'):
        iv_krs = spec.iv(n, krs)
        denom = ((eps * spec.kvp(n, krs) * iv_krs) - (spec.kn(n, krs) * spec.ivp(n, krs)))
        result = spec.iv(n, k2 * re) / (denom * krs)
    negligible = (iv_krs == 0.0)
    return result, negligible


def goldwyn_phi(eps, k, rs, re, reval, n):  # Note Goldwyn alpha is zero for eval pt. outide cylinder
    beta, negligible = goldwyn_beta(eps, k, rs, re, n)
    with np.errstate(invalid='ignore'):
        phi = beta * spec.kn(n, k * reval)
    # See the note in goldwyn_beta: force genuinely-negligible orders to exactly 0 rather than
    # propagating a NaN from either this multiplication or the one inside goldwyn_beta.
    return np.where(negligible, 0.0, phi)


def integ_func_over_z(x, m_max, pratio, rad, reval, z_arr, theta, relec):
    """Vector-valued integrand: returns one value per z in z_arr, for a single scalar x. The
    (expensive) Bessel-order sum below doesn't depend on z at all -- only the final cos(x*z)
    factor does -- so it's computed once here and shared across every z, instead of being
    recomputed once per z as the original per-z integ_func() effectively did across separate
    quad() calls. Passed to scipy.integrate.quad_vec() instead of integ.quad()."""
    n_arr = np.arange(m_max)
    gamma = np.full(m_max, 2.0)
    gamma[0] = 1.0
    phi_vals = goldwyn_phi(pratio, x, rad, relec, reval, n_arr)
    sum_contents = np.sum(gamma * np.cos(n_arr * theta) * phi_vals)
    return np.cos(x * z_arr) * sum_contents


def _process_electrode(i, r_elec, z_eval, m_max, res_ratio, cyl_radius, reval_base, int_start, int_end,
                        itol, y_vals, n_yeval, n_ypos, y_inc, n_relec):
    """Worker task: compute the full (y, z) voltage/activation grid for one electrode radial
    position. Electrode positions are fully independent of each other (each only reads its own
    'r_elec' and writes its own row of voltage_vals/activation_vals), so this loop is parallelized
    across a process pool exactly as in voltage_calc.py. Within one electrode, the z loop is
    handled by quad_vec (see integ_func_over_z) rather than a Python-level loop over z."""
    z_eval = np.asarray(z_eval)  # fp['zEval'] is a plain tuple; integ_func_over_z needs an array
    # for its x * z_arr broadcast (quad_vec's vector-valued integrand contract).
    n_z = len(z_eval)
    voltage_row = np.zeros((n_ypos, n_z))

    print('starting electrode position ', i, ' of ', n_relec)

    # loop on y positions to get 2nd spatial derivative -- only n_yeval+1 of them (the rest are
    # filled in by mirror symmetry below), same as voltage_calc.py
    for j, y_val in enumerate(y_vals[0:n_yeval + 1]):
        this_theta = np.arctan(y_val / reval_base)
        r_prime = np.sqrt((np.power(y_val, 2)) + (reval_base ** 2))  # distance of eval pt. from center of cylinder
        itemp, error = integ.quad_vec(integ_func_over_z, int_start, int_end, epsabs=itol, limit=1000,
                                      norm='max', args=(m_max, res_ratio, cyl_radius, r_prime, z_eval,
                                                         this_theta, r_elec))
        temp_v = itemp / (2 * (np.pi ** 2))  # From Goldwyn eqn. 11 -- one value per z, shape (n_z,)
        voltage_row[j, :] = temp_v
        voltage_row[n_ypos - (j + 1), :] = temp_v  # place same values in mirror-symmetric position

    # Calculate the second spatial derivative in the y dimension, for all z at once. Same math as
    # voltage_calc.py's per-z np.diff(np.diff(trans_voltage))/(y_inc**2), just done in one shot
    # along axis 0 (the y axis) instead of once per z inside a Python loop.
    trans_v_prime = np.diff(np.diff(voltage_row, axis=0), axis=0) / (y_inc ** 2)  # shape (1, n_z)
    activation_row = trans_v_prime[n_yeval - 1, :]  # Value in center; shape (n_z,)

    nan_ms = np.where(np.isnan(activation_row))[0]
    if nan_ms.size:
        print('nan value for i == ', i, " and m in ", nan_ms)

    return i, voltage_row, activation_row


if __name__ == '__main__':
    # See Rubinstein dissertation, 1988, Ch 6.

    run_start_time = time.perf_counter()  # wall-clock timer for the entire run (see the end of this
    # block for where it's reported). This is a proper wall-clock measurement covering the whole run,
    # including the parallelized electrode-position loop -- unlike the cProfile-based 'run_duration'
    # previously stored below, which only measures CPU time in the main process and therefore massively
    # under-counts the real time once the heavy computation moved into worker processes.

    # Main parameters to vary
    radius = 1.0  # cylinder radius
    res_int = 70.0  # internal resistivity
    res_ext = 250.0  # external resistivity
    vtable_dir = 'v_tables/'
    if not os.path.exists(vtable_dir):
        os.makedirs(vtable_dir)

    date = '18October2026'
    resol = '_MedResolution_'
    rext_text = 'Rext' + str(round(res_ext))
    r_int_text = '_Rint' + str(round(res_int)) + '_quadvec.dat'
    output_filename = vtable_dir + date + resol + rext_text + r_int_text

    pr = cProfile.Profile()

    # Field parameters. zEval can be higher (more precise, slower) or lower resolution (less precise, faster)
    fp = {'model': 'cylinder_3d', 'fieldtype': 'Voltage_Activation', 'evaltype': 'SG', 'cylRadius': radius,
          'relec': np.arange(-0.95, 0.951, 0.05), 'resInt': res_int, 'resExt': res_ext, 'rspace': 1.3, 'theta': 0.0,
          # "High" resolution
          # 'zEval': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
          #              1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
          #              3.0, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
          #              4.8, 4.9, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8,
          #              8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
          #              13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
          #              22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0),
          # Medium resolution
          'zEval': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2,
                 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
                   8.5, 9.0, 9.5, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
                   32.0, 34.0, 36.0, 38.0, 40.0),
          # Low resolution
          # 'zEval': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2,
          #            1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
          #           8.5, 9.0, 9.5, 10.0, 15, 20.0, 25.0, 30.0, 35.0, 40),
          # Very low resolution for debugging
          #'zEval': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
          'mMax': 47, 'intStart': 1e-12, 'intEnd': 500.0, 'reval': 1.3,
          'ITOL': 1e-6, 'runDate': 'rundate', 'runOutFile': 'savefile', 'run_duration': 0.0}

    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    fp['runDate'] = date_time
    fp['runOutFile'] = output_filename

    # Derived values
    res_ratio = fp['resInt'] / fp['resExt']
    n_z = len(fp['zEval'])  # this may be a problem if zEval is just a scalar
    n_ypos = 3  # # of values to calculate across the y dimension for the 2nd spatial derivative

    relec_range = fp['relec']  # electrode radial positions
    n_relec = len(relec_range)
    voltage_vals = np. zeros((n_relec, n_ypos, n_z))
    activation_vals = np.zeros((n_relec, n_z))
    if_plot = False

    pr.enable()  # Start the profiler
    # Note: this only profiles the main process below (task submission/collection), not the actual
    # per-electrode computation, which now runs in worker processes and won't show up here.

    n_yeval = n_ypos // 2  # floor division
    y_inc = 0.001  # 10 microns
    y_vals = np.arange(-n_yeval * y_inc, n_yeval * (y_inc * 1.01), y_inc)

    # loop on electrode radial positions -- each position is fully independent (reads only its own
    # r_elec, writes only its own row of voltage_vals/activation_vals), so this is parallelized
    # across processes rather than run serially.
    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print('Parallelizing electrode-position loop across ', n_workers, ' worker processes')

    with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_process_electrode, i, r_elec, fp['zEval'], fp['mMax'], res_ratio,
                                    fp['cylRadius'], fp['reval'], fp['intStart'], fp['intEnd'], fp['ITOL'],
                                    y_vals, n_yeval, n_ypos, y_inc, n_relec)
                   for i, r_elec in enumerate(relec_range)]

        n_done = 0
        for fut in cf.as_completed(futures):
            i, voltage_row, activation_row = fut.result()
            voltage_vals[i, :, :] = voltage_row
            activation_vals[i, :] = activation_row
            n_done += 1
            print('electrode position ', n_done, ' of ', n_relec, ' complete (i=', i, ')')

    pr.disable()  # stop the profiler

    # Display profiler results
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(10)
    print(s.getvalue())

    # Save the data into a single file
    # Note: ps.total_tt (cProfile's cumulative CPU time) only covers the main process -- the actual
    # per-electrode computation above now runs in worker processes and isn't included in it, so it
    # would badly under-report the real time since parallelization. Use the wall-clock timer instead.
    fp['run_duration'] = time.perf_counter() - run_start_time  # Place run duration in field parameters
    with open(fp['runOutFile'], 'wb') as combined_data:
        pickle.dump([fp, voltage_vals, activation_vals], combined_data)  # Save all data
    combined_data.close()

    nan_locs = np.argwhere(np.isnan(activation_vals))
    if nan_locs.size != 0:
        print('WARNING: There were NaN values in the act_vals array at locations: ', nan_locs)
        r_nans.remove_nans(fp['runOutFile'])
        print('These have been replaced by interpolated values in a new file with no_nans appended to the filename.')
    else:
        print('Complete: No nans are in the file.')

    run_elapsed_time = time.perf_counter() - run_start_time
    print(f'Total run duration: {run_elapsed_time:.1f} s ({run_elapsed_time / 60:.2f} min)')

    if if_plot:
        # Some plots that might be helpful
        #  plt.plot(relec_range, voltageVals[:, 10], 'or')
        #  plt.plot(relec_range, activationVals[:, 0], 'ob')
        #  plt.xlabel('Electrode radial position (mm)')
        #  plt.ylabel('Red: voltage; Blue: activation')
        #  plt.xlim(-1.0, 1.0)
        #  plt.yscale('log')

        plt.figure()
        plt.xlabel('y position (mm)')
        plt.ylabel('Voltage')
        plt.plot(y_vals, voltage_vals[0, :, 0], '.b')
        plt.plot(y_vals, voltage_vals[1, :, 0], '.g')
        plt.show()
