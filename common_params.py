# Common parameters for the implant model

import numpy as np

params_version = 2.0  # new version of common params for this
# Basic parameters
NELEC = 16
# ELEC_BASALPOS = 26.25  # in mm ** removed this 20 May 2024 to make sure electrode array is centered in neurons

# Neural activation parameters
R_EXT = 250.0  # ohm-cm
RE_TEXT = 'RE' + str(round(R_EXT))
R_INT = 70.0  # ohm-cm
RI_TEXT = 'RI' + str(round(R_INT))
THRTARG = 100
M_level_target = 100
TARG_TEXT = '_TARG' + str(round(THRTARG)) + '/'
ACTR = 100
ACTR_TEXT = '_ACTR' + str(round(ACTR)) + '_'
ACT_STDREL = 0.6  # 0.6#0.01#0.7
STD_TEXT = 'STDR' + str(ACT_STDREL)
STD_TEXT = STD_TEXT.replace('.', '_')

# File locations
sigmaVals = [0.0, 0.9]  # Always explore monopolar stimulation and one value of sigma for triploar
# sigmaVals is overridden if we're working on a subject, because we have specific values for them
all_sigs = [0.0, 0.5, 0.9]
# can be overridden for individual subjects
TP_SIGMA = sigmaVals[-1]  # standard partial-tripolar sigma value; use this instead of hardcoding 0.9 elsewhere
# behaviorVals = [0, 0]  # change to [0,1] if looking at M levels

COCHLEA = {'source': 'manual', 'timestamp': [], 'radius': []}
electrodes = {'source': 'manual', 'timestamp': [], 'zpos': [], 'rpos': []}
NEURONS = {'act_ctr': ACTR, 'act_stdrel': ACT_STDREL, 'nsurvival': [], 'sidelobe': 1, 'neur_per_clust': 10,
           'rlvl': [], 'rule': 'proportional', 'coef': 0.0, 'power': 1.0, 'thrtarg': THRTARG, 'M_targ': M_level_target}
# For COEF convex: <0 | 0.4, 0.9  linear: 0 | 1; concave: >0 | 1.0, 1.8
CHANNEL = {'source': 'manual', 'behavior': 0, 'number': range(0, NELEC), 'config': 'pTP', 'sigma': sigmaVals[0], 'alpha': 0.5,
           'current': 10000000000.0}
GRID = {'r': 0.1, 'th': 0.0, 'z': np.arange(0, 33, 0.01)}  # mm

#
# fit_using_val = 'mp' and 'tp' and 'mp_mcl' and 'ecap'  # could include any or all of these three types of data
# fit_using_val = 'mp' and 'tp'  # could include any or all of these three types of data

# all_measures = {'mp': True, 'tp': True, 'mp_mcl': True, 'ecap': True}  # measures calculated by the forward model
# all_measures = {'mp': True, 'tp': True}  # measures calculated by the forward model
all_measures = {'threshold': True}
# new system 17 June 2026. all_measures can now have 'threshold', 'mcl' and/or 'ecap'
# for 'threshold', all sigma values in all_sigs will be calculated by fwd_model and fwd_model_2d
# This allows us to have 2D models read for any subjects with 0.0, 0.5 or 0.9 sigma values
fit_measures = {'mp': 1.0, 'tp': 1.0, 'mp_mcl': 0.0, 'ecap': 0.0}  # measure weights for fitting with the inverse model
RUN_INFO = {'scenario': 'scenario', 'run_time': [], 'run_duration': 0.0, 'fit_measures': fit_measures}
global simParams
simParams = {'cochlea': COCHLEA, 'electrodes': electrodes, 'channel': CHANNEL, 'grid': GRID, 'neurons': NEURONS,
             'run_info': RUN_INFO}

nZ = len(GRID['z'])
ct_uncertainty = 0.1  # uncertainty for CT values in case one wants to display it on graphs

# Set specific scenarios to run with forward model.

# NOTE!!
# scenario names beginning with 'A' or 'S' and followed by 2 numerals are considered subjects
# other scenario names are considered to be synthetic and for use with the forward model

global tp_extend
tp_extend = False  # Whether to fit position and density at end electrodes

# Not used by the 2D exploration tool. These are left in for convenience
global scenarios
# scenarios = ['Gradual80R00']
# scenarios = ['Gradual80R75']
# scenarios = ['Uniform80R05']
# scenarios = ['pilot']
# scenarios = ['Uniform20R05']
# scenarios=['Uniform20R01']
# scenarios = ['Uniform100R05', 'Uniform100R10', 'Uniform100R15']
# scenarios = ['Uniform80R05', 'Uniform80R10', 'Uniform80R15']  # Used for Fig 3
# scenarios = ['Ramp80Rvariable1']
# scenarios = ['RampRpos_revSGradual80']
# scenarios = ['Rpos-03S0_4']
# scenarios = ['Ramp80Rvariable1']
# scenarios = ['RampRposS70']
# scenarios = ['Gradual80R-50_e11']
# scenarios = ['Gradual80R-50_e085']
# scenarios = ['RampRposS80']
# scenarios = ['RampRpos2SGradual80']
# scenarios = ['RampRposSOneHoleGradual80']
# scenarios = ['subject_2596']
scenarios = ['Gradual80R00', 'RampRposS80', 'RampRposSGradual80']  # for paper figure 6
# scenarios = ['RampRposSGradual80']  # for paper figure 6
# scenarios = ['Gradual2_80R00']
# scenarios=['Gradual80R-75']
# scenarios = ['RampRposRampSurv']
# scenarios = ['ExtremeHole']
# scenarios = ['RampRposS80']
# scenarios = ['RampRposSGradual80']
# scenarios = ['Checking_REXT_2500']
# scenarios = ['CustomForECAPFigure']
# scenarios = ['Step40_80R00']
# scenarios = ['Gradual07_02R00', 'Gradual07_02UR00']
# scenarios = ['RampRposSGradual80']


# Actual subject data. For inverse model only
# scenarios = ['A023R']  # paper "good fit" examples. Figure 7
# scenarios = ['Gradual80R00', 'RampRposS80', 'RampRposSGradual80', 'S22','S29','S38','S40', 'S41','S42','S43','S46','S47','S48','S49R','S50','S52','S53','S54','S55','S56','S57']
# scenarios = ['S29', 'S56']  # paper "poor fit" examples. Figure 8
# scenarios = ['A002R', 'A005L', 'A014L', 'A022L', 'A022R', 'A023R', 'A024L']
#scenarios = ['S42_test']
# all subjects with CT data
# scenarios = ['RampRposSGradual80', 'S22']
# scenarios = ['Gradual80R00', 'RampRposS80', 'RampRposSGradual80', 'S42', 'S43']
# scenarios = ['RampRposSGradual80']

# scenarios = ['Gradual80R00', 'RampRposS80', 'RampRposSGradual80', 'S22', 'S29', 'S38', 'S40', 'S41', 'S42',
#              'S43', 'S44', 'S46', 'S47', 'S49R', 'S50', 'S52', 'S53', 'S54', 'S55', 'S56', 'S57']
# scenarios = ['RampRposSGradual80', 'S22', 'S42',]

# scenarios = ['A014L']
# scenarios = ['A014L_constrained']
# scenarios = ['A014L_constrained_corrected']
# scenarios = ['A022L']
# scenarios = ['A022L_constrained']
# scenarios = ['A022L_constrained_corrected']
# scenarios = ['A022R']
# scenarios = ['A022R_constrained']
# scenarios = ['A022R_constrained_corrected']
# scenarios = ['A035R']
# scenarios = ['A035R_constrained']
# scenarios = ['A035R_constrained_corrected']
# scenarios = ['A041']
# scenarios = ['A041_constrained']
# scenarios = ['A041_constrained_corrected']

# File locations
FWD_OUT_PRFIX = 'FWD_OUTPUT/'
new_dir_suffix = RE_TEXT + '_' + RI_TEXT + 'std_%.1f' % ACT_STDREL + '_thr_%d' % THRTARG + '/'
# offset = len(cp.FWD_OUT_PRFIX)
FWDOUTPUTDIR = FWD_OUT_PRFIX + new_dir_suffix
INV_OUT_PRFIX = 'INV_OUTPUT/'
INVOUTPUTDIR = INV_OUT_PRFIX + new_dir_suffix
vtable_dir = 'v_tables/'

if RE_TEXT == 'RE70':
    FIELDTABLE = '16May2024_MedResolution_Rext70_nonans.dat'
elif RE_TEXT == 'RE125':
    FIELDTABLE = '3June2024_MedResolution_Rext125_nonans.dat'
elif RE_TEXT == 'RE250':
    if RI_TEXT == 'RI70':
        FIELDTABLE = '18October2026_MedResolution_Rext250_Rint70_quadvec.dat'
    elif RI_TEXT == 'RI250':
        FIELDTABLE = '16June2024_MedResolution_Rext250_Rint250_nonans.dat'

elif RE_TEXT == 'RE375':
    FIELDTABLE = '18May2024_MedResolution_Rext375.dat'
elif RE_TEXT == 'RE500':
    FIELDTABLE = '3June2024_MedResolution_Rext500.dat'
elif RE_TEXT == 'RE750':
    FIELDTABLE = '18Jan2024_MedResolution_Rext750_nonans.dat'
elif RE_TEXT == 'RE1000':
    FIELDTABLE = '3June2024_MedResolution_Rext1000_nonans.dat'
elif RE_TEXT == 'RE125O':
    FIELDTABLE = '28Dec2023_MedResolution_Rext1250.dat'
elif RE_TEXT == 'RE2500':
    FIELDTABLE = '7Dec2023_MedResolution_Rext2500_nonans.dat'

#  to override the automatic naming system
#  FIELDTABLE = 'Sample_Voltage.dat'#'Sample_voltage_70_ext.dat'#'Sample_Voltage.dat'#'Sample_Voltage.dat'#'Sample_voltage_70_ext.dat'#'Sample_Voltage.dat'
FIELDTABLE = vtable_dir + FIELDTABLE