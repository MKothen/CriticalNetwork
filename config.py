"""
Configuration file for neural network simulation
All parameters and settings are defined here for easy modification
"""

from brian2 import *
import numpy as np

# ============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# ============================================================================
SEED = 42

# ============================================================================
# NETWORK STRUCTURE PARAMETERS
# ============================================================================
N_TOTAL_NEURONS = 1000          # Total number of neurons in the reservoir
P_MAX = 0.1                     # Connection probability for random connectivity
FRACTION_EXCITATORY = 0.8       # Fraction of excitatory neurons (0.8 = 80%)

# ============================================================================
# NEURON MODEL PARAMETERS (Adaptive Exponential Integrate-and-Fire)
# ============================================================================
# Membrane properties
C_mem = 200 * pF                # Membrane capacitance
V_L = -70 * mV                  # Leak potential
g_mem_val = 12 * nS             # Membrane conductance
Vr = -75 * mV                   # Reset potential

# Spike generation
D_T = 2 * mV                    # Slope factor for exponential
V_T_val = -50 * mV              # Threshold potential

# Synaptic reversal potentials
V_syn_exc = 0 * mV              # Excitatory reversal potential
V_syn_inh = -75 * mV            # Inhibitory reversal potential

# Synaptic time constants
tau_r_syn = 5 * ms              # Synaptic rise time
tau_d_syn = 50 * ms             # Synaptic decay time

# Adaptation parameters
g_A = 4 * nS                    # Adaptation conductance

# Noise parameters
tau_noise = 10 * ms             # Noise time constant
sigma_noise = 0.01 * nA         # Noise amplitude

# Heterogeneity parameters (for neuron diversity)
Vt_mean = V_T_val + 10 * mV
Vt_std = (Vt_mean - V_L) * 0.05
g_mem_mean = g_mem_val
g_mem_std = g_mem_mean * 0.15
b_mean = 0.02 * nA
b_std = b_mean * 0.15
tau_A_mean = 500 * ms
tau_A_std = tau_A_mean * 0.10

# ============================================================================
# SYNAPTIC CONDUCTANCE PARAMETERS
# ============================================================================
base_g_syn_max_exc_value = 1e-8 #(0.01 uS)
base_g_syn_max_inh_value = 1e-7 #(0.1 uS)


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
SIM_INITIAL_SETTLE_TIME = 0.3 * second    # Initial settling time
SIM_RUNTIME = 20 * second                  # Main simulation duration
ANALYSIS_DELAY_AFTER_SETTLE = 1.0 * second # Delay before analysis starts
set_dt = 0.1 * ms                          # Simulation timestep

# Analysis windows for CV calculation
CV_WINDOW_SIZE = 1000 * ms
CV_STEP_SIZE = 100 * ms

# ============================================================================
# RESERVOIR COMPUTING (RC) PARAMETERS
# ============================================================================
N_INPUT_NEURONS = 784                      # Number of MNIST pixels (28x28)
MNIST_CURRENT_AMPLITUDE = 0.2 * nA         # Input current amplitude
STIMULUS_DURATION_PER_IMAGE = 100 * ms     # How long to show each image
POST_STIMULUS_DURATION_TOTAL = 10 * ms    # Delay after stimulus
READOUT_SNAPSHOT_TIME_OFFSET = 10 * ms    # When to read out network state

# RC state calculation parameters
RC_STATE_SMOOTHING_WIDTH_STD_DEV = 20 * ms
RC_STATE_RATE_CALC_WINDOW_DURATION = 3 * RC_STATE_SMOOTHING_WIDTH_STD_DEV
RC_TRIAL_INTERNAL_SETTLE_TIME = 100 * ms

# Training parameters
NUM_TRAIN_SAMPLES_MAX = 100
NUM_TEST_SAMPLES_MAX = 100
READOUT_TRAINING_SUBSETS = [33, 66, 100]  # Sample sizes for learning curves
RIDGE_ALPHA = 1.0                          # Ridge regression regularization
FEED_INPUT_TO_INHIBITORY = True            # Whether inhibitory neurons receive input

# Learning metrics
QUICKNESS_TARGET_ACCURACY = 0.40           # Target accuracy for "quickness" metric
QUICKNESS_FIXED_SAMPLE_SIZE = 100          # Sample size for fixed accuracy metric

# ============================================================================
# EXPERIMENTAL DESIGN PARAMETERS
# ============================================================================
Imid_values_nA = np.array([0.3333])         # Input current values to test
EI_ratio_values = np.array([0.001, 0.385, 1.0])  # E/I ratio conditions
NUM_REPETITIONS = 8                        # Number of repetitions per condition
EXC_FACTOR_FIXED = 1.0

# Condition naming
sorted_ei_ratios = sorted(EI_ratio_values)
if len(sorted_ei_ratios) == 3:
    condition_map = {
        sorted_ei_ratios[0]: 'subcritical',
        sorted_ei_ratios[1]: 'critical',
        sorted_ei_ratios[2]: 'supercritical'
    }
else:
    condition_map = {val: f"EI_{val:.3f}" for val in sorted_ei_ratios}

# ============================================================================
# PLOTTING STYLE (Thesis-ready)
# ============================================================================
THESIS_STYLE = {
    "font.family": "serif",
    "font.serif": "Times New Roman",
    "mathtext.fontset": "dejavuserif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 300,
}