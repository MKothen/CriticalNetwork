# Central Functions Reference Guide

This document provides detailed explanations of all central functions in the codebase, organized by module.

---

## Table of Contents

1. [config.py](#configpy)
2. [data_utils.py](#data_utilspy)
3. [network_model.py](#network_modelpy)
4. [analysis.py](#analysispy)
5. [reservoir.py](#reservoirpy)
6. [plotting.py](#plottingpy)
7. [statistics.py](#statisticspy)
8. [main_simulation.py](#main_simulationpy)

---

## config.py

**Purpose**: Central configuration file containing all simulation parameters.

### Key Parameter Groups

#### Network Structure
```python
N_TOTAL_NEURONS = 1000        # Total neurons in network
FRACTION_EXCITATORY = 0.8     # 80% excitatory, 20% inhibitory
P_MAX = 0.1                   # Connection probability
```

#### AdEx Neuron Model
```python
C_mem = 200 * pF              # Membrane capacitance
V_L = -70 * mV                # Leak (resting) potential
g_mem_val = 12 * nS           # Leak conductance
V_T_val = -50 * mV            # Spike threshold
D_T = 2 * mV                  # Exponential slope factor
g_A = 4 * nS                  # Adaptation conductance
```

#### Simulation Settings
```python
SIM_INITIAL_SETTLE_TIME = 0.3 * second   # Let network stabilize
SIM_RUNTIME = 20 * second                # Main simulation time
set_dt = 0.1 * ms                        # Integration timestep
```

#### Experimental Design
```python
Imid_values_nA = np.array([0.3333])               # Background currents to test
EI_ratio_values = np.array([0.001, 0.385, 1.0])  # E/I balance to test
NUM_REPETITIONS = 8                               # Repetitions per condition
```

**No functions** - this is a pure configuration module.

---

## data_utils.py

**Purpose**: MNIST data loading and preprocessing utilities.

### Function: `load_and_preprocess_mnist()`

**Signature:**
```python
def load_and_preprocess_mnist(num_train_max, num_test_max, seed=42):
    """
    Load and preprocess MNIST dataset for reservoir computing tasks.
    
    Parameters
    ----------
    num_train_max : int
        Maximum number of training samples to load
    num_test_max : int
        Maximum number of test samples to load
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    tuple
        (X_train_sample, y_train_onehot, y_train_sample,
         X_test_sample, y_test_onehot, y_test_sample)
    """
```

**What it does:**
1. Downloads MNIST from OpenML (caches for future use)
2. Normalizes images to [0, 1]
3. Binarizes images (pixel > 0.5 → 1, else → 0)
4. Splits into train/test (75%/25%)
5. Samples requested number of examples
6. One-hot encodes labels (e.g., digit 3 → [0,0,0,1,0,0,0,0,0,0])

**Example usage:**
```python
X_train, y_train_oh, y_train, X_test, y_test_oh, y_test = \
    load_and_preprocess_mnist(100, 100, seed=42)
# X_train shape: (100, 784)  - 100 images, 784 pixels each
# y_train_oh shape: (100, 10)  - 100 labels, 10 classes
```

### Function: `calculate_samples_to_reach_threshold()`

**Signature:**
```python
def calculate_samples_to_reach_threshold(learning_curve_dict, target_accuracy, 
                                         training_subsets_sorted):
    """
    Calculate how many training samples are needed to reach a target accuracy.
    
    Parameters
    ----------
    learning_curve_dict : dict
        Dictionary mapping sample sizes to accuracies
    target_accuracy : float
        Target accuracy threshold (e.g., 0.40)
    training_subsets_sorted : list
        Sorted list of training subset sizes tested
    
    Returns
    -------
    float
        Minimum number of samples needed, or np.nan if threshold not reached
    """
```

**What it does:**
- Finds the smallest training set size that reaches target accuracy
- Returns NaN if target never reached

**Example:**
```python
learning_curve = {33: 0.30, 66: 0.45, 100: 0.60}
samples_needed = calculate_samples_to_reach_threshold(learning_curve, 0.40, [33, 66, 100])
# Returns: 66 (first size with accuracy >= 0.40)
```

### Function: `get_accuracy_at_fixed_samples()`

**Signature:**
```python
def get_accuracy_at_fixed_samples(learning_curve_dict, fixed_sample_size):
    """
    Get accuracy at a specific sample size.
    
    Parameters
    ----------
    learning_curve_dict : dict
        Dictionary mapping sample sizes to accuracies
    fixed_sample_size : int
        Sample size to query
    
    Returns
    -------
    float
        Accuracy at the specified sample size, or np.nan if not available
    """
```

**What it does:**
- Simple lookup in learning curve dictionary

**Example:**
```python
accuracy = get_accuracy_at_fixed_samples(learning_curve, 100)
# Returns: 0.60
```

---

## network_model.py

**Purpose**: Neural network equations and construction.

### Function: `get_neuron_equations()`

**Signature:**
```python
def get_neuron_equations():
    """
    Returns the AdEx neuron model differential equations.
    
    Returns
    -------
    tuple
        (equations_string, reset_string)
    """
```

**What it does:**
- Returns Brian2 equation strings for AdEx neuron model
- **Voltage dynamics**: Leak current + exponential spike + synaptic input + noise - adaptation
- **Adaptation dynamics**: Slow current that increases after spikes
- **Noise**: Ornstein-Uhlenbeck process for background fluctuations

**Equations returned:**
```
dV/dt = (I_leak + I_exp - I_syn + I_noise - A + I_stim) / C_mem
dA/dt = (g_A*(V-V_L) - A) / tau_A
dI_noise/dt = -(I_noise-Imid) / tau_noise + sqrt(2/tau_noise) * sigma_noise * xi
```

### Function: `get_synapse_equations()`

**Signature:**
```python
def get_synapse_equations():
    """
    Returns synaptic dynamics equations.
    
    Returns
    -------
    tuple
        (exc_model, inh_model, exc_onpre, inh_onpre)
    """
```

**What it does:**
- Returns Brian2 equations for double-exponential synapses
- Models realistic synaptic rise and decay times
- Separate equations for excitatory (AMPA-like) and inhibitory (GABA-like) synapses

**Dynamics:**
- Each spike causes conductance to rise then decay
- Rise time constant: `tau_r_syn = 5 ms`
- Decay time constant: `tau_d_syn = 50 ms`

### Function: `initialize_neuron_population()`

**Signature:**
```python
def initialize_neuron_population(pop, current_Imid):
    """
    Initialize a neuron population with heterogeneous parameters.
    
    Parameters
    ----------
    pop : NeuronGroup
        The Brian2 neuron group to initialize
    current_Imid : Quantity
        The mid-point current value (with Brian2 units)
    """
```

**What it does:**
- Sets each neuron's parameters from Gaussian distributions
- **Heterogeneous parameters** (vary per neuron):
  - Threshold voltage: `V_thresh = Vt_mean ± Vt_std`
  - Leak conductance: `g_mem = g_mem_mean ± g_mem_std`
  - Adaptation step: `b = b_mean ± b_std`
  - Adaptation time constant: `tau_A = tau_A_mean ± tau_A_std`
- **Fixed parameters** (same for all):
  - Background current: `Imid`
  - Initial voltage near leak potential
  - Zero initial adaptation

**Why heterogeneity matters:**
- Makes network more biologically realistic
- Prevents artificial synchronization
- Improves computational richness

### Function: `create_network()`

**Signature:**
```python
def create_network(N_exc, N_inh, current_Imid, exc_factor, inh_factor, 
                   connection_prob=0.1):
    """
    Create a balanced excitatory-inhibitory network.
    
    Parameters
    ----------
    N_exc : int
        Number of excitatory neurons
    N_inh : int
        Number of inhibitory neurons
    current_Imid : Quantity
        Background input current
    exc_factor : float
        Scaling factor for excitatory synapses
    inh_factor : float
        Scaling factor for inhibitory synapses
    connection_prob : float
        Connection probability
    
    Returns
    -------
    dict
        Dictionary containing all network components:
        - 'Pop_exc': Excitatory NeuronGroup
        - 'Pop_inh': Inhibitory NeuronGroup
        - 'Syn_exc_to_exc': E→E synapses
        - 'Syn_exc_to_inh': E→I synapses
        - 'Syn_inh_to_exc': I→E synapses
        - 'Syn_inh_to_inh': I→I synapses
        - 'synapses': List of all synapse objects
    """
```

**What it does:**
1. Creates excitatory and inhibitory neuron populations
2. Initializes neuron parameters with heterogeneity
3. Creates four synapse groups (E→E, E→I, I→E, I→I)
4. Connects neurons randomly with probability `connection_prob`
5. Sets synaptic weights scaled by `exc_factor` and `inh_factor`

**Network connectivity:**
- **Random sparse**: Each neuron connects to ~10% of others
- **All-to-all connectivity types**: Excitatory and inhibitory neurons can connect to both populations
- **Dale's principle**: Excitatory neurons only make excitatory synapses (and vice versa)

**Example usage:**
```python
network_dict = create_network(
    N_exc=800, N_inh=200,
    current_Imid=0.3*nA,
    exc_factor=1.0, inh_factor=5.0,
    connection_prob=0.1
)
Pop_exc = network_dict['Pop_exc']
Pop_inh = network_dict['Pop_inh']
```

---

## analysis.py

**Purpose**: Analysis functions for neural activity (firing rates, variability, criticality).

### Function: `calculate_average_iei()`

**Signature:**
```python
def calculate_average_iei(spike_monitor, analysis_start_time):
    """
    Calculate average inter-event interval from spike monitor.
    
    Parameters
    ----------
    spike_monitor : SpikeMonitor
        Brian2 spike monitor
    analysis_start_time : Quantity
        Time to start analysis (with Brian2 units)
    
    Returns
    -------
    float or None
        Average IEI in seconds, or None if insufficient data
    """
```

**What it does:**
- Calculates time between successive network-wide spike events
- Used to set adaptive bin width for avalanche detection
- Returns mean time between any two consecutive spikes (across all neurons)

**Example:**
```python
iei = calculate_average_iei(SpikeMon_exc, 0.3*second)
# Returns: 0.023 (seconds) = 23 ms average time between spikes
```

### Function: `calculate_cv()`

**Signature:**
```python
def calculate_cv(spike_monitor, num_neurons, start_time=0*second):
    """
    Calculate coefficient of variation (CV) of inter-spike intervals.
    
    Parameters
    ----------
    spike_monitor : SpikeMonitor
        Brian2 spike monitor
    num_neurons : int
        Total number of neurons
    start_time : Quantity
        Time to start analysis
    
    Returns
    -------
    float
        Mean CV across all neurons
    """
```

**What it does:**
1. For each neuron, calculates its inter-spike intervals (ISIs)
2. For each neuron: CV = std(ISI) / mean(ISI)
3. Returns average CV across all neurons

**Interpretation:**
- **CV < 1**: Regular, clock-like firing (low variability)
- **CV ≈ 1**: Poisson-like, random firing
- **CV > 1**: Irregular, bursty firing (high variability)

**Example:**
```python
cv_value = calculate_cv(SpikeMon_exc, N_exc, start_time=0.3*second)
# Returns: 1.23 (irregular firing)
```

### Function: `calculate_live_cv()`

**Signature:**
```python
def calculate_live_cv(spike_monitor, num_neurons, analysis_start_time,
                     analysis_duration, window_size, step_size,
                     min_spikes_per_neuron_window=3):
    """
    Calculate CV over sliding windows for temporal dynamics.
    
    Parameters
    ----------
    spike_monitor : SpikeMonitor
    num_neurons : int
    analysis_start_time, analysis_duration : Quantity
    window_size, step_size : Quantity
        Sliding window parameters
    min_spikes_per_neuron_window : int
        Minimum spikes required per neuron in window
    
    Returns
    -------
    tuple
        (window_centers_ms, mean_cv_values)
    """
```

**What it does:**
- Calculates CV in sliding time windows
- Shows how firing regularity changes over time
- Useful for detecting transients or dynamic states

**Example:**
```python
times_ms, cvs = calculate_live_cv(
    SpikeMon_exc, N_exc,
    analysis_start_time=0.3*second,
    analysis_duration=20*second,
    window_size=1*second,
    step_size=0.1*second
)
# times_ms: [300, 400, 500, ..., 20000] (window centers in ms)
# cvs: [1.1, 1.2, 1.15, ..., 1.25] (CV in each window)
```

### Function: `calculate_branching_parameter()`

**Signature:**
```python
def calculate_branching_parameter(binned_activity):
    """
    Calculate branching parameter (sigma) from binned activity.
    
    Branching parameter = <n_{t+1}> / <n_t>
    
    Parameters
    ----------
    binned_activity : array
        Array of activity counts in each time bin
    
    Returns
    -------
    float
        Branching parameter (sigma)
    """
```

**What it does:**
- Calculates how activity propagates from one time bin to the next
- **σ = mean(activity[t+1] / activity[t])** for all t where activity[t] > 0

**Interpretation:**
- **σ < 1**: Subcritical - activity dies out
- **σ ≈ 1**: Critical - balanced propagation
- **σ > 1**: Supercritical - activity grows

**Example:**
```python
binned_activity = np.array([5, 6, 5, 7, 6, 5, 0, 0, 3, 4])
sigma = calculate_branching_parameter(binned_activity)
# Calculates: (6/5 + 5/6 + 7/5 + 6/7 + 5/6 + 3/0(skip) + 4/3) / 6
# Returns: ~1.02 (slightly supercritical)
```

### Function: `analyze_bin_width()`

**Signature:**
```python
def analyze_bin_width(timestamps, bin_width_seconds, max_time_seconds):
    """
    Analyze avalanche statistics for a given bin width.
    
    Parameters
    ----------
    timestamps : array
        Array of spike times in seconds
    bin_width_seconds : float
        Width of time bins in seconds
    max_time_seconds : float
        Maximum time for analysis
    
    Returns
    -------
    dict
        Dictionary containing avalanche statistics:
        - num_avalanches
        - mean_size, mean_duration
        - size_alpha, duration_alpha (power-law exponents)
        - gamma (scaling exponent)
        - branching_parameter
        - avalanches (list of avalanche events)
    """
```

**What it does:**
1. Bins spike times into discrete time bins
2. Detects avalanches: continuous periods of activity separated by silent bins
3. Calculates avalanche sizes (total spikes) and durations (number of bins)
4. Fits power-law distributions to sizes and durations
5. Calculates branching parameter
6. Analyzes scaling relationship between size and duration

**Avalanche definition:**
- Activity starts when a bin has spikes
- Continues while consecutive bins have spikes
- Ends when a bin has zero spikes
- Minimum 2 bins to count as avalanche

**Power-law fitting:**
- Uses maximum likelihood estimation (via `powerlaw` package)
- Critical networks show power-law distributions

**Example:**
```python
spike_times = SpikeMon_exc.t[SpikeMon_exc.t > 0.3*second] / second
results = analyze_bin_width(spike_times, bin_width_seconds=0.023, 
                            max_time_seconds=20.0)
# results['num_avalanches']: 1245
# results['size_alpha']: 1.52 (power-law exponent)
# results['branching_parameter']: 0.98
```

### Function: `analyze_model_spikes()`

**Signature:**
```python
def analyze_model_spikes(spike_times_seconds, bin_widths_to_analyze_seconds, 
                        group_name="Model"):
    """
    Analyze spikes from model simulation for avalanche statistics.
    
    Parameters
    ----------
    spike_times_seconds : array
        Array of spike times in seconds
    bin_widths_to_analyze_seconds : list
        List of bin widths to test
    group_name : str
        Name for this analysis group
    
    Returns
    -------
    dict
        Dictionary mapping bin widths to analysis results
    """
```

**What it does:**
- Wrapper function that calls `analyze_bin_width()` for multiple bin widths
- Typically used to test sensitivity to binning

**Example:**
```python
results_dict = analyze_model_spikes(
    spike_times,
    bin_widths=[0.010, 0.023, 0.050],  # Test 10ms, 23ms, 50ms bins
    group_name="Critical_Network"
)
# results_dict[0.023] contains full analysis for 23ms bins
```

---

## reservoir.py

**Purpose**: Reservoir computing functionality for MNIST classification.

### Function: `create_input_projection_map()`

**Signature:**
```python
def create_input_projection_map(n_input_pixels, n_exc_neurons, n_inh_neurons,
                                neurons_per_pixel=5, feed_to_inhibitory=True):
    """
    Create a random projection map from input pixels to neurons.
    
    Parameters
    ----------
    n_input_pixels : int
        Number of input channels (784 for MNIST)
    n_exc_neurons, n_inh_neurons : int
        Number of excitatory and inhibitory neurons
    neurons_per_pixel : int
        How many neurons each pixel projects to
    feed_to_inhibitory : bool
        Whether to project to inhibitory neurons
    
    Returns
    -------
    dict
        Mapping from pixel index to lists of neuron indices
        Format: {pixel_idx: {'exc': [neuron_ids], 'inh': [neuron_ids]}}
    """
```

**What it does:**
- Creates random mapping from each MNIST pixel to neurons
- Each active pixel (value > 0.5) will stimulate specific neurons
- Random projections provide dimensionality expansion (784 → 1000)

**Example:**
```python
projection_map = create_input_projection_map(
    n_input_pixels=784,
    n_exc_neurons=800,
    n_inh_neurons=200,
    neurons_per_pixel=1  # Each pixel targets 1 neuron
)
# projection_map[0] = {'exc': [342], 'inh': [67]}
# → Pixel 0 projects to excitatory neuron 342 and inhibitory neuron 67
```

### Function: `calculate_per_neuron_smoothed_rates()`

**Signature:**
```python
def calculate_per_neuron_smoothed_rates(spike_monitor_t_arr, spike_monitor_i_arr,
                                       num_neurons, target_time_s, 
                                       total_window_width_s, sim_dt_s, 
                                       kernel_std_dev_s):
    """
    Calculate smoothed firing rates for all neurons at a specific time point.
    
    Uses Gaussian smoothing kernel for temporal integration.
    
    Parameters
    ----------
    spike_monitor_t_arr, spike_monitor_i_arr : array
        Spike times and neuron indices
    num_neurons : int
    target_time_s : float
        Target time point for rate calculation
    total_window_width_s : float
        Total width of analysis window
    sim_dt_s : float
        Simulation timestep
    kernel_std_dev_s : float
        Standard deviation of Gaussian kernel (controls smoothing)
    
    Returns
    -------
    array
        Smoothed rates for each neuron at target time (length = num_neurons)
    """
```

**What it does:**
1. Creates time histogram of spikes around target time
2. Applies Gaussian smoothing kernel
3. Extracts firing rate at exact target time for each neuron
4. Returns rate vector (one value per neuron)

**Gaussian smoothing:**
- Broader kernel (larger `kernel_std_dev_s`) → smoother rates
- Narrower kernel → rates reflect instantaneous activity
- Default: `kernel_std_dev_s = 20 ms`

**Why smooth rates?**
- Raw spike counts are noisy
- Smoothed rates better represent underlying neural state
- Improves classifier performance

**Example:**
```python
rates = calculate_per_neuron_smoothed_rates(
    SpikeMon_exc.t/second,
    SpikeMon_exc.i,
    num_neurons=800,
    target_time_s=5.110,  # Read out at this time
    total_window_width_s=0.060,  # ±30ms window
    sim_dt_s=0.0001,
    kernel_std_dev_s=0.020  # 20ms Gaussian
)
# rates shape: (800,)
# rates[0] = 5.2 Hz for neuron 0
# rates[1] = 0.3 Hz for neuron 1
# ...
```

### Function: `run_rc_simulation_for_input()`

**Signature:**
```python
def run_rc_simulation_for_input(network_sim_object, input_img_pattern_flat, 
                                projection_map, pop_exc_group, pop_inh_group, 
                                n_exc_val, n_inh_val, stim_duration, 
                                post_stim_total_duration, mnist_input_current_amp, 
                                sim_dt_brian, spike_mon_exc_obj, spike_mon_inh_obj,
                                trial_internal_settle_time, 
                                readout_snapshot_time_offset):
    """
    Run simulation for a single input pattern and extract reservoir state.
    
    Parameters
    ----------
    network_sim_object : Network
        Brian2 network object
    input_img_pattern_flat : array
        Flattened input image (784 pixels)
    projection_map : dict
        Pixel-to-neuron mapping
    pop_exc_group, pop_inh_group : NeuronGroup
        Neuron populations
    n_exc_val, n_inh_val : int
        Number of neurons
    stim_duration : Quantity
        How long to present stimulus (default: 100ms)
    post_stim_total_duration : Quantity
        Duration after stimulus (default: 10ms)
    mnist_input_current_amp : Quantity
        Input current amplitude (default: 0.2nA)
    sim_dt_brian : Quantity
        Simulation timestep
    spike_mon_exc_obj, spike_mon_inh_obj : SpikeMonitor
        Monitors to record activity
    trial_internal_settle_time : Quantity
        Settling before stimulus (default: 100ms)
    readout_snapshot_time_offset : Quantity
        When to read out state after stimulus (default: 10ms)
    
    Returns
    -------
    array
        Reservoir state vector (concatenated firing rates of all neurons)
        Length: n_exc_val + n_inh_val
    """
```

**What it does:**
1. Resets stimulus currents to zero
2. Lets network settle for `trial_internal_settle_time`
3. Applies input currents based on active pixels (value > 0.5)
4. Runs for `stim_duration`
5. Turns off input
6. Runs for `post_stim_total_duration`
7. Calculates smoothed firing rates at specific readout time
8. Returns concatenated state vector [exc_rates, inh_rates]

**Timeline:**
```
|---settle---|---stimulus---|---post-stim---|
  (100ms)       (100ms)        (10ms)
                                  ↑
                            Read out here (+10ms offset)
```

**Example:**
```python
digit_image = X_train[0]  # Shape: (784,)
state = run_rc_simulation_for_input(
    net, digit_image, projection_map,
    Pop_exc, Pop_inh, 800, 200,
    stim_duration=100*ms,
    post_stim_total_duration=10*ms,
    mnist_input_current_amp=0.2*nA,
    sim_dt_brian=0.1*ms,
    SpikeMon_exc, SpikeMon_inh,
    trial_internal_settle_time=100*ms,
    readout_snapshot_time_offset=10*ms
)
# state shape: (1000,)  [800 exc rates + 200 inh rates]
# state[0] = 3.2 Hz (exc neuron 0)
# state[799] = 1.5 Hz (exc neuron 799)
# state[800] = 8.1 Hz (inh neuron 0)
```

### Function: `train_readout_weights()`

**Signature:**
```python
def train_readout_weights(reservoir_states_matrix_train, 
                         target_outputs_onehot_train, ridge_alpha_val):
    """
    Train linear readout using Ridge regression.
    
    Parameters
    ----------
    reservoir_states_matrix_train : array
        Training reservoir states (n_samples × n_neurons)
    target_outputs_onehot_train : array
        One-hot encoded target labels (n_samples × 10)
    ridge_alpha_val : float
        Ridge regularization parameter (default: 1.0)
    
    Returns
    -------
    array
        Trained readout weight matrix
        Shape: (n_neurons+1, 10)  [+1 for bias term]
    """
```

**What it does:**
1. Adds bias term (column of ones) to state matrix
2. Trains Ridge regression: `W_out = argmin ||X @ W - Y||^2 + alpha ||W||^2`
3. Returns weight matrix

**Ridge regression:**
- Linear classifier with L2 regularization
- Prevents overfitting to small training sets
- `alpha` controls regularization strength

**Example:**
```python
X_train_states shape: (100, 1000)  # 100 images, 1000 neuron states
y_train_onehot shape: (100, 10)    # 100 labels, 10 classes

W_out = train_readout_weights(X_train_states, y_train_onehot, ridge_alpha_val=1.0)
# W_out shape: (1001, 10)
# To classify: predictions = (X_test @ W_out).argmax(axis=1)
```

### Function: `evaluate_readout_performance()`

**Signature:**
```python
def evaluate_readout_performance(reservoir_states_matrix_test, 
                                trained_weights_W_out, original_labels_test):
    """
    Evaluate readout performance on test data.
    
    Parameters
    ----------
    reservoir_states_matrix_test : array
        Test reservoir states (n_test_samples × n_neurons)
    trained_weights_W_out : array
        Trained readout weights
    original_labels_test : array
        True labels for test data
    
    Returns
    -------
    tuple
        (accuracy, predicted_labels)
    """
```

**What it does:**
1. Adds bias term to test states
2. Computes predictions: `output = X_test @ W_out`
3. Takes argmax to get predicted class
4. Compares to true labels
5. Returns accuracy and predictions

**Example:**
```python
accuracy, predictions = evaluate_readout_performance(
    X_test_states,  # (100, 1000)
    W_out,          # (1001, 10)
    y_test          # (100,)
)
# accuracy: 0.56 (56% correct)
# predictions: [3, 5, 1, 0, ..., 7]  (predicted digit for each image)
```

---

## plotting.py

**Purpose**: All visualization functions.

### Key Functions Summary

#### `plot_basic_activity()`
- **What**: Comprehensive 6-panel plot showing network dynamics
- **Panels**: Raster, voltage traces, adaptation, live CV, population rates
- **Output**: Saved to `results_phase_diagram_runs/{condition}/basic_activity_plot.png`

#### `plot_initial_raster()`
- **What**: Detailed raster plot of first 5 seconds
- **Shows**: Every spike from every neuron
- **Useful for**: Seeing fine-grained spiking patterns

#### `plot_detailed_stimulus_raster()`
- **What**: Raster during MNIST presentation trials
- **Shows**: Highlighted spikes from stimulated neurons
- **Labels**: Digit being presented for each trial

#### `plot_neural_manifold()`
- **What**: PCA visualization of reservoir states
- **Shows**: How different digits cluster in neural activity space
- **Output**: 3D scatter plot colored by digit class

#### `plot_all_learning_accuracy_curves()`
- **What**: Learning curves across all conditions
- **Shows**: Accuracy vs. training set size with error bars
- **Compares**: Subcritical, Critical, Supercritical networks

#### `generate_summary_plots()`
- **What**: Phase diagrams (heatmaps or line plots)
- **Shows**: How metrics vary across parameter space
- **Metrics**: Firing rate, CV, branching parameter, RC accuracy, etc.

---

## statistics.py

**Purpose**: Statistical hypothesis testing.

### Function: `run_and_print_statistical_tests()`

**What it does:**
1. **Checks assumptions** for parametric tests:
   - Shapiro-Wilk test for normality
   - Levene's test for homogeneity of variance
2. **Chooses appropriate test**:
   - If assumptions met: **ANOVA** (parametric)
   - If assumptions violated: **Kruskal-Wallis** (non-parametric)
3. **If significant, runs post-hoc tests**:
   - ANOVA → **Tukey's HSD** (all pairwise comparisons)
   - Kruskal-Wallis → **Mann-Whitney U** with Bonferroni correction

**Example output:**
```
Using parametric test: One-Way ANOVA
 - F-statistic: 5.67
 - P-value: 0.012
 - Result: **Significant difference detected**

Post-Hoc Test: Tukey's HSD
   group1       group2    meandiff p-adj  reject
   Critical   Subcritical   0.123  0.023   True
```

### Function: `run_learning_curve_statistics()`

**What it does:**
- Performs ANOVA at each training set size
- Identifies when conditions start to diverge
- Shows which sample sizes have significant differences

---

## main_simulation.py

**Purpose**: Orchestrates the entire simulation workflow.

### Main Workflow

1. **Initialize** data storage arrays
2. **Load MNIST** dataset
3. **For each condition**:
   - Set random seed
   - Create network
   - Run intrinsic dynamics (20s)
   - Analyze activity (FR, CV, avalanches)
   - Run RC task (present images, train readout)
   - Generate plots
   - Store results
4. **Aggregate results** across conditions
5. **Generate summary plots**
6. **Run statistical tests**
7. **Save to Excel**

### Console Output Structure

The main simulation prints detailed progress information (see CONSOLE_OUTPUT_GUIDE.md).

---

## Summary: What Each Module Does

| Module | Primary Purpose | Key Output |
|--------|----------------|------------|
| **config.py** | Parameter storage | Configuration values |
| **data_utils.py** | MNIST handling | Preprocessed images & labels |
| **network_model.py** | Network construction | Brian2 network object |
| **analysis.py** | Activity analysis | CV, firing rates, avalanches, σ |
| **reservoir.py** | MNIST classification | RC accuracy, state vectors |
| **plotting.py** | Visualization | PNG figures |
| **statistics.py** | Hypothesis testing | Statistical test results |
| **main_simulation.py** | Orchestration | Excel file, organized results |

---

## Quick Reference: Common Operations

### Create and run a network
```python
from network_model import create_network
from brian2 import Network, second

net_dict = create_network(800, 200, 0.3*nA, 1.0, 5.0)
net = Network(collect())
net.run(20*second)
```

### Analyze activity
```python
from analysis import calculate_cv, analyze_model_spikes

cv = calculate_cv(SpikeMon_exc, 800)
spike_times = SpikeMon_exc.t[SpikeMon_exc.t > 0.3*second] / second
avalanche_results = analyze_model_spikes(spike_times, [0.023])
```

### Run RC task
```python
from reservoir import create_input_projection_map, run_rc_simulation_for_input, train_readout_weights

projection_map = create_input_projection_map(784, 800, 200)
states = [run_rc_simulation_for_input(net, img, projection_map, ...) for img in X_train]
W_out = train_readout_weights(np.array(states), y_train_onehot, 1.0)
accuracy, preds = evaluate_readout_performance(test_states, W_out, y_test)
```
