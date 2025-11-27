"""
Reservoir Computing (RC) specific functions
Handles MNIST classification task using the spiking network as a reservoir
"""

import numpy as np
from brian2 import *
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score


def create_input_projection_map(n_input_pixels, n_exc_neurons, n_inh_neurons, 
                                  neurons_per_pixel=5, feed_to_inhibitory=True):
    """
    Create a random projection map from input pixels to neurons.

    Parameters
    ----------
    n_input_pixels : int
        Number of input channels (e.g., 784 for MNIST)
    n_exc_neurons : int
        Number of excitatory neurons
    n_inh_neurons : int
        Number of inhibitory neurons
    neurons_per_pixel : int
        How many neurons each pixel projects to
    feed_to_inhibitory : bool
        Whether to project to inhibitory neurons

    Returns
    -------
    dict
        Mapping from pixel index to lists of excitatory and inhibitory neuron indices
    """
    projection_map = {}

    for pixel_idx in range(n_input_pixels):
        # Random excitatory targets
        exc_targets = np.random.choice(n_exc_neurons, neurons_per_pixel, replace=False)

        # Random inhibitory targets if enabled
        if feed_to_inhibitory:
            inh_targets = np.random.choice(n_inh_neurons, neurons_per_pixel, replace=False)
        else:
            inh_targets = []

        projection_map[pixel_idx] = {
            'exc': list(exc_targets),
            'inh': list(inh_targets)
        }

    return projection_map


def calculate_per_neuron_smoothed_rates(spike_monitor_t_arr, spike_monitor_i_arr, 
                                        num_neurons, target_time_s, total_window_width_s,
                                        sim_dt_s, kernel_std_dev_s):
    """
    Calculate smoothed firing rates for all neurons at a specific time point.
    Uses Gaussian smoothing kernel.

    Parameters
    ----------
    spike_monitor_t_arr : array
        Spike times in seconds
    spike_monitor_i_arr : array
        Neuron indices for each spike
    num_neurons : int
        Total number of neurons
    target_time_s : float
        Target time point for rate calculation
    total_window_width_s : float
        Total width of analysis window
    sim_dt_s : float
        Simulation timestep
    kernel_std_dev_s : float
        Standard deviation of Gaussian kernel

    Returns
    -------
    array
        Smoothed rates for each neuron at target time
    """
    rates = np.zeros(num_neurons)

    if sim_dt_s <= 0 or kernel_std_dev_s <= 0 or total_window_width_s <= 0:
        return rates

    if len(spike_monitor_t_arr) == 0:
        return rates

    kernel_half_pts = int(np.ceil(total_window_width_s / (2 * sim_dt_s)))
    if kernel_half_pts <= 0:
        return rates

    # Create Gaussian kernel
    kernel_time_rel = np.arange(-kernel_half_pts, kernel_half_pts + 1) * sim_dt_s
    gaussian_kernel = (1.0 / (kernel_std_dev_s * np.sqrt(2 * np.pi))) * \
                      np.exp(-kernel_time_rel**2 / (2 * kernel_std_dev_s**2))
    gaussian_kernel *= sim_dt_s

    # Define histogram window
    hist_start_time_s = target_time_s - total_window_width_s / 2.0
    hist_end_time_s = target_time_s + total_window_width_s / 2.0 + sim_dt_s
    num_hist_bins = int(round((hist_end_time_s - hist_start_time_s) / sim_dt_s))

    if num_hist_bins <= 0:
        return rates

    hist_bin_edges = np.linspace(hist_start_time_s, hist_end_time_s, num_hist_bins + 1)

    # Get relevant spikes
    analysis_window_start = hist_start_time_s - total_window_width_s
    analysis_window_end = hist_end_time_s + total_window_width_s

    relevant_spike_mask = (spike_monitor_t_arr >= analysis_window_start) & \
                          (spike_monitor_t_arr <= analysis_window_end)
    relevant_spike_times = spike_monitor_t_arr[relevant_spike_mask]
    relevant_spike_indices = spike_monitor_i_arr[relevant_spike_mask]

    if len(relevant_spike_times) == 0:
        return rates

    # Calculate smoothed rate for each neuron
    for i in range(num_neurons):
        neuron_spike_times = relevant_spike_times[relevant_spike_indices == i]

        if len(neuron_spike_times) > 0:
            spike_counts, _ = np.histogram(neuron_spike_times, bins=hist_bin_edges)
            binned_rates_hz = spike_counts / sim_dt_s
            smoothed_rate_profile = np.convolve(binned_rates_hz, gaussian_kernel, mode='same')

            center_bin_index = num_hist_bins // 2
            if 0 <= center_bin_index < len(smoothed_rate_profile):
                rates[i] = smoothed_rate_profile[center_bin_index]

    return rates


def run_rc_simulation_for_input(network_sim_object, input_img_pattern_flat, projection_map,
                                pop_exc_group, pop_inh_group, n_exc_val, n_inh_val,
                                stim_duration, post_stim_total_duration, 
                                mnist_input_current_amp, sim_dt_brian,
                                spike_mon_exc_obj, spike_mon_inh_obj,
                                trial_internal_settle_time, readout_snapshot_time_offset):
    """
    Run simulation for a single input pattern and extract reservoir state.

    Parameters
    ----------
    network_sim_object : Network
        Brian2 network object
    input_img_pattern_flat : array
        Flattened input image (784 pixels for MNIST)
    projection_map : dict
        Mapping from pixels to neurons
    pop_exc_group : NeuronGroup
        Excitatory neurons
    pop_inh_group : NeuronGroup
        Inhibitory neurons
    n_exc_val : int
        Number of excitatory neurons
    n_inh_val : int
        Number of inhibitory neurons
    stim_duration : Quantity
        Duration to present stimulus
    post_stim_total_duration : Quantity
        Duration after stimulus
    mnist_input_current_amp : Quantity
        Input current amplitude
    sim_dt_brian : Quantity
        Simulation timestep
    spike_mon_exc_obj : SpikeMonitor
        Excitatory spike monitor
    spike_mon_inh_obj : SpikeMonitor
        Inhibitory spike monitor
    trial_internal_settle_time : Quantity
        Settling time before stimulus
    readout_snapshot_time_offset : Quantity
        When to read out state after stimulus

    Returns
    -------
    array
        Reservoir state vector (concatenated firing rates)
    """
    from config import (RC_STATE_SMOOTHING_WIDTH_STD_DEV, 
                       RC_STATE_RATE_CALC_WINDOW_DURATION,
                       FEED_INPUT_TO_INHIBITORY, V_syn_exc, V_syn_inh, Vr, C_mem,tau_noise, sigma_noise, g_mem_val, V_L, g_A, D_T,
                       tau_r_syn, tau_d_syn, base_g_syn_max_exc_value, base_g_syn_max_inh_value,)

    # Reset stimulus and settle
    pop_exc_group.I_stim = 0 * nA
    pop_inh_group.I_stim = 0 * nA
    run_namespace = {
        'tau_noise': tau_noise,
        'sigma_noise': sigma_noise,
        'g_mem_val': g_mem_val,
        'V_L': V_L,
        'g_A': g_A,
        'D_T': D_T,
        'V_syn_exc': V_syn_exc,
        'V_syn_inh': V_syn_inh,
        'Vr': Vr,
        'C_mem': C_mem,'tau_r_syn': tau_r_syn,
        'tau_d_syn': tau_d_syn,
        'base_g_syn_max_exc_value': base_g_syn_max_exc_value,
        'base_g_syn_max_inh_value': base_g_syn_max_inh_value,
    }
    if trial_internal_settle_time > 0 * ms:
        network_sim_object.run(trial_internal_settle_time, report=None,namespace=run_namespace)

    time_before_stim = network_sim_object.t

    # Apply stimulus current
    stim_current_exc = np.zeros(n_exc_val) * nA
    stim_current_inh = np.zeros(n_inh_val) * nA

    for pixel_idx, pixel_value in enumerate(input_img_pattern_flat):
        if pixel_value > 0.5:
            exc_targets = projection_map[pixel_idx]['exc']
            inh_targets = projection_map[pixel_idx]['inh']

            if exc_targets:
                stim_current_exc[exc_targets] += mnist_input_current_amp

            if FEED_INPUT_TO_INHIBITORY and inh_targets:
                stim_current_inh[inh_targets] += mnist_input_current_amp

    pop_exc_group.I_stim = stim_current_exc
    pop_inh_group.I_stim = stim_current_inh
    network_sim_object.run(stim_duration, report=None,namespace=run_namespace)

    # Turn off stimulus
    pop_exc_group.I_stim = 0 * nA
    pop_inh_group.I_stim = 0 * nA
    network_sim_object.run(post_stim_total_duration, report=None,namespace=run_namespace)

    # Extract reservoir state at readout time
    readout_target_time_quantity = time_before_stim + stim_duration + readout_snapshot_time_offset
    readout_target_time_s = readout_target_time_quantity / second

    # Get spike data
    all_spike_times_exc = spike_mon_exc_obj.t / second
    all_spike_indices_exc = spike_mon_exc_obj.i
    all_spike_times_inh = spike_mon_inh_obj.t / second
    all_spike_indices_inh = spike_mon_inh_obj.i

    # Calculate rates at readout time
    rates_exc = calculate_per_neuron_smoothed_rates(
        all_spike_times_exc, all_spike_indices_exc, n_exc_val,
        readout_target_time_s,
        RC_STATE_RATE_CALC_WINDOW_DURATION / second, 
        sim_dt_brian / second,
        RC_STATE_SMOOTHING_WIDTH_STD_DEV / second
    )

    rates_inh = calculate_per_neuron_smoothed_rates(
        all_spike_times_inh, all_spike_indices_inh, n_inh_val,
        readout_target_time_s,
        RC_STATE_RATE_CALC_WINDOW_DURATION / second,
        sim_dt_brian / second,
        RC_STATE_SMOOTHING_WIDTH_STD_DEV / second
    )

    # Concatenate rates to form state vector
    current_reservoir_state = np.concatenate((rates_exc, rates_inh))

    return current_reservoir_state


def train_readout_weights(reservoir_states_matrix_train, target_outputs_onehot_train, 
                          ridge_alpha_val):
    """
    Train linear readout using Ridge regression.

    Parameters
    ----------
    reservoir_states_matrix_train : array
        Training reservoir states (n_samples × n_neurons)
    target_outputs_onehot_train : array
        One-hot encoded target labels
    ridge_alpha_val : float
        Ridge regularization parameter

    Returns
    -------
    array
        Trained readout weight matrix
    """
    print(f"Training readout with {reservoir_states_matrix_train.shape[0]} samples.")
    print(f"Reservoir state dim: {reservoir_states_matrix_train.shape[1]}")

    # Add bias term
    X_train_with_bias = np.hstack([
        reservoir_states_matrix_train,
        np.ones((reservoir_states_matrix_train.shape[0], 1))
    ])

    # Train Ridge regression
    ridge_regression_model = Ridge(alpha=ridge_alpha_val, fit_intercept=False)
    ridge_regression_model.fit(X_train_with_bias, target_outputs_onehot_train)

    W_out = ridge_regression_model.coef_.T
    print(f"Readout weights W_out shape: {W_out.shape}")

    return W_out


def evaluate_readout_performance(reservoir_states_matrix_test, trained_weights_W_out, 
                                 original_labels_test):
    """
    Evaluate readout performance on test data.

    Parameters
    ----------
    reservoir_states_matrix_test : array
        Test reservoir states
    trained_weights_W_out : array
        Trained readout weights
    original_labels_test : array
        True labels for test data

    Returns
    -------
    tuple
        (accuracy, predicted_labels)
    """
    # Add bias term
    X_test_with_bias = np.hstack([
        reservoir_states_matrix_test,
        np.ones((reservoir_states_matrix_test.shape[0], 1))
    ])

    # Make predictions
    predicted_outputs_onehot_format = X_test_with_bias @ trained_weights_W_out
    predicted_class_labels = np.argmax(predicted_outputs_onehot_format, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(original_labels_test, predicted_class_labels)

    return accuracy, predicted_class_labels
