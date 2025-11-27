"""
Analysis functions for neural activity
Includes firing rate, CV, avalanche, and criticality measures
"""

import numpy as np
import pandas as pd
from scipy import stats
import powerlaw as pl
from brian2 import second


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
    if not hasattr(spike_monitor, 't') or len(spike_monitor.t) == 0:
        return None

    relevant_spike_times_brian = spike_monitor.t[spike_monitor.t >= analysis_start_time]
    if len(relevant_spike_times_brian) < 2:
        return None

    relevant_spike_times_seconds = np.sort(np.array(relevant_spike_times_brian / second))
    ieis = np.diff(relevant_spike_times_seconds)

    if len(ieis) == 0:
        return None

    mean_iei = np.mean(ieis)
    if mean_iei <= 1e-6:
        return None

    return mean_iei


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
    all_cvs = []

    if not hasattr(spike_monitor, 't') or len(spike_monitor.t) == 0:
        return np.nan

    spike_times_t = spike_monitor.t
    spike_times_i = spike_monitor.i

    for neuron_idx in range(num_neurons):
        neuron_spike_times = spike_times_t[
            (spike_times_i == neuron_idx) & (spike_times_t >= start_time)
        ]

        if len(neuron_spike_times) >= 2:
            neuron_isis = np.diff(neuron_spike_times / second)

            if len(neuron_isis) >= 1:
                mean_isi = np.mean(neuron_isis)
                std_isi = np.std(neuron_isis)

                if mean_isi > 1e-12:
                    cv = std_isi / mean_isi
                    if np.isfinite(cv):
                        all_cvs.append(cv)

    if not all_cvs:
        return np.nan
    else:
        return np.nanmean(all_cvs)


def calculate_live_cv(spike_monitor, num_neurons, analysis_start_time, 
                      analysis_duration, window_size, step_size, 
                      min_spikes_per_neuron_window=3):
    """
    Calculate CV over sliding windows for temporal dynamics.

    Parameters
    ----------
    spike_monitor : SpikeMonitor
        Brian2 spike monitor
    num_neurons : int
        Number of neurons
    analysis_start_time : Quantity
        Start time for analysis
    analysis_duration : Quantity
        Duration of analysis period
    window_size : Quantity
        Size of sliding window
    step_size : Quantity
        Step size for sliding window
    min_spikes_per_neuron_window : int
        Minimum spikes required per neuron in window

    Returns
    -------
    tuple
        (window_centers_ms, mean_cv_values)
    """
    if not hasattr(spike_monitor, 't') or len(spike_monitor.t) == 0:
        return np.array([]), np.array([])

    spike_times_s = spike_monitor.t / second
    spike_indices = spike_monitor.i

    analysis_start_s = analysis_start_time / second
    analysis_end_s = analysis_start_s + (analysis_duration / second)
    window_size_s = window_size / second
    step_size_s = step_size / second

    if analysis_end_s < analysis_start_s + window_size_s:
        return np.array([]), np.array([])

    window_starts = np.arange(
        analysis_start_s, 
        analysis_end_s - window_size_s + step_size_s, 
        step_size_s
    )

    if len(window_starts) == 0 and analysis_end_s >= analysis_start_s + window_size_s:
        window_starts = np.array([analysis_start_s])

    mean_cv_values = []
    window_centers_s = []

    for win_start_s in window_starts:
        win_end_s = win_start_s + window_size_s
        window_centers_s.append(win_start_s + window_size_s / 2.0)

        cvs_in_window = []
        for neuron_idx in range(num_neurons):
            neuron_mask = (spike_indices == neuron_idx)
            neuron_spike_times_s = spike_times_s[neuron_mask]

            spikes_in_current_window = neuron_spike_times_s[
                (neuron_spike_times_s >= win_start_s) & 
                (neuron_spike_times_s < win_end_s)
            ]

            if len(spikes_in_current_window) >= min_spikes_per_neuron_window:
                isis = np.diff(spikes_in_current_window)
                if len(isis) > 0:
                    mean_isi = np.mean(isis)
                    std_isi = np.std(isis)
                    if mean_isi > 1e-12:
                        cv = std_isi / mean_isi
                        if np.isfinite(cv):
                            cvs_in_window.append(cv)

        if cvs_in_window:
            mean_cv_values.append(np.nanmean(cvs_in_window))
        else:
            mean_cv_values.append(np.nan)

    return np.array(window_centers_s) * 1000, np.array(mean_cv_values)


def calculate_branching_parameter(binned_activity):
    """
    Calculate branching parameter (sigma) from binned activity.
    Branching parameter = <n_{t+1}> / <n_t> where n_t is activity at time t

    Parameters
    ----------
    binned_activity : array
        Array of activity counts in each time bin

    Returns
    -------
    float
        Branching parameter (sigma)
    """
    try:
        if not isinstance(binned_activity, np.ndarray) or len(binned_activity) < 2:
            return np.nan

        ratios = []
        for i in range(len(binned_activity) - 1):
            ancestor = binned_activity[i]
            descendant = binned_activity[i+1]
            if ancestor > 0:
                ratios.append(descendant / ancestor)

        if not ratios:
            return np.nan

        valid_ratios = np.array(ratios)[np.isfinite(ratios)]
        if len(valid_ratios) == 0:
            return np.nan

        return np.mean(valid_ratios)

    except Exception:
        return np.nan


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
        Dictionary containing avalanche statistics including:
        - num_avalanches
        - mean_size, mean_duration
        - size_alpha, duration_alpha (power-law exponents)
        - gamma (scaling exponent)
        - branching_parameter
    """
    try:
        hist = np.array([])

        if bin_width_seconds <= 0:
            pass
        elif max_time_seconds < 0:
            max_time_seconds = 0

        bins = np.array([])
        if bin_width_seconds > 0:
            bins = np.arange(0, max_time_seconds + bin_width_seconds, bin_width_seconds)

        if len(timestamps) == 0 or np.isnan(timestamps).any() or np.isinf(timestamps).any():
            pass
        elif bin_width_seconds > 0 and bins.size > 0:
            valid_timestamps = timestamps[np.isfinite(timestamps)]
            if len(valid_timestamps) > 0:
                actual_max_timestamp = np.max(valid_timestamps)
                if bins[-1] < actual_max_timestamp:
                    bins = np.arange(
                        bins[0], 
                        actual_max_timestamp + bin_width_seconds, 
                        bin_width_seconds
                    )
                if bins.size > 0:
                    hist, _ = np.histogram(timestamps, bins=bins)

        # Identify avalanches
        avalanches = []
        if hist.size > 0 and not (np.isnan(hist).any() or np.isinf(hist).any()):
            current_avalanche = []
            in_avalanche = False

            for spike_count in hist:
                if spike_count > 0:
                    current_avalanche.append(spike_count)
                    in_avalanche = True
                elif in_avalanche:
                    if len(current_avalanche) >= 2:
                        avalanches.append(current_avalanche)
                    current_avalanche = []
                    in_avalanche = False

            if in_avalanche and len(current_avalanche) >= 2:
                avalanches.append(current_avalanche)

        # Calculate branching parameter
        branching_param_val = np.nan
        if isinstance(hist, np.ndarray) and hist.size > 0:
            if not (np.isnan(hist).any() or np.isinf(hist).any()):
                branching_param_val = calculate_branching_parameter(hist)

        # Return early if no avalanches
        if not avalanches:
            return {
                'num_avalanches': 0,
                'mean_size': np.nan,
                'mean_duration': np.nan,
                'size_alpha': np.nan,
                'duration_alpha': np.nan,
                'gamma': np.nan,
                'intercept': np.nan,
                'r_value': np.nan,
                'p_value': np.nan,
                'std_err': np.nan,
                'size_fit': None,
                'duration_fit': None,
                'duration_groups': pd.DataFrame(),
                'log_durations': np.array([]),
                'log_sizes': np.array([]),
                'avalanches': [],
                'branching_parameter': branching_param_val
            }

        # Calculate avalanche properties
        sizes = [sum(av) for av in avalanches]
        durations = [len(av) for av in avalanches]

        # Fit power laws
        size_alpha, duration_alpha, size_fit, duration_fit = np.nan, np.nan, None, None

        if len(sizes) > 10:
            try:
                valid_sizes = [s for s in sizes if s > 0]
                if len(valid_sizes) > 10:
                    size_fit = pl.Fit(valid_sizes, discrete=True, xmin=None, verbose=False)
                    if size_fit and hasattr(size_fit, 'power_law') and size_fit.power_law:
                        size_alpha = size_fit.power_law.alpha
            except Exception:
                size_fit = None

        if len(durations) > 10:
            try:
                valid_durations = [d for d in durations if d > 0]
                if len(valid_durations) > 10:
                    duration_fit = pl.Fit(valid_durations, discrete=True, xmin=None, verbose=False)
                    if duration_fit and hasattr(duration_fit, 'power_law') and duration_fit.power_law:
                        duration_alpha = duration_fit.power_law.alpha
            except Exception:
                duration_fit = None

        # Calculate scaling relationship
        slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan
        log_durations, log_sizes = np.array([]), np.array([])

        df_duration_groups = pd.DataFrame({'duration': durations, 'size': sizes})

        if not df_duration_groups.empty:
            df_duration_groups['duration'] = pd.to_numeric(df_duration_groups['duration'], errors='coerce')
            df_duration_groups['size'] = pd.to_numeric(df_duration_groups['size'], errors='coerce')
            df_duration_groups.dropna(subset=['duration', 'size'], inplace=True)

            if not df_duration_groups.empty:
                df_duration_groups = df_duration_groups.groupby('duration')['size'].mean().reset_index()

                if len(df_duration_groups) >= 2:
                    valid_data = df_duration_groups[
                        (df_duration_groups['duration'] > 0) & 
                        (df_duration_groups['size'] > 0)
                    ]

                    if len(valid_data) >= 2:
                        log_durations = np.log10(valid_data['duration'].values)
                        log_sizes = np.log10(valid_data['size'].values)

                        if len(log_durations) >= 2:
                            if not np.isnan(log_durations).any() and not np.isnan(log_sizes).any():
                                slope, intercept, r_value, p_value, std_err = stats.linregress(
                                    log_durations, log_sizes
                                )

        return {
            'num_avalanches': len(avalanches),
            'mean_size': np.mean(sizes) if sizes else np.nan,
            'mean_duration': np.mean(durations) if durations else np.nan,
            'size_alpha': size_alpha,
            'duration_alpha': duration_alpha,
            'gamma': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err,
            'size_fit': size_fit,
            'duration_fit': duration_fit,
            'duration_groups': df_duration_groups,
            'log_durations': log_durations,
            'log_sizes': log_sizes,
            'avalanches': avalanches,
            'branching_parameter': branching_param_val
        }

    except Exception as e:
        print(f"Error analyzing bin_width={bin_width_seconds}: {str(e)}")
        return {
            'num_avalanches': 0,
            'mean_size': np.nan,
            'mean_duration': np.nan,
            'size_alpha': np.nan,
            'duration_alpha': np.nan,
            'gamma': np.nan,
            'intercept': np.nan,
            'r_value': np.nan,
            'p_value': np.nan,
            'std_err': np.nan,
            'size_fit': None,
            'duration_fit': None,
            'duration_groups': pd.DataFrame(),
            'log_durations': np.array([]),
            'log_sizes': np.array([]),
            'avalanches': [],
            'branching_parameter': np.nan
        }


def analyze_model_spikes(spike_times_seconds, bin_widths_to_analyze_seconds, group_name="Model"):
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
    simulation_results_keys_seconds = {}
    spike_times_seconds = np.sort(spike_times_seconds)

    if len(spike_times_seconds) == 0:
        for bw_s in bin_widths_to_analyze_seconds:
            simulation_results_keys_seconds[bw_s] = analyze_bin_width(np.array([]), bw_s, 0)
        return simulation_results_keys_seconds

    max_spike_time_s = spike_times_seconds[-1] if len(spike_times_seconds) > 0 else 0
    analysis_max_hist_time_s = max_spike_time_s

    unique_sorted_bws_seconds = sorted(list(set(
        bw_s for bw_s in bin_widths_to_analyze_seconds if bw_s > 0
    )))

    for bw_s in unique_sorted_bws_seconds:
        analysis_data = analyze_bin_width(spike_times_seconds, bw_s, analysis_max_hist_time_s)
        simulation_results_keys_seconds[bw_s] = analysis_data

    return simulation_results_keys_seconds
