"""
Main simulation script
Orchestrates parameter sweeps, network creation, simulation, analysis, and output
"""

from brian2 import *
import brian2
import numpy as np
import pandas as pd
import os
import traceback

# Import all our custom modules
from config import *
from network_model import create_network
from data_utils import (
    load_and_preprocess_mnist,
    calculate_samples_to_reach_threshold,
    get_accuracy_at_fixed_samples
)
from analysis import (
    calculate_cv,
    calculate_average_iei,
    calculate_live_cv,
    analyze_model_spikes
)
from reservoir import (
    create_input_projection_map,
    run_rc_simulation_for_input,
    train_readout_weights,
    evaluate_readout_performance
)
from plotting import *
from statistics import *

# Set random seeds for reproducibility
numpy.random.seed(SEED)
brian2.seed(SEED)

print("=" * 80)
print(" " * 15 + "NEURAL NETWORK SIMULATION - PARAMETER SWEEP")
print("=" * 80)
print(f"Configuration: {len(Imid_values_nA)} Imid values × {len(EI_ratio_values)} E/I ratios × {NUM_REPETITIONS} repetitions")
print(f"Total simulations: {len(Imid_values_nA) * len(EI_ratio_values) * NUM_REPETITIONS}")
print("=" * 80)

# ============================================================================
# INITIALIZATION - Data Storage Arrays
# ============================================================================

print("\nInitializing data storage arrays...")

# Create arrays to store results
phase_diagram_results_firing_rate = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
phase_diagram_results_overall_cv = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
phase_diagram_results_sigma = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
phase_diagram_results_size_alpha = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
phase_diagram_results_duration_alpha = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
phase_diagram_results_gamma = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
phase_diagram_results_rc_accuracy = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
phase_diagram_samples_to_threshold = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)
phase_diagram_accuracy_at_fixed_samples = np.full((len(Imid_values_nA), len(EI_ratio_values), NUM_REPETITIONS), np.nan)

# Storage for learning curves
phase_diagram_learning_curves_data = [
    [[{} for _ in range(NUM_REPETITIONS)] for _ in range(len(EI_ratio_values))]
    for _ in range(len(Imid_values_nA))
]

# Storage for avalanche distributions
all_avalanche_distributions = []

# Storage for Excel export
all_runs_results_list = []

print("✓ Data structures initialized")

# ============================================================================
# LOAD MNIST DATA
# ============================================================================

print("\nLoading MNIST dataset...")

X_train_global, y_train_onehot_global, y_train_labels_global, \
X_test_global, y_test_onehot_global, y_test_labels_global = \
    load_and_preprocess_mnist(NUM_TRAIN_SAMPLES_MAX, NUM_TEST_SAMPLES_MAX, seed=SEED)

NUM_TRAIN_SAMPLES_EFFECTIVE = len(X_train_global)
NUM_TEST_SAMPLES_EFFECTIVE = len(X_test_global)

print(f"✓ MNIST loaded: {NUM_TRAIN_SAMPLES_EFFECTIVE} train / {NUM_TEST_SAMPLES_EFFECTIVE} test samples")


# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================

print("\n" + "=" * 80)
print(" " * 20 + "STARTING PARAMETER SWEEP")
print("=" * 80)

total_runs = len(Imid_values_nA) * len(EI_ratio_values) * NUM_REPETITIONS
current_run_counter = 0

for i_idx, imid_val_nA in enumerate(Imid_values_nA):
    for j_idx, ei_ratio in enumerate(EI_ratio_values):
        for rep_idx in range(NUM_REPETITIONS):
            current_run_counter += 1

            # Set seed for this repetition
            current_seed = SEED + rep_idx
            numpy.random.seed(current_seed)
            brian2.seed(current_seed)
            start_scope()

            # Current parameters
            current_Imid = imid_val_nA * nA

            # Skip invalid configurations
            if ei_ratio <= 1e-6:
                print(f"Warning: Skipping E/I ratio <= 0: {ei_ratio}")
                continue

            # Calculate synaptic factors
            current_exc_factor = EXC_FACTOR_FIXED
            current_inh_factor = (base_g_syn_max_exc_value * current_exc_factor) / \
                                 (base_g_syn_max_inh_value * ei_ratio)

            if current_inh_factor <= 1e-6:
                print(f"Warning: Calculated inh_factor {current_inh_factor:.4f} <= 0. Skipping.")
                continue

            # Condition name
            condition_name_str = condition_map.get(ei_ratio, f"EI_{ei_ratio:.3f}")
            param_combo_str = f"{condition_name_str.capitalize()}_Rep{rep_idx}"

            print(f"\n{'='*80}")
            print(f"RUN {current_run_counter}/{total_runs}: {param_combo_str}")
            print(f"  Imid={imid_val_nA:.3f} nA, E/I ratio={ei_ratio:.3f}")
            print(f"{'='*80}")

            try:
                # ============================================================
                # NETWORK CREATION
                # ============================================================

                N_total_neurons_for_sim = N_TOTAL_NEURONS
                N_exc = int(0.8 * N_total_neurons_for_sim)
                N_inh = N_total_neurons_for_sim - N_exc

                print(f"Creating network: {N_exc} Exc + {N_inh} Inh = {N_total_neurons_for_sim} total")

                # Create network using our modular function
                network_dict = create_network(
                    N_exc, N_inh,
                    current_Imid,
                    current_exc_factor, current_inh_factor,
                    P_MAX
                )

                Pop_exc = network_dict['Pop_exc']
                Pop_inh = network_dict['Pop_inh']
                all_synapses = network_dict['synapses']

                # ============================================================
                # MONITORS
                # ============================================================

                SpikeMon_exc = SpikeMonitor(Pop_exc)
                SpikeMon_inh = SpikeMonitor(Pop_inh)
                StateMon_exc = StateMonitor(Pop_exc, variables=['V', 'A'], record=0, dt=1*ms)
                StateMon_inh = StateMonitor(Pop_inh, variables=['V', 'A'], record=0, dt=1*ms)
                RateMon_exc = PopulationRateMonitor(Pop_exc)
                RateMon_inh = PopulationRateMonitor(Pop_inh)

                # ============================================================
                # NETWORK OBJECT AND INTRINSIC DYNAMICS
                # ============================================================

                net_objects = [
                    Pop_exc, Pop_inh,
                    SpikeMon_exc, SpikeMon_inh,
                    StateMon_exc, StateMon_inh,
                    RateMon_exc, RateMon_inh
                ] + all_synapses

                net = Network(net_objects)

                print(f"Running initial settling: {SIM_INITIAL_SETTLE_TIME/second:.1f} s")
                net.run(SIM_INITIAL_SETTLE_TIME, report='text', report_period=60*second)

                print(f"Running main simulation: {SIM_RUNTIME/second:.1f} s")
                net.run(SIM_RUNTIME, report='text', report_period=60*second)

                print("✓ Intrinsic dynamics simulation complete")

                # ============================================================
                # BASIC ANALYSIS
                # ============================================================

                print("Analyzing network activity...")

                # Firing rate
                analysis_start = SIM_INITIAL_SETTLE_TIME + ANALYSIS_DELAY_AFTER_SETTLE
                total_spikes_exc = len(SpikeMon_exc.t)
                mean_fr_exc = total_spikes_exc / (N_exc * (SIM_RUNTIME / second))
                phase_diagram_results_firing_rate[i_idx, j_idx, rep_idx] = mean_fr_exc

                # CV
                overall_mean_cv_val_exc = calculate_cv(SpikeMon_exc, N_exc, start_time=analysis_start)
                phase_diagram_results_overall_cv[i_idx, j_idx, rep_idx] = overall_mean_cv_val_exc

                print(f"  Mean FR: {mean_fr_exc:.2f} Hz")
                print(f"  CV: {overall_mean_cv_val_exc:.3f}")

                # ============================================================
                # AVALANCHE ANALYSIS
                # ============================================================

                print("Analyzing avalanches...")

                # Calculate IEI for adaptive bin width
                mean_iei_seconds = calculate_average_iei(SpikeMon_exc, analysis_start)

                if mean_iei_seconds and mean_iei_seconds > 0:
                    adaptive_bin_width_seconds = 4* mean_iei_seconds
                    print(f"  Adaptive bin width: {adaptive_bin_width_seconds*1000:.2f} ms (2×IEI)")

                    # Extract spike times
                    relevant_spikes = SpikeMon_exc.t[SpikeMon_exc.t >= analysis_start]
                    spike_times_for_analysis = np.array(relevant_spikes / second)

                    # Run avalanche analysis
                    avalanche_results = analyze_model_spikes(
                        spike_times_for_analysis,
                        [adaptive_bin_width_seconds],
                        param_combo_str
                    )

                    # Extract metrics
                    if adaptive_bin_width_seconds in avalanche_results:
                        res = avalanche_results[adaptive_bin_width_seconds]
                        phase_diagram_results_sigma[i_idx, j_idx, rep_idx] = res['branching_parameter']
                        phase_diagram_results_size_alpha[i_idx, j_idx, rep_idx] = res['size_alpha']
                        phase_diagram_results_duration_alpha[i_idx, j_idx, rep_idx] = res['duration_alpha']
                        phase_diagram_results_gamma[i_idx, j_idx, rep_idx] = res['gamma']

                        print(f"  Branching parameter (σ): {res['branching_parameter']:.3f}")
                        print(f"  Size exponent (α): {res['size_alpha']:.2f}")
                        print(f"  Avalanches detected: {res['num_avalanches']}")

                        # Store for aggregated plots
                        all_avalanche_distributions.append({
                            'condition': condition_name_str,
                            'repetition': rep_idx,
                            'sizes': [sum(av) for av in res['avalanches']],
                            'durations': [len(av) for av in res['avalanches']]
                        })
                else:
                    print("  Warning: Could not calculate IEI, skipping avalanche analysis")

                # ============================================================
                # RESERVOIR COMPUTING TASK
                # ============================================================

                print("Running reservoir computing task...")

                # Create input projection map
                pixel_to_neuron_map = create_input_projection_map(
                    N_INPUT_NEURONS, N_exc, N_inh,
                    neurons_per_pixel=1,  # 1-to-1 mapping
                    feed_to_inhibitory=FEED_INPUT_TO_INHIBITORY
                )
                avg_on_pixels = np.mean(np.sum(X_train_global, axis=1))
                total_trial_duration = RC_TRIAL_INTERNAL_SETTLE_TIME + STIMULUS_DURATION_PER_IMAGE + POST_STIMULUS_DURATION_TOTAL

                # The compensation is the average current per neuron, spread over the whole trial duration
                I_compensation_nA = (MNIST_CURRENT_AMPLITUDE / nA) * \
                                    (avg_on_pixels / N_TOTAL_NEURONS) * \
                                    (STIMULUS_DURATION_PER_IMAGE / total_trial_duration)
                I_compensation = I_compensation_nA * nA

                print(f"INFO: Calculated average number of ON pixels per image: {avg_on_pixels:.2f}")
                print(f"INFO: Total trial duration for compensation calculation: {total_trial_duration/ms:.0f} ms")
                print(f"INFO: Calculated statistical compensation current: {I_compensation/pA:.3f} pA")

                reduced_Imid = current_Imid - I_compensation
                if reduced_Imid < 0 * nA:
                    reduced_Imid = 0 * nA

                print(f"                                                                Reducing background current for RC task from {current_Imid/nA:.4f} nA to {reduced_Imid/nA:.4f} nA.")
                Pop_exc.Imid_val_eq = reduced_Imid
                Pop_inh.Imid_val_eq = reduced_Imid
                # Collect training states
                print(f"  Collecting {NUM_TRAIN_SAMPLES_EFFECTIVE} training states...")
                train_reservoir_states_list = []
                trial_details_for_plot = []

                for k_train in range(NUM_TRAIN_SAMPLES_EFFECTIVE):
                    if k_train > 0 and k_train % 50 == 0:
                        print(f"    Processed {k_train}/{NUM_TRAIN_SAMPLES_EFFECTIVE} train images")

                    input_img_flat_train = X_train_global[k_train]
                    digit_label = y_train_labels_global[k_train]

                    # Record time for plotting
                    trial_start_time = net.t

                    # Run simulation for this input
                    state_train = run_rc_simulation_for_input(
                        net, input_img_flat_train,
                        pixel_to_neuron_map,
                        Pop_exc, Pop_inh, N_exc, N_inh,
                        STIMULUS_DURATION_PER_IMAGE, POST_STIMULUS_DURATION_TOTAL,
                        MNIST_CURRENT_AMPLITUDE, defaultclock.dt,
                        SpikeMon_exc, SpikeMon_inh,
                        RC_TRIAL_INTERNAL_SETTLE_TIME,
                        READOUT_SNAPSHOT_TIME_OFFSET
                    )

                    train_reservoir_states_list.append(state_train)

                    # Store trial details for first few trials (for plotting)
                    if k_train < 10:
                        trial_details_for_plot.append({
                            'trial_start_ms': trial_start_time / ms,
                            'stim_start_ms': (trial_start_time + RC_TRIAL_INTERNAL_SETTLE_TIME) / ms,
                            'stim_end_ms': (trial_start_time + RC_TRIAL_INTERNAL_SETTLE_TIME + STIMULUS_DURATION_PER_IMAGE) / ms,
                            'trial_end_ms': (trial_start_time + RC_TRIAL_INTERNAL_SETTLE_TIME + STIMULUS_DURATION_PER_IMAGE + POST_STIMULUS_DURATION_TOTAL) / ms,
                            'digit': digit_label,
                            'stimulated_exc_indices': [idx for pixel in range(N_INPUT_NEURONS) if input_img_flat_train[pixel] > 0.5 for idx in pixel_to_neuron_map[pixel]['exc']],
                            'stimulated_inh_indices': [idx for pixel in range(N_INPUT_NEURONS) if input_img_flat_train[pixel] > 0.5 for idx in pixel_to_neuron_map[pixel]['inh']]
                        })

                X_train_states_collected = np.array(train_reservoir_states_list)
                print(f"  ✓ Collected training states: {X_train_states_collected.shape}")

                # Collect test states
                print(f"  Collecting {NUM_TEST_SAMPLES_EFFECTIVE} test states...")
                test_reservoir_states_list = []

                for k_test in range(NUM_TEST_SAMPLES_EFFECTIVE):
                    if k_test > 0 and k_test % 50 == 0:
                        print(f"    Processed {k_test}/{NUM_TEST_SAMPLES_EFFECTIVE} test images")

                    input_img_flat_test = X_test_global[k_test]

                    state_test = run_rc_simulation_for_input(
                        net, input_img_flat_test,
                        pixel_to_neuron_map,
                        Pop_exc, Pop_inh, N_exc, N_inh,
                        STIMULUS_DURATION_PER_IMAGE, POST_STIMULUS_DURATION_TOTAL,
                        MNIST_CURRENT_AMPLITUDE, defaultclock.dt,
                        SpikeMon_exc, SpikeMon_inh,
                        RC_TRIAL_INTERNAL_SETTLE_TIME,
                        READOUT_SNAPSHOT_TIME_OFFSET
                    )

                    test_reservoir_states_list.append(state_test)

                X_test_states_collected = np.array(test_reservoir_states_list)
                print(f"  ✓ Collected test states: {X_test_states_collected.shape}")

                # Train readout for different training set sizes
                rc_accuracies_for_run = {}
                best_accuracy_this_run = 0.0
                num_samples_for_best_acc = 0

                for n_samples_subset in READOUT_TRAINING_SUBSETS:
                    if n_samples_subset > X_train_states_collected.shape[0]:
                        continue

                    subset_indices = np.random.choice(X_train_states_collected.shape[0], n_samples_subset, replace=False)
                    X_subset = X_train_states_collected[subset_indices]
                    y_subset = y_train_onehot_global[subset_indices]

                    W_out = train_readout_weights(X_subset, y_subset, RIDGE_ALPHA)
                    accuracy, _ = evaluate_readout_performance(X_test_states_collected, W_out, y_test_labels_global)

                    rc_accuracies_for_run[n_samples_subset] = accuracy
                    print(f"    RC accuracy with {n_samples_subset} samples: {accuracy:.4f}")

                    if accuracy > best_accuracy_this_run:
                        best_accuracy_this_run = accuracy
                        num_samples_for_best_acc = n_samples_subset

                # Store results
                if rc_accuracies_for_run:
                    phase_diagram_learning_curves_data[i_idx][j_idx][rep_idx] = rc_accuracies_for_run
                    final_accuracy = rc_accuracies_for_run.get(READOUT_TRAINING_SUBSETS[-1], np.nan)
                    phase_diagram_results_rc_accuracy[i_idx, j_idx, rep_idx] = final_accuracy
                    print(f"  ✓ Best RC accuracy: {best_accuracy_this_run:.4f} ({num_samples_for_best_acc} samples)")

                rc_plot_info = {
                    'best_accuracy': best_accuracy_this_run,
                    'num_samples_for_best_accuracy': num_samples_for_best_acc,
                    'learning_curve': rc_accuracies_for_run
                }

                # ============================================================
                # PLOTTING (only first repetition)
                # ============================================================

                if rep_idx == 0:
                    print("Generating plots...")
                    output_dir_for_run = f"results_phase_diagram_runs/{param_combo_str}"
                    os.makedirs(output_dir_for_run, exist_ok=True)

                    # Basic activity plot
                    plot_basic_activity(
                        SpikeMon_exc, SpikeMon_inh,
                        StateMon_exc, StateMon_inh,
                        RateMon_exc, RateMon_inh,
                        N_exc, N_inh,
                        SIM_INITIAL_SETTLE_TIME, SIM_RUNTIME,
                        imid_val_nA, overall_mean_cv_val_exc,
                        CV_WINDOW_SIZE, CV_STEP_SIZE,
                        param_combo_str, rcaccuracyinfo=rc_plot_info
                    )

                    # Initial 5s raster
                    plot_initial_raster(
                        SpikeMon_exc, SpikeMon_inh, N_exc, N_inh,
                        starttime=SIM_INITIAL_SETTLE_TIME,
                        duration=5*second,
                        outputdir=output_dir_for_run,
                        groupname=param_combo_str
                    )

                    # Detailed stimulus raster (RC trials)
                    if trial_details_for_plot:
                        plot_detailed_stimulus_raster(
                            SpikeMon_exc, SpikeMon_inh,
                            trial_details_for_plot,
                            N_exc, N_inh,
                            output_dir_for_run,
                            param_combo_str
                        )

                    print("  ✓ Plots saved")

                # ============================================================
                # STORE RESULTS
                # ============================================================

                run_data_dict = {
                    'Repetition': rep_idx,
                    'IMID (nA)': imid_val_nA,
                    'E/I Ratio': ei_ratio,
                    'Mean Firing Rate (Hz)': phase_diagram_results_firing_rate[i_idx, j_idx, rep_idx],
                    'Overall Mean CV': phase_diagram_results_overall_cv[i_idx, j_idx, rep_idx],
                    'Branching Parameter (sigma)': phase_diagram_results_sigma[i_idx, j_idx, rep_idx],
                    'Avalanche Size Alpha': phase_diagram_results_size_alpha[i_idx, j_idx, rep_idx],
                    'Avalanche Duration Alpha': phase_diagram_results_duration_alpha[i_idx, j_idx, rep_idx],
                    'Avalanche Gamma': phase_diagram_results_gamma[i_idx, j_idx, rep_idx],
                    'RC Accuracy (Best)': best_accuracy_this_run,
                    'Samples to Threshold': np.nan,  # Will be filled in post-processing
                    'RC Accuracy (Fixed Samples)': np.nan  # Will be filled in post-processing
                }
                all_runs_results_list.append(run_data_dict)

                print(f"✓ Run {current_run_counter}/{total_runs} complete")

            except Exception as e:
                print(f"ERROR during simulation for {param_combo_str}: {e}")
                traceback.print_exc()

            finally:
                # Cleanup
                BrianObject.__instances__().clear()

print("\n" + "=" * 80)
print(" " * 20 + "PARAMETER SWEEP COMPLETE")
print("=" * 80)

# ============================================================================
# POST-PROCESSING AND ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print(" " * 20 + "POST-PROCESSING RESULTS")
print("=" * 80)

# Calculate learning curve metrics
print("\nCalculating learning curve metrics...")

for r_idx in range(len(Imid_values_nA)):
    for c_idx in range(len(EI_ratio_values)):
        for rep_idx in range(NUM_REPETITIONS):
            lc_data = phase_diagram_learning_curves_data[r_idx][c_idx][rep_idx]

            if lc_data and isinstance(lc_data, dict) and lc_data:
                # Samples to threshold
                samples_to_thresh = calculate_samples_to_reach_threshold(
                    lc_data, QUICKNESS_TARGET_ACCURACY, READOUT_TRAINING_SUBSETS
                )
                phase_diagram_samples_to_threshold[r_idx, c_idx, rep_idx] = samples_to_thresh

                # Accuracy at fixed sample size
                acc_at_fixed = get_accuracy_at_fixed_samples(lc_data, QUICKNESS_FIXED_SAMPLE_SIZE)
                phase_diagram_accuracy_at_fixed_samples[r_idx, c_idx, rep_idx] = acc_at_fixed

                # Update stored results
                for run_dict in all_runs_results_list:
                    if (run_dict['IMID (nA)'] == Imid_values_nA[r_idx] and
                        run_dict['E/I Ratio'] == EI_ratio_values[c_idx] and
                        run_dict['Repetition'] == rep_idx):
                        run_dict['Samples to Threshold'] = samples_to_thresh
                        run_dict['RC Accuracy (Fixed Samples)'] = acc_at_fixed
                        break

print("✓ Learning curve metrics calculated")

# ============================================================================
# SAVE RESULTS TO EXCEL
# ============================================================================

print("\nSaving results to Excel...")

output_directory = "results_phase_diagram_summary"
os.makedirs(output_directory, exist_ok=True)

# Create Excel file
if all_runs_results_list:
    output_path = os.path.join(output_directory, "simulation_summary.xlsx")
    df = pd.DataFrame(all_runs_results_list)

    # Add condition names
    df['Condition'] = df['E/I Ratio'].map(condition_map)

    # Define column order
    column_order = [
        'Repetition', 'IMID (nA)', 'E/I Ratio', 'Condition',
        'Mean Firing Rate (Hz)', 'Overall Mean CV',
        'Branching Parameter (sigma)', 'RC Accuracy (Best)',
        'RC Accuracy (Fixed Samples)', 'Samples to Threshold',
        'Avalanche Size Alpha', 'Avalanche Duration Alpha', 'Avalanche Gamma'
    ]

    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]

    # Sort for readability
    df.sort_values(by=['IMID (nA)', 'E/I Ratio', 'Repetition'], inplace=True)

    try:
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"✓ Results saved to {output_path}")
    except Exception as e:
        print(f"Warning: Could not save Excel file: {e}")
        csv_path = os.path.join(output_directory, "simulation_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to CSV: {csv_path}")
else:
    print("Warning: No results to save")

# ============================================================================
# GENERATE SUMMARY PLOTS
# ============================================================================

print("\nGenerating summary plots...")

summary_metrics = {
    'firing_rate': (
        np.nanmean(phase_diagram_results_firing_rate, axis=2),
        'Mean Firing Rate (Exc, Hz)',
        np.nanstd(phase_diagram_results_firing_rate, axis=2)
    ),
    'cv': (
        np.nanmean(phase_diagram_results_overall_cv, axis=2),
        'Overall Mean CV (Exc)',
        np.nanstd(phase_diagram_results_overall_cv, axis=2)
    ),
    'sigma': (
        np.nanmean(phase_diagram_results_sigma, axis=2),
        'Branching Parameter (σ, bin=2×IEI)',
        np.nanstd(phase_diagram_results_sigma, axis=2)
    ),
    'rc_accuracy': (
        np.nanmean(phase_diagram_results_rc_accuracy, axis=2),
        'MNIST Classification Accuracy',
        np.nanstd(phase_diagram_results_rc_accuracy, axis=2)
    ),
    'samples_to_threshold': (
        np.nanmean(phase_diagram_samples_to_threshold, axis=2),
        f'Samples to reach {QUICKNESS_TARGET_ACCURACY*100:.0f}% accuracy',
        np.nanstd(phase_diagram_samples_to_threshold, axis=2)
    ),
    'accuracy_at_fixed_samples': (
        np.nanmean(phase_diagram_accuracy_at_fixed_samples, axis=2),
        f'Accuracy with {QUICKNESS_FIXED_SAMPLE_SIZE} training samples',
        np.nanstd(phase_diagram_accuracy_at_fixed_samples, axis=2)
    )
}

generate_summary_plots(Imid_values_nA, EI_ratio_values, summary_metrics, output_directory)
print("✓ Phase diagrams generated")

# ============================================================================
# LEARNING CURVES
# ============================================================================

print("\nPlotting learning curves...")

plot_all_learning_accuracy_curves(
    phase_diagram_learning_curves_data,
    Imid_values_nA,
    EI_ratio_values,
    condition_map,
    output_directory
)
print("✓ Learning curves plotted")

# ============================================================================
# NEURAL MANIFOLD
# ============================================================================

print("\nPlotting neural manifold...")

if 'X_test_states_collected' in locals() and X_test_states_collected.size > 0:
    last_condition = condition_map.get(EI_ratio_values[-1], f"EI_{EI_ratio_values[-1]:.3f}")
    plot_neural_manifold(
        X_test_states_collected,
        y_test_labels_global,
        title=f"Neural Manifold of Test Set ({last_condition.capitalize()})",
        outputdir=output_directory
    )
    print("✓ Neural manifold plotted")
else:
    print("  Skipped: No test states available")

# ============================================================================
# AGGREGATED AVALANCHE PLOTS
# ============================================================================

print("\nPlotting aggregated avalanche distributions...")

if all_avalanche_distributions:
    from plotting import plot_separate_aggregated_avalanche_ccdfs
    plot_separate_aggregated_avalanche_ccdfs(all_avalanche_distributions, output_directory)
    print("✓ Avalanche distributions plotted")
else:
    print("  Skipped: No avalanche data available")

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

print("\n" + "=" * 80)
print(" " * 20 + "STATISTICAL ANALYSIS")
print("=" * 80)

statistical_results_to_test = {
    "Final Accuracy": phase_diagram_results_rc_accuracy,
    "Accuracy at Fixed Samples": phase_diagram_accuracy_at_fixed_samples,
    "Samples to Threshold": phase_diagram_samples_to_threshold
}

run_and_print_statistical_tests(
    statistical_results_to_test,
    EI_ratio_values,
    Imid_values_nA,
    NUM_REPETITIONS,
    condition_map
)

# Statistical analysis of learning curves
run_learning_curve_statistics(
    phase_diagram_learning_curves_data,
    Imid_values_nA,
    EI_ratio_values,
    condition_map
)

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "=" * 80)
print(" " * 25 + "SIMULATION COMPLETE!")
print("=" * 80)
print(f"\nResults saved to: {output_directory}/")
print("  - simulation_summary.xlsx")
print("  - phase_diagram_*.png")
print("  - comparative_learning_accuracy_curves.png")
print("  - neural_manifold_pca.png")
print("  - aggregated_avalanche_ccdf_*.png")
print("\nIndividual run results in: results_phase_diagram_runs/")
print("=" * 80)
