"""
Plotting functions for neural network simulation
All visualization and figure generation
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from brian2 import ms

from config import THESIS_STYLE

# Apply thesis style
plt.rcParams.update(THESIS_STYLE)


def plot_basic_activity(SpikeMonexc, SpikeMoninh, StateMonexc, StateMoninh, 
                        RateMonexc, RateMoninh, Nexc, Ninh, 
                        initialsettletimeval, runtimeval, currentImidnAval, 
                        overallmeancvexcval, cvwindowsize, cvstepsize, 
                        groupname="Model", rcaccuracyinfo=None):
    """
    Create comprehensive activity plot showing raster, voltage, adaptation, CV, and rate.

    Parameters
    ----------
    SpikeMonexc, SpikeMoninh : SpikeMonitor
        Excitatory and inhibitory spike monitors
    StateMonexc, StateMoninh : StateMonitor
        State monitors for voltage and adaptation
    RateMonexc, RateMoninh : PopulationRateMonitor
        Population rate monitors
    Nexc, Ninh : int
        Number of excitatory and inhibitory neurons
    initialsettletimeval, runtimeval : Quantity
        Settling and runtime durations
    currentImidnAval : float
        Background current in nA
    overallmeancvexcval : float
        Overall mean CV value
    cvwindowsize, cvstepsize : Quantity
        Parameters for live CV calculation
    groupname : str
        Name for this condition
    rcaccuracyinfo : dict, optional
        RC accuracy information

    Returns
    -------
    None (saves plot to disk)
    """
    from analysis import calculate_live_cv
    from brian2 import ms, nA, mV

    outputdir = f"results_phase_diagram_runs/{groupname}/"
    os.makedirs(outputdir, exist_ok=True)

    fig = plt.figure(figsize=(12, 18))
    gs = plt.GridSpec(6, 1, figure=fig, height_ratios=[0.5, 2, 1, 1, 1, 1.5])

    exccolor = "crimson"
    inhcolor = "royalblue"

    # Info panel
    axinfo = fig.add_subplot(gs[0, 0])
    axinfo.axis("off")
    cvtext = f"Overall Mean Exc CV (Delayed Analysis): {overallmeancvexcval:.3f}" if not np.isnan(overallmeancvexcval) else "Overall Mean Exc CV (Delayed Analysis): NA"

    rctext = ""
    if rcaccuracyinfo:
        bestacc = rcaccuracyinfo.get("best_accuracy", np.nan)
        numsamplesforbestacc = rcaccuracyinfo.get("num_samples_for_best_accuracy", "NA")
        rctext = f"\nRC MNIST Accuracy: {bestacc:.4f} with {numsamplesforbestacc} training samples"

    titletext = f"Network {groupname}\n{cvtext}{rctext}"
    axinfo.text(0.5, 0.5, titletext, ha="center", va="center", fontsize=10, wrap=True)

    # Raster plot
    ax1 = fig.add_subplot(gs[1, 0])
    if SpikeMonexc and hasattr(SpikeMonexc, "t") and len(SpikeMonexc.t) > 0:
        ax1.plot(SpikeMonexc.t/ms, SpikeMonexc.i, ".", color=exccolor, markersize=0.1, label="Excitatory")
    if SpikeMoninh and hasattr(SpikeMoninh, "t") and len(SpikeMoninh.t) > 0:
        ax1.plot(SpikeMoninh.t/ms, SpikeMoninh.i + Nexc, ".", color=inhcolor, markersize=0.1, label="Inhibitory")
    ax1.set_ylabel("Neuron index")
    ax1.legend(markerscale=20, loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.tick_params(labelbottom=False)

    # Voltage traces
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
    if StateMonexc and hasattr(StateMonexc, "V") and len(StateMonexc.t) > 0 and StateMonexc.V.shape[1] > 0:
        ax2.plot(StateMonexc.t/ms, StateMonexc.V[0]/mV, color=exccolor, linewidth=0.8, label="Exc N0 Vm")
    if StateMoninh and hasattr(StateMoninh, "V") and len(StateMoninh.t) > 0 and StateMoninh.V.shape[1] > 0:
        ax2.plot(StateMoninh.t/ms, StateMoninh.V[0]/mV, color=inhcolor, linewidth=0.8, label="Inh N0 Vm")
    ax2.set_ylabel("Voltage (mV)")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.tick_params(labelbottom=False)
    ax2.legend(loc="upper right")

    # Adaptation traces
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax1)
    if StateMonexc and hasattr(StateMonexc, "A") and len(StateMonexc.t) > 0 and StateMonexc.A.shape[1] > 0:
        ax3.plot(StateMonexc.t/ms, StateMonexc.A[0]/nA, color="black", linewidth=0.8, label="Exc N0 Adapt")
    if StateMoninh and hasattr(StateMoninh, "A") and len(StateMoninh.t) > 0 and StateMoninh.A.shape[1] > 0:
        ax3.plot(StateMoninh.t/ms, StateMoninh.A[0]/nA, color="dimgray", linestyle="-", linewidth=0.8, label="Inh N0 Adapt")
    ax3.set_ylabel("Adaptation (nA)")
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.tick_params(labelbottom=False)
    ax3.legend(loc="upper right")

    # Live CV plot
    ax4 = fig.add_subplot(gs[4, 0], sharex=ax1)
    if SpikeMonexc:
        livecvtimesms, livecvvalues = calculate_live_cv(
            SpikeMonexc, Nexc,
            analysis_start_time=initialsettletimeval,
            analysis_duration=runtimeval,
            window_size=cvwindowsize,
            step_size=cvstepsize
        )
        if len(livecvtimesms) > 0 and not np.all(np.isnan(livecvvalues)):
            ax4.plot(livecvtimesms, livecvvalues, "m-", linewidth=1.5, label="Live CV Exc")
            ax4.set_ylabel("Live CV Exc")
            ax4.legend(loc="upper right")
            ax4.grid(True, linestyle=":", alpha=0.5)
        else:
            ax4.text(0.5, 0.5, "No Live CV data", ha="center", va="center", transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, "No Exc Spikes for CV", ha="center", va="center", transform=ax4.transAxes)
    ax4.set_ylabel("Live CV Exc")
    ax4.tick_params(labelbottom=False)

    # Rate and input current plot
    ax5 = fig.add_subplot(gs[5, 0], sharex=ax1)
    lines_ax5, labels_ax5 = [], []

    if RateMonexc and hasattr(RateMonexc, "t") and len(RateMonexc.t) > 0:
        try:
            from brian2 import Hz
            smooth_rate_exc = RateMonexc.smooth_rate(window="gaussian", width=10*ms) / Hz
            line, = ax5.plot(RateMonexc.t/ms, smooth_rate_exc, color=exccolor, linewidth=1.2, label="Exc Rate (Hz)")
            lines_ax5.append(line)
            labels_ax5.append("Exc Rate (Hz)")
        except Exception:
            pass

    if RateMoninh and hasattr(RateMoninh, "t") and len(RateMoninh.t) > 0:
        try:
            from brian2 import Hz
            smooth_rate_inh = RateMoninh.smooth_rate(window="gaussian", width=10*ms) / Hz
            line, = ax5.plot(RateMoninh.t/ms, smooth_rate_inh, color=inhcolor, linewidth=1.2, label="Inh Rate (Hz)")
            lines_ax5.append(line)
            labels_ax5.append("Inh Rate (Hz)")
        except Exception:
            pass

    ax5.set_ylabel("Rate (Hz)")

    # Add input current on secondary axis
    ax5b = ax5.twinx()
    total_duration_ms_plot_base = (initialsettletimeval + runtimeval) / ms
    max_mon_time_ms = total_duration_ms_plot_base

    if SpikeMonexc and hasattr(SpikeMonexc, "t") and len(SpikeMonexc.t) > 0:
        max_mon_time_ms = max(max_mon_time_ms, np.max(SpikeMonexc.t/ms))
    if RateMonexc and hasattr(RateMonexc, "t") and len(RateMonexc.t) > 0:
        max_mon_time_ms = max(max_mon_time_ms, np.max(RateMonexc.t/ms))

    ax1.set_xlim(0, max_mon_time_ms)

    time_points_plot_ms = np.linspace(0, max_mon_time_ms, 500)
    imid_signal_plot = np.zeros_like(time_points_plot_ms)
    settle_time_plot_ms = initialsettletimeval / ms
    imid_signal_plot[time_points_plot_ms >= settle_time_plot_ms] = currentImidnAval

    line, = ax5b.plot(time_points_plot_ms, imid_signal_plot, "g--", linewidth=1.5, label="Imid (nA)")
    lines_ax5.append(line)
    labels_ax5.append("Imid (nA)")

    ax5b.set_ylabel("Imid (nA)", color="g")
    ax5b.tick_params(axis="y", labelcolor="g")

    min_imid_plot = min(0, currentImidnAval*1.1 if currentImidnAval < 0 else currentImidnAval*0.9) - 0.05*abs(currentImidnAval+1e-9)
    max_imid_plot = max(0.1, currentImidnAval*1.1 if currentImidnAval > 0 else currentImidnAval*0.9) + 0.05*abs(currentImidnAval+1e-9)
    ax5b.set_ylim(min_imid_plot, max_imid_plot)

    ax5.set_xlabel("Time (ms)")
    ax5.grid(True, linestyle=":", alpha=0.5)
    ax5.legend(lines_ax5, labels_ax5, loc="upper right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.suptitle(f"Network Activity {groupname}", y=0.99)

    plt.savefig(f"{outputdir}/basic_activity_plot.png", dpi=300, bbox_inches="tight")
    if plt.fignum_exists(fig.number):
        plt.close(fig)


def plot_initial_raster(spikemonexc, spikemoninh, nexc, ninh, starttime, duration, 
                        outputdir, groupname):
    """
    Plots a raster of the first few seconds of the main simulation run.

    Parameters
    ----------
    spikemonexc, spikemoninh : SpikeMonitor
        Spike monitors
    nexc, ninh : int
        Number of neurons
    starttime : Quantity
        Start time for raster
    duration : Quantity
        Duration to plot
    outputdir : str
        Output directory
    groupname : str
        Condition name
    """
    from brian2 import second

    fig, ax = plt.subplots(figsize=(12, 7))

    start_time_ms = starttime / second * 1000
    end_time_ms = start_time_ms + duration / second * 1000

    # Filter excitatory spikes
    exc_mask = (spikemonexc.t/second*1000 >= start_time_ms) & (spikemonexc.t/second*1000 <= end_time_ms)
    ax.plot(spikemonexc.t[exc_mask]/second*1000, spikemonexc.i[exc_mask], ".", 
            color="crimson", markersize=1.5, label="Excitatory")

    # Filter inhibitory spikes
    inh_mask = (spikemoninh.t/second*1000 >= start_time_ms) & (spikemoninh.t/second*1000 <= end_time_ms)
    ax.plot(spikemoninh.t[inh_mask]/second*1000, spikemoninh.i[inh_mask] + nexc, ".", 
            color="royalblue", markersize=1.5, label="Inhibitory")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron Index")
    ax.set_title(f"Initial {duration/second}s Raster Plot - {groupname}")
    ax.set_xlim(start_time_ms, end_time_ms)
    ax.set_ylim(-10, nexc + ninh + 10)
    ax.axhline(y=nexc, color="black", linestyle="--", linewidth=1.0)
    ax.legend(markerscale=10, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plot_filename = os.path.join(outputdir, "initial_5s_raster.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)


def plot_detailed_stimulus_raster(spikemonexc, spikemoninh, trial_details, nexc, ninh, 
                                   outputdir, groupname):
    """
    Plot detailed raster showing stimulus presentation trials with digit labels.

    Parameters
    ----------
    spikemonexc, spikemoninh : SpikeMonitor
        Spike monitors
    trial_details : list of dict
        Trial information including timing and stimulated neurons
    nexc, ninh : int
        Number of neurons
    outputdir : str
        Output directory
    groupname : str
        Condition name
    """
    fig, ax = plt.subplots(figsize=(15, 8))

    exc_color_muted = "#b2df8a"  # light green
    inh_color_muted = "#a1c9f4"  # light blue
    highlight_color = "crimson"

    # Plot all spikes as background
    ax.plot(spikemonexc.t/1e-3, spikemonexc.i, ".", color=exc_color_muted, 
            markersize=0.7, label="Exc. Spike (Unstimulated)", zorder=1)
    ax.plot(spikemoninh.t/1e-3, spikemoninh.i + nexc, ".", color=inh_color_muted, 
            markersize=0.7, label="Inh. Spike (Unstimulated)", zorder=1)

    # Iterate through trials
    for i, trial in enumerate(trial_details):
        stim_start = trial["stim_start_ms"]
        stim_end = trial["stim_end_ms"]
        trial_end = trial["trial_end_ms"]
        digit = trial["digit"]
        stimulated_exc_indices = trial["stimulated_exc_indices"]
        stimulated_inh_indices = trial["stimulated_inh_indices"]

        # Add shaded regions
        ax.axvspan(stim_start, stim_end, facecolor="mistyrose", alpha=0.7, 
                  edgecolor="none", zorder=0)
        ax.axvspan(stim_end, trial_end, facecolor="aliceblue", alpha=0.7, 
                  edgecolor="none", zorder=0)

        # Add digit label
        ax.text((stim_start + stim_end) / 2, ax.get_ylim()[1] * 1.02, 
               f"Digit {digit}", ha="center", va="bottom", fontsize=12, fontweight="bold")

        # Highlight stimulated neuron spikes
        exc_mask = ((spikemonexc.t/ms >= stim_start) &    # <--- FIXED
                    (spikemonexc.t/ms <= stim_end) &
                    np.isin(spikemonexc.i, stimulated_exc_indices))
        ax.plot(spikemonexc.t[exc_mask]/1e-3, spikemonexc.i[exc_mask], ".", 
               color=highlight_color, markersize=2.5, zorder=2)

        if stimulated_inh_indices:
            inh_mask = ((spikemoninh.t/ms >= stim_start) & 
                        (spikemoninh.t/ms <= stim_end) &
                        np.isin(spikemoninh.i, stimulated_inh_indices))
            ax.plot(spikemoninh.t[inh_mask]/1e-3, spikemoninh.i[inh_mask] + nexc, ".", 
                   color=highlight_color, markersize=2.5, zorder=2)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron Index")
    ax.set_title(f"Detailed Stimulus Presentation Raster: {groupname}")
    ax.grid(True, linestyle=":", alpha=0.6)

    if trial_details:
        ax.set_xlim(trial_details[0]["trial_start_ms"] - 50, trial_details[-1]["trial_end_ms"] + 50)
    ax.set_ylim(-10, nexc + ninh + 10)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker=".", color=exc_color_muted, label="Exc. Spike", 
               linestyle="None", markersize=8),
        Line2D([0], [0], marker=".", color=inh_color_muted, label="Inh. Spike", 
               linestyle="None", markersize=8),
        Line2D([0], [0], marker=".", color=highlight_color, label="Stimulated Neuron Spike", 
               linestyle="None", markersize=8),
        mpatches.Patch(color="mistyrose", alpha=0.7, label="Stimulus Period"),
        mpatches.Patch(color="aliceblue", alpha=0.7, label="Delay Period")
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, 
             facecolor="white", framealpha=1.0)
    ax.axhline(y=nexc, color="black", linestyle="--", linewidth=1.2)

    plt.tight_layout()
    plot_filename = os.path.join(outputdir, "detailed_stimulus_raster.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved detailed stimulus raster plot to {plot_filename}")

    if plt.fignum_exists(fig.number):
        plt.close(fig)


def plot_neural_manifold(reservoir_states, labels, title="Neural Manifold of Digit Representations", 
                        outputdir="."):
    """
    Generate 3D PCA visualization of reservoir states colored by digit class.

    Parameters
    ----------
    reservoir_states : array
        Reservoir state matrix (n_samples × n_neurons)
    labels : array
        Digit labels for each sample
    title : str
        Plot title
    outputdir : str
        Output directory
    """
    if reservoir_states.size == 0:
        print("Warning: Cannot plot manifold, reservoir states are empty.")
        return

    print("--- Generating Neural Manifold Plot ---")
    print(f"Original data shape: {reservoir_states.shape}")

    # Standardize and reduce to 3D
    scaler = StandardScaler()
    scaled_states = scaler.fit_transform(reservoir_states)

    pca = PCA(n_components=3)
    states_3d = pca.fit_transform(scaled_states)

    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Variance explained by 3 principal components: {explained_variance:.2%}")

    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    colors = cm.get_cmap("tab10", 10)

    for i in range(10):
        digit_indices = np.where(labels == i)[0]
        if len(digit_indices) > 0:
            ax.scatter(
                states_3d[digit_indices, 0],
                states_3d[digit_indices, 1],
                states_3d[digit_indices, 2],
                color=colors(i),
                label=str(i),
                s=60,
                alpha=0.8,
                edgecolor="k",
                linewidth=0.5
            )

    ax.set_title(title, pad=20)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")

    legend = ax.legend(title="Digits", markerscale=1.5, bbox_to_anchor=(1.1, 0.8))
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 0, 0.0))

    ax.grid(True)
    ax.view_init(elev=20., azim=-65)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")

    output_path = os.path.join(outputdir, "neural_manifold_pca.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved neural manifold plot to {output_path}")
    plt.close(fig)


def create_combined_plots(results_dict_keys_seconds, bin_widths_labels_us, groupname):
    """
    Create combined avalanche distribution plots for multiple bin widths.

    Parameters
    ----------
    results_dict_keys_seconds : dict
        Avalanche analysis results keyed by bin width
    bin_widths_labels_us : list
        List of bin widths in microseconds
    groupname : str
        Condition name
    """
    outputdir = f"results_phase_diagram_runs/{groupname}/"
    os.makedirs(outputdir, exist_ok=True)

    valid_bw_labels_us = [
        bw_label for bw_label in bin_widths_labels_us
        if (bw_label * 1e-6) in results_dict_keys_seconds 
        and results_dict_keys_seconds[bw_label * 1e-6] is not None
    ]

    if not valid_bw_labels_us:
        return

    cmap_func = plt.cm.viridis
    colors = cmap_func(np.linspace(0, 1, len(valid_bw_labels_us))) if len(valid_bw_labels_us) > 1 else [cmap_func(0.5)]

    # Size and duration distributions
    fig_dist, (ax_size, ax_dur) = plt.subplots(1, 2, figsize=(15, 6))

    for i, bw_label_us in enumerate(valid_bw_labels_us):
        bw_seconds_key = bw_label_us * 1e-6
        res = results_dict_keys_seconds[bw_seconds_key]
        color = colors[i]
        label_suffix = f"{bw_label_us:.0f}µs"

        # Plot size distribution
        if res["size_fit"] and hasattr(res["size_fit"], "data") and len(res["size_fit"].data) > 1:
            try:
                fit_label = label_suffix + f" α={res['size_alpha']:.2f}" if not np.isnan(res['size_alpha']) else label_suffix
                res["size_fit"].plot_ccdf(ax=ax_size, color=color, linewidth=1.5, 
                                         marker="o", markersize=4, label=fit_label)
            except Exception:
                pass

        # Plot duration distribution
        if res["duration_fit"] and hasattr(res["duration_fit"], "data") and len(res["duration_fit"].data) > 1:
            try:
                fit_label = label_suffix + f" α={res['duration_alpha']:.2f}" if not np.isnan(res['duration_alpha']) else label_suffix
                res["duration_fit"].plot_ccdf(ax=ax_dur, color=color, linewidth=1.5, 
                                            marker="o", markersize=4, label=fit_label)
            except Exception:
                pass

    ax_size.set_xlabel("Avalanche Size (spikes)")
    ax_size.set_ylabel("CCDF P(S≥s)")
    ax_size.set_title("Size Distribution")
    ax_size.set_xscale("log")
    ax_size.set_yscale("log")
    if len(valid_bw_labels_us) > 0:
        ax_size.legend(fontsize=9)
    ax_size.grid(True, which="both", linestyle="--", alpha=0.3)

    ax_dur.set_xlabel("Avalanche Duration (bins)")
    ax_dur.set_ylabel("CCDF P(T≥t)")
    ax_dur.set_title("Duration Distribution")
    ax_dur.set_xscale("log")
    ax_dur.set_yscale("log")
    if len(valid_bw_labels_us) > 0:
        ax_dur.legend(fontsize=9)
    ax_dur.grid(True, which="both", linestyle="--", alpha=0.3)

    fig_dist.suptitle(f"Avalanche Distributions {groupname}", fontsize=16, y=1.02)
    fig_dist.tight_layout(rect=[0,0,1,0.98])
    fig_dist.savefig(f"{outputdir}/avalanche_distributions_combined.png", dpi=300)
    if plt.fignum_exists(fig_dist.number):
        plt.close(fig_dist)

    # Scaling relation
    fig_scaling, ax_scaling = plt.subplots(figsize=(8, 6))

    for i, bw_label_us in enumerate(valid_bw_labels_us):
        bw_seconds_key = bw_label_us * 1e-6
        res = results_dict_keys_seconds[bw_seconds_key]
        color = colors[i]
        label_suffix = f"{bw_label_us:.0f}µs"

        if isinstance(res["duration_groups"], pd.DataFrame) and not res["duration_groups"].empty:
            if "duration" in res["duration_groups"].columns and "size" in res["duration_groups"].columns:
                if len(res["duration_groups"]) > 1:
                    try:
                        plot_data = res["duration_groups"][
                            (res["duration_groups"]["duration"] > 0) & 
                            (res["duration_groups"]["size"] > 0)
                        ]
                        if not plot_data.empty:
                            ax_scaling.scatter(plot_data["duration"], plot_data["size"], 
                                             color=color, alpha=0.6, s=40, 
                                             label=label_suffix + f" γ={res['gamma']:.2f}" if not np.isnan(res['gamma']) else label_suffix)

                            # Plot fit line
                            if not np.isnan(res['gamma']) and not np.isnan(res['intercept']):
                                if len(res.get("log_durations", [])) >= 2:
                                    min_dur_log, max_dur_log = np.min(res["log_durations"]), np.max(res["log_durations"])
                                    if min_dur_log < max_dur_log:
                                        x_fit_log = np.linspace(min_dur_log, max_dur_log, 50)
                                        y_fit_log = res["intercept"] + res["gamma"] * x_fit_log
                                        ax_scaling.plot(10**x_fit_log, 10**y_fit_log, 
                                                      color=color, linewidth=2, linestyle="--")
                    except Exception:
                        pass

    ax_scaling.set_xscale("log")
    ax_scaling.set_yscale("log")
    ax_scaling.set_xlabel("Duration (bins)")
    ax_scaling.set_ylabel("Size (spikes)")
    ax_scaling.set_title(f"Size-Duration Scaling {groupname}")
    if len(valid_bw_labels_us) > 0:
        ax_scaling.legend(fontsize=9)
    ax_scaling.grid(True, which="both", linestyle="--", alpha=0.3)

    fig_scaling.tight_layout()
    fig_scaling.savefig(f"{outputdir}/avalanche_scaling_relation.png", dpi=300)
    if plt.fignum_exists(fig_scaling.number):
        plt.close(fig_scaling)


def create_individual_plots(results_dict_keys_seconds, bin_widths_labels_us, groupname):
    """
    Create individual avalanche distribution plots for each bin width.

    Parameters
    ----------
    results_dict_keys_seconds : dict
        Avalanche analysis results
    bin_widths_labels_us : list
        Bin widths in microseconds
    groupname : str
        Condition name
    """
    outputdir = f"results_phase_diagram_runs/{groupname}/individual_avalanche_plots/"
    os.makedirs(outputdir, exist_ok=True)

    valid_bw_labels_us = [
        bw_label for bw_label in bin_widths_labels_us
        if (bw_label * 1e-6) in results_dict_keys_seconds 
        and results_dict_keys_seconds[bw_label * 1e-6] is not None
    ]

    if not valid_bw_labels_us:
        return

    cmap_func = plt.cm.viridis
    colors = cmap_func(np.linspace(0, 1, len(valid_bw_labels_us))) if len(valid_bw_labels_us) > 1 else [cmap_func(0.5)]

    for i, bw_label_us in enumerate(valid_bw_labels_us):
        bw_seconds_key = bw_label_us * 1e-6
        res = results_dict_keys_seconds[bw_seconds_key]
        color = colors[i]

        # Size distribution
        if res["size_fit"] and hasattr(res["size_fit"], "power_law") and res["size_fit"].power_law:
            if hasattr(res["size_fit"], "data") and len(res["size_fit"].data) > 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                try:
                    res["size_fit"].plot_ccdf(color=color, linewidth=1.5, marker="o", markersize=4, label="Empirical", ax=ax)
                    res["size_fit"].power_law.plot_ccdf(color="black", linestyle="--", linewidth=2, 
                                                        label=f"Fit α={res['size_alpha']:.2f}, xmin={res['size_fit'].xmin:.1f}", ax=ax)
                    ax.set_title(f"Size Distribution - {bw_label_us:.0f}µs {groupname}")
                    ax.set_xlabel("Avalanche Size (s)")
                    ax.set_ylabel("CCDF P(S≥s)")
                    ax.legend()
                    ax.grid(True, which="both", ls="--", alpha=0.3)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    plt.tight_layout()
                    plt.savefig(f"{outputdir}/size_dist_{bw_label_us:.0f}us.png", dpi=300)
                except Exception:
                    pass
                finally:
                    if plt.fignum_exists(fig.number):
                        plt.close(fig)

        # Duration distribution
        if res["duration_fit"] and hasattr(res["duration_fit"], "power_law") and res["duration_fit"].power_law:
            if hasattr(res["duration_fit"], "data") and len(res["duration_fit"].data) > 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                try:
                    res["duration_fit"].plot_ccdf(color=color, linewidth=1.5, marker="o", markersize=4, label="Empirical", ax=ax)
                    res["duration_fit"].power_law.plot_ccdf(color="black", linestyle="--", linewidth=2,
                                                           label=f"Fit α={res['duration_alpha']:.2f}, xmin={res['duration_fit'].xmin:.1f}", ax=ax)
                    ax.set_title(f"Duration Distribution - {bw_label_us:.0f}µs {groupname}")
                    ax.set_xlabel("Avalanche Duration (t, bins)")
                    ax.set_ylabel("CCDF P(T≥t)")
                    ax.legend()
                    ax.grid(True, which="both", ls="--", alpha=0.3)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    plt.tight_layout()
                    plt.savefig(f"{outputdir}/duration_dist_{bw_label_us:.0f}us.png", dpi=300)
                except Exception:
                    pass
                finally:
                    if plt.fignum_exists(fig.number):
                        plt.close(fig)


def plot_all_learning_accuracy_curves(all_learning_data, imid_param_values, ei_ratio_param_values, 
                                       condition_map, output_dir_base):
    """
    Plot comparative learning curves across conditions with mean and SEM.

    Parameters
    ----------
    all_learning_data : array
        Learning curve data (Imid × EI × Repetitions × training_sizes)
    imid_param_values : array
        Imid values tested
    ei_ratio_param_values : array
        E/I ratio values tested
    condition_map : dict
        Mapping from E/I ratios to condition names
    output_dir_base : str
        Output directory
    """
    os.makedirs(output_dir_base, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    colormap = {"subcritical": "blue", "critical": "green", "supercritical": "red"}
    plot_count = 0

    for i_idx, imid_val in enumerate(imid_param_values):
        for j_idx, ei_val in enumerate(ei_ratio_param_values):
            # Aggregate curves across repetitions
            aggregated_curves = {}
            for rep_idx in range(len(all_learning_data[i_idx][j_idx])):
                learning_curve = all_learning_data[i_idx][j_idx][rep_idx]
                if learning_curve:
                    for n_samples, acc in learning_curve.items():
                        if n_samples not in aggregated_curves:
                            aggregated_curves[n_samples] = []
                        aggregated_curves[n_samples].append(acc)

            if aggregated_curves:
                sorted_samples = sorted(aggregated_curves.keys())
                mean_accuracies = np.array([np.mean(aggregated_curves[s]) for s in sorted_samples])
                std_errors = np.array([np.std(aggregated_curves[s], ddof=1) / np.sqrt(len(aggregated_curves[s])) 
                                      for s in sorted_samples])

                condition_name = condition_map.get(ei_val, f"EI_{ei_val:.3f}")
                current_color = colormap.get(condition_name, "black")

                ax.plot(sorted_samples, mean_accuracies, marker="o", linestyle="-", 
                       color=current_color, label=condition_name.capitalize(), markersize=5)
                ax.fill_between(sorted_samples, mean_accuracies - std_errors, 
                               mean_accuracies + std_errors, color=current_color, alpha=0.2)
                plot_count += 1

    if plot_count == 0:
        ax.text(0.5, 0.5, "No learning curve data to plot.", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.set_xlabel("Number of Training Samples")
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification Accuracy Across Conditions (Mean ± SEM)")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", title="Condition")
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_filename = f"{output_dir_base}/comparative_learning_accuracy_curves.png"
    plt.savefig(plot_filename, dpi=300)
    if plt.fignum_exists(fig.number):
        plt.close(fig)
    print(f"Saved comparative learning accuracy curves plot to {plot_filename}")


def generate_summary_plots(imid_vals, ei_ratio_vals, metrics_data, output_dir_base):
    """
    Generate summary plots (phase diagrams) for all metrics.
    Wrapper function that delegates to 1D or 2D plotting based on parameter sweep type.

    Parameters
    ----------
    imid_vals : array
        Input current values
    ei_ratio_vals : array
        E/I ratio values
    metrics_data : dict
        Dictionary containing mean and std for each metric
    output_dir_base : str
        Output directory
    """
    os.makedirs(output_dir_base, exist_ok=True)

    is_1d_sweep_on_ei = (len(imid_vals) == 1) and (len(ei_ratio_vals) > 1)
    is_1d_sweep_on_imid = (len(imid_vals) > 1) and (len(ei_ratio_vals) == 1)
    is_2d_sweep = (len(imid_vals) > 1) and (len(ei_ratio_vals) > 1)

    if is_2d_sweep:
        print("--- Generating 2D summary plots (heatmaps)... ---")
        _plot_2d_heatmaps(imid_vals, ei_ratio_vals, metrics_data, output_dir_base)
    elif is_1d_sweep_on_ei:
        print(f"--- Generating 1D summary plots vs. E/I Ratio (at Imid = {imid_vals[0]} nA)... ---")
        _plot_1d_graphs(xdata=ei_ratio_vals, xlabel="E/I Conductance Ratio (gE,max/gI,max)", 
                       metrics_data=metrics_data, output_dir_base=output_dir_base, 
                       param_str=f"Imid_{imid_vals[0]:.3f}nA")
    elif is_1d_sweep_on_imid:
        print(f"--- Generating 1D summary plots vs. Imid (at E/I Ratio = {ei_ratio_vals[0]})... ---")
        _plot_1d_graphs(xdata=imid_vals, xlabel="Input Current Imid (nA)", 
                       metrics_data=metrics_data, output_dir_base=output_dir_base, 
                       param_str=f"EIRatio_{ei_ratio_vals[0]:.3f}")
    else:
        print("--- Skipping summary plot generation (not a 1D or 2D sweep). ---")


def _plot_1d_graphs(xdata, xlabel, metrics_data, output_dir_base, param_str=""):
    """
    Generate 1D line plots with error bars for each metric.

    Parameters
    ----------
    xdata : array
        X-axis data (either Imid or E/I ratio)
    xlabel : str
        X-axis label
    metrics_data : dict
        Dictionary of metrics with mean and std
    output_dir_base : str
        Output directory
    param_str : str
        Parameter string for filename
    """
    for key, (data, ylabel, error_data) in metrics_data.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_data = data.flatten()

        if np.all(np.isnan(plot_data)):
            ax.text(0.5, 0.5, f"No valid data for {key}", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.errorbar(xdata, plot_data, yerr=error_data.flatten(), 
                       marker="o", linestyle="None", color="royalblue", 
                       capsize=5, ecolor="lightgray", elinewidth=3)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.6)

        ax.set_title(f"{ylabel} vs. {xlabel.split()[0]}")
        plt.tight_layout()

        safe_key = key.replace(" ", "_").replace("/", "_")
        plt.savefig(f"{output_dir_base}/phase_diagram_{safe_key}_{param_str}.png", dpi=300)
        if plt.fignum_exists(fig.number):
            plt.close(fig)


def _plot_2d_heatmaps(imid_vals, ei_ratio_vals, metrics_data, output_dir_base):
    """
    Generate 2D heatmap phase diagrams for each metric.

    Parameters
    ----------
    imid_vals : array
        Input current values
    ei_ratio_vals : array
        E/I ratio values
    metrics_data : dict
        Dictionary of metrics with mean and std
    output_dir_base : str
        Output directory
    """
    X, Y = np.meshgrid(ei_ratio_vals, imid_vals)

    # Plot Firing Rate
    fr_data, fr_label, _ = metrics_data["firing_rate"]
    fig_rate, ax_rate = plt.subplots(figsize=(8, 6))
    fr_masked = np.ma.masked_invalid(fr_data)
    if not fr_masked.mask.all():
        contour_rate = ax_rate.contourf(X, Y, fr_masked, levels=50, cmap="viridis", extend="both")
        fig_rate.colorbar(contour_rate, ax=ax_rate, label=fr_label)
    else:
        ax_rate.text(0.5, 0.5, "No FR data", ha="center", va="center", transform=ax_rate.transAxes)
    ax_rate.set_xlabel("E/I Conductance Ratio (gE,max/gI,max)")
    ax_rate.set_ylabel("Input Current Imid (nA)")
    ax_rate.set_title(f"Phase Diagram: {fr_label}")
    plt.tight_layout()
    plt.savefig(f"{output_dir_base}/phase_diagram_firing_rate.png", dpi=300)
    if plt.fignum_exists(fig_rate.number):
        plt.close(fig_rate)

    # Plot CV
    cv_data, cv_label, _ = metrics_data["cv"]
    fig_cv, ax_cv = plt.subplots(figsize=(8, 6))
    cvs_masked = np.ma.masked_invalid(cv_data)
    if not cvs_masked.mask.all():
        norm_cv = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=2.0)
        contour_cv = ax_cv.contourf(X, Y, cvs_masked, levels=50, cmap="coolwarm", norm=norm_cv, extend="both")
        if np.any(cvs_masked < 1.0) and np.any(cvs_masked > 1.0):
            ax_cv.contour(X, Y, cvs_masked, levels=[1.0], colors="black", linestyles="--", linewidths=1.5)
        fig_cv.colorbar(contour_cv, ax=ax_cv, label=cv_label)
    else:
        ax_cv.text(0.5, 0.5, "No CV data", ha="center", va="center", transform=ax_cv.transAxes)
    ax_cv.set_xlabel("E/I Conductance Ratio (gE,max/gI,max)")
    ax_cv.set_ylabel("Input Current Imid (nA)")
    ax_cv.set_title(f"Phase Diagram: {cv_label}")
    plt.tight_layout()
    plt.savefig(f"{output_dir_base}/phase_diagram_overall_cv.png", dpi=300)
    if plt.fignum_exists(fig_cv.number):
        plt.close(fig_cv)

    # Plot Branching Parameter (Sigma)
    sigma_data, sigma_label, _ = metrics_data["sigma"]
    fig_sigma, ax_sigma = plt.subplots(figsize=(8, 6))
    sigmas_masked = np.ma.masked_invalid(sigma_data)
    if not sigmas_masked.mask.all():
        norm_sigma = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=np.nanmin(sigmas_masked), vmax=np.nanmax(sigmas_masked))
        contour_sigma = ax_sigma.contourf(X, Y, sigmas_masked, levels=50, cmap="coolwarm", norm=norm_sigma, extend="both")
        if np.any(sigmas_masked < 1.0) and np.any(sigmas_masked > 1.0):
            ax_sigma.contour(X, Y, sigmas_masked, levels=[1.0], colors="black", linestyles="--", linewidths=1.5)
        fig_sigma.colorbar(contour_sigma, ax=ax_sigma, label=sigma_label)
    else:
        ax_sigma.text(0.5, 0.5, "No Sigma data", ha="center", va="center", transform=ax_sigma.transAxes)
    ax_sigma.set_xlabel("E/I Conductance Ratio (gE,max/gI,max)")
    ax_sigma.set_ylabel("Input Current Imid (nA)")
    ax_sigma.set_title(f"Phase Diagram: {sigma_label}")
    plt.tight_layout()
    plt.savefig(f"{output_dir_base}/phase_diagram_sigma.png", dpi=300)
    if plt.fignum_exists(fig_sigma.number):
        plt.close(fig_sigma)

    # Plot RC Accuracy
    rc_acc_data, rc_acc_label, _ = metrics_data["rc_accuracy"]
    fig_rc_acc, ax_rc_acc = plt.subplots(figsize=(8, 6))
    rc_acc_masked = np.ma.masked_invalid(rc_acc_data)
    if not rc_acc_masked.mask.all():
        contour_rc_acc = ax_rc_acc.contourf(X, Y, rc_acc_masked, levels=50, cmap="magma", extend="both")
        fig_rc_acc.colorbar(contour_rc_acc, ax=ax_rc_acc, label=rc_acc_label)
    else:
        ax_rc_acc.text(0.5, 0.5, "No RC Accuracy data", ha="center", va="center", transform=ax_rc_acc.transAxes)
    ax_rc_acc.set_xlabel("E/I Conductance Ratio (gE,max/gI,max)")
    ax_rc_acc.set_ylabel("Input Current Imid (nA)")
    ax_rc_acc.set_title(f"Phase Diagram: {rc_acc_label}")
    plt.tight_layout()
    plt.savefig(f"{output_dir_base}/phase_diagram_rc_accuracy.png", dpi=300)
    if plt.fignum_exists(fig_rc_acc.number):
        plt.close(fig_rc_acc)

    # Plot Samples to Threshold
    from config import QUICKNESS_TARGET_ACCURACY
    sampl_thresh_data, sampl_thresh_label, _ = metrics_data["samples_to_threshold"]
    fig_sampl_thresh, ax_sampl_thresh = plt.subplots(figsize=(8, 6))
    sampl_thresh_masked = np.ma.masked_invalid(sampl_thresh_data)
    if not sampl_thresh_masked.mask.all():
        contour_sampl_thresh = ax_sampl_thresh.contourf(X, Y, sampl_thresh_masked, levels=50, cmap="viridis_r", extend="max")
        fig_sampl_thresh.colorbar(contour_sampl_thresh, ax=ax_sampl_thresh, label=sampl_thresh_label)
    else:
        ax_sampl_thresh.text(0.5, 0.5, "No Samples to Threshold data", ha="center", va="center", transform=ax_sampl_thresh.transAxes)
    ax_sampl_thresh.set_xlabel("E/I Conductance Ratio (gE,max/gI,max)")
    ax_sampl_thresh.set_ylabel("Input Current Imid (nA)")
    ax_sampl_thresh.set_title(f"Phase Diagram: Learning Quickness (Samples to {QUICKNESS_TARGET_ACCURACY*100:.0f}% Acc)")
    plt.tight_layout()
    plt.savefig(f"{output_dir_base}/phase_diagram_samples_to_threshold.png", dpi=300)
    if plt.fignum_exists(fig_sampl_thresh.number):
        plt.close(fig_sampl_thresh)

    # Plot Accuracy at Fixed Samples
    from config import QUICKNESS_FIXED_SAMPLE_SIZE
    acc_fixed_data, acc_fixed_label, _ = metrics_data["accuracy_at_fixed_samples"]
    fig_acc_fixed, ax_acc_fixed = plt.subplots(figsize=(8, 6))
    acc_fixed_masked = np.ma.masked_invalid(acc_fixed_data)
    if not acc_fixed_masked.mask.all():
        contour_acc_f = ax_acc_fixed.contourf(X, Y, acc_fixed_masked, levels=50, cmap="magma", extend="both")
        fig_acc_fixed.colorbar(contour_acc_f, ax=ax_acc_fixed, label=acc_fixed_label)
    else:
        ax_acc_fixed.text(0.5, 0.5, f"No Accuracy data at {QUICKNESS_FIXED_SAMPLE_SIZE} samples", 
                         ha="center", va="center", transform=ax_acc_fixed.transAxes)
    ax_acc_fixed.set_xlabel("E/I Conductance Ratio (gE,max/gI,max)")
    ax_acc_fixed.set_ylabel("Input Current Imid (nA)")
    ax_acc_fixed.set_title(f"Phase Diagram: Learning Quickness (Acc. at {QUICKNESS_FIXED_SAMPLE_SIZE} samples)")
    plt.tight_layout()
    plt.savefig(f"{output_dir_base}/phase_diagram_accuracy_at_fixed_samples.png", dpi=300)
    if plt.fignum_exists(fig_acc_fixed.number):
        plt.close(fig_acc_fixed)
