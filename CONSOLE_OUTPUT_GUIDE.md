# Console Output Guide

## Understanding the Simulation Output

When you run `main_simulation.py`, you'll see detailed console output tracking the simulation's progress. This guide explains what each section means and what to expect.

---

## Startup Phase

### Initial Banner
```
================================================================================
               NEURAL NETWORK SIMULATION - PARAMETER SWEEP
================================================================================
Configuration: 1 Imid values × 3 E/I ratios × 8 repetitions
Total simulations: 24
================================================================================
```

**What this means:**
- The simulation will run **24 total network simulations** (1 input current × 3 E/I balance conditions × 8 repetitions)
- Each simulation tests a different network configuration

---

## Data Loading

### MNIST Data Loading
```
Loading MNIST dataset...
Loading MNIST data (max 100 train, 100 test)...
Loaded 100 training samples, 100 test samples.
X_train_sample shape: (100, 784), y_train_onehot shape: (100, 10)
✓ MNIST loaded: 100 train / 100 test samples
```

**What this means:**
- The MNIST handwritten digit dataset is being downloaded and preprocessed
- Each image has 784 pixels (28×28 flattened)
- Labels are one-hot encoded (10 classes for digits 0-9)
- This only happens once - subsequent runs use cached data

---

## Main Simulation Loop

### Run Header
```
================================================================================
RUN 1/24: Subcritical_Rep0
 Imid=0.333 nA, E/I ratio=0.001
================================================================================
```

**What this means:**
- **Run number**: Current simulation out of total
- **Condition name**: "Subcritical", "Critical", or "Supercritical" based on E/I ratio
- **Repetition**: Which repetition of this condition (Rep0 through Rep7)
- **Parameters**: Background current (Imid) and excitation/inhibition balance

### Network Creation
```
Creating network: 800 Exc + 200 Inh = 1000 total
```

**What this means:**
- 800 excitatory neurons (80%)
- 200 inhibitory neurons (20%)
- Total of 1000 neurons in the recurrent network

### Simulation Progress
```
Running initial settling: 0.3 s
Starting simulation...
... (Brian2 simulation progress reports)
Running main simulation: 20.0 s
Starting simulation...
... (Brian2 simulation progress reports)
✓ Intrinsic dynamics simulation complete
```

**What this means:**
- **Initial settling (0.3s)**: Network stabilizes from random initial conditions
- **Main simulation (20s)**: Network runs freely without external input
- Brian2 shows progress bars and time estimates
- **Expected duration**: 1-3 minutes per simulation (depends on your CPU)

### Activity Analysis
```
Analyzing network activity...
 Mean FR: 3.45 Hz
 CV: 1.234
```

**What this means:**
- **Mean FR (Firing Rate)**: Average spikes per second per neuron
  - Typical range: 1-10 Hz
  - Too low (<0.5 Hz): Network is too quiet
  - Too high (>20 Hz): Network might be hyperactive
- **CV (Coefficient of Variation)**: Irregularity of spiking
  - CV < 1: Regular, clock-like firing
  - CV ≈ 1: Poisson-like (random) firing
  - CV > 1: Irregular, bursty firing

### Avalanche Analysis
```
Analyzing avalanches...
 Adaptive bin width: 45.23 ms (2×IEI)
 Branching parameter (σ): 0.987
 Size exponent (α): 1.52
 Avalanches detected: 1245
```

**What this means:**
- **Adaptive bin width**: Time bins for detecting avalanches (based on Inter-Event Interval)
- **Branching parameter (σ)**: Key criticality measure
  - σ < 1: **Subcritical** (activity dies out)
  - σ ≈ 1: **Critical** (balanced, power-law dynamics)
  - σ > 1: **Supercritical** (activity grows)
- **Size exponent (α)**: Power-law exponent for avalanche sizes
  - Critical networks typically have α ≈ 1.5
- **Avalanches detected**: Number of discrete avalanche events found

### Reservoir Computing Task
```
Running reservoir computing task...
INFO: Calculated average number of ON pixels per image: 145.67
INFO: Total trial duration for compensation calculation: 110 ms
INFO: Calculated statistical compensation current: 0.263 pA
 Reducing background current for RC task from 0.3333 nA to 0.3331 nA.
```

**What this means:**
- The network is being tested on MNIST digit classification
- Background current is slightly reduced to compensate for input
- This prevents the network from becoming over-excited during stimulus presentation

```
 Collecting 100 training states...
 Processed 50/100 train images
✓ Collected training states: (100, 1000)
 Collecting 100 test states...
 Processed 50/100 test images
✓ Collected test states: (100, 1000)
```

**What this means:**
- Each MNIST image is presented to the network
- The network's firing rates at a specific time point are recorded as a "state vector"
- State vectors have length 1000 (one value per neuron)
- **Expected duration**: 5-15 minutes for 100+100 images

### Readout Training
```
Training readout with 33 samples.
Reservoir state dim: 1000
Readout weights W_out shape: (1001, 10)
 RC accuracy with 33 samples: 0.3200
Training readout with 66 samples.
Reservoir state dim: 1000
Readout weights W_out shape: (1001, 10)
 RC accuracy with 66 samples: 0.4500
Training readout with 100 samples.
Reservoir state dim: 1000
Readout weights W_out shape: (1001, 10)
 RC accuracy with 100 samples: 0.5600
✓ Best RC accuracy: 0.5600 (100 samples)
```

**What this means:**
- A linear classifier is trained on the network's states to predict digits
- Training is done with different amounts of data (33, 66, 100 samples)
- **W_out shape (1001, 10)**: 1001 inputs (1000 neurons + 1 bias) → 10 outputs (digit classes)
- Accuracy increases with more training data
- **Typical accuracy**: 30-70% depending on network state
  - Random chance: 10% (10 classes)
  - Perfect: 100%

### Plotting
```
Generating plots...
✓ Plots saved
```

**What this means:**
- Individual plots for this run are saved to `results_phase_diagram_runs/Subcritical_Rep0/`
- Includes raster plots, voltage traces, population rates, etc.

---

## Post-Simulation Analysis

### Learning Curves
```
Plotting learning curves...
✓ Learning curves plotted
```

**What this means:**
- Accuracy vs. training set size is plotted for all conditions
- Shows how quickly each network state learns

### Neural Manifold
```
Plotting neural manifold...
✓ Neural manifold plotted
```

**What this means:**
- PCA visualization of how the network represents different digits
- Shows separation between digit classes in neural activity space

### Statistical Analysis
```
================================================================================
                    STATISTICAL ANALYSIS OF RESULTS
================================================================================

--- Analysis for Metric: Final Accuracy ---

--- Condition: Imid = 0.3333 nA ---

(1) Checking assumptions for parametric tests (ANOVA)...
 - Normality (Shapiro-Wilk) for group 'Subcritical': p = 0.1234 (Normal)
 - Normality (Shapiro-Wilk) for group 'Critical': p = 0.2345 (Normal)
 - Normality (Shapiro-Wilk) for group 'Supercritical': p = 0.3456 (Normal)
 - Homogeneity of Variances (Levene's Test): p = 0.4567 (Variances are Equal)

Conclusion: Assumptions for ANOVA are MET.

(2) Performing appropriate statistical test...

Using parametric test: One-Way ANOVA
 - F-statistic: 5.6789
 - P-value: 0.0123
 - Result: **Significant difference detected** among E/I ratio groups (p < 0.05).

 Post-Hoc Test: Tukey's HSD
        Multiple Comparison of Means - Tukey HSD, FWER=0.05         
====================================================================
   group1       group2    meandiff p-adj   lower   upper  reject
--------------------------------------------------------------------
 Critical   Subcritical   0.1234  0.0234  0.0123  0.2345   True
 Critical Supercritical   0.0987  0.0456  0.0012  0.1962   True
Subcritical Supercritical -0.0247 0.8901 -0.1358  0.0864  False
--------------------------------------------------------------------
```

**What this means:**
- Statistical tests compare performance across conditions
- **Shapiro-Wilk test**: Checks if data is normally distributed (p > 0.05: Data is normal)
- **Levene's test**: Checks if variance is equal across groups (p > 0.05: Equal variances)
- **ANOVA**: Tests if there's any difference between groups (p < 0.05: At least one group is different)
- **Tukey HSD**: Pairwise comparisons to see which specific groups differ ("reject = True": Those two groups are significantly different)

---

## Completion

```
================================================================================
                         SIMULATION COMPLETE!
================================================================================

Results saved to: results_phase_diagram_summary/
 - simulation_summary.xlsx
 - phase_diagram_*.png
 - comparative_learning_accuracy_curves.png
 - neural_manifold_pca.png
 - aggregated_avalanche_ccdf_*.png

Individual run results in: results_phase_diagram_runs/
================================================================================
```

**What this means:**
- All results are saved
- **Excel file**: Contains all numerical data in tabular format
- **Phase diagrams**: Heatmaps or line plots showing metrics across conditions
- **Learning curves**: How accuracy improves with training data
- **Neural manifold**: PCA visualization of digit representations
- **Avalanche plots**: Power-law distributions

---

## Troubleshooting Console Output

### Common Warnings

**Warning: Skipping E/I ratio <= 0**
- A configuration with invalid E/I ratio was attempted
- This is normal and can be ignored

**No avalanches detected**
- Network might be too quiet or too active
- Check firing rate (should be 1-10 Hz)
- Adjust `Imid_values_nA` if needed

**RC accuracy near random (≈10%)**
- Network states might not separate digit classes well
- This is expected for subcritical or supercritical states
- Critical state typically gives best performance

---

## Expected Total Runtime

For default parameters (N=1000, 8 reps, 3 conditions):
- **Per simulation**: 5-10 minutes
- **Total time**: 2-4 hours

For reduced parameters (N=500, 2 reps, 3 conditions):
- **Per simulation**: 2-5 minutes
- **Total time**: 12-30 minutes
