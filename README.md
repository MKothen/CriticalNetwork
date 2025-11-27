# Adaptive Exponential Network Simulation

This codebase simulates an excitatory-inhibitory spiking neural network using Adaptive Exponential Integrate-and-Fire (AdEx) neurons with Brian2. The network can be used for:
- Studying criticality and avalanche dynamics
- Reservoir computing with MNIST classification
- Phase diagram exploration of E/I balance

## 📁 File Organization

```
├── config.py          → All parameters and settings
├── data_utils.py      → MNIST data loading and preprocessing
├── network_model.py   → Network equations and setup
├── analysis.py        → Analysis functions (CV, avalanches, branching parameter)
├── reservoir.py       → Reservoir computing functions
├── plotting.py        → All visualization functions
├── statistics.py      → Statistical testing (ANOVA, Kruskal-Wallis)
├── main_simulation.py → Main execution script
└── simple_example.py  → Minimal working example

Documentation files:
├── README.md                   → This file
├── SETUP_INSTRUCTIONS.md       → Installation and troubleshooting
├── CONSOLE_OUTPUT_GUIDE.md     → Understanding simulation output
├── FUNCTION_REFERENCE.md       → Detailed function documentation
├── ORGANIZATION_GUIDE.md       → Code structure explanation
└── FILE_ORGANIZATION_TREE.md   → Visual file structure
```

**All modules are now complete and ready to use!** ✓

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install brian2 numpy scipy matplotlib scikit-learn pandas powerlaw statsmodels
```

### 2. Test Installation with Simple Example

```bash
python simple_example.py
```

This will:
- Create a network with 1000 neurons (800 excitatory, 200 inhibitory)
- Run for ~7 seconds of simulation time
- Generate basic plots showing network activity
- Save output as `simple_example_output.png`

**Expected runtime**: 2-5 minutes

### 3. Modify Parameters

Open `config.py` and adjust parameters. For a quick test:

```python
# Network size
N_TOTAL_NEURONS = 500          # Reduced from 1000 for faster testing

# Simulation duration
SIM_RUNTIME = 10 * second      # Reduced from 20 seconds

# Experimental conditions
Imid_values_nA = np.array([0.3333])
EI_ratio_values = np.array([0.001, 0.385, 1.0])
NUM_REPETITIONS = 2            # Reduced from 8 for testing

# RC parameters
NUM_TRAIN_SAMPLES_MAX = 50     # Reduced from 100
NUM_TEST_SAMPLES_MAX = 50
```

### 4. Run Full Simulation

```bash
python main_simulation.py
```

**Expected runtime**: 
- With default parameters (N=1000, 8 reps): **3-6 hours**
- With reduced parameters (N=500, 2 reps): **20-40 minutes**

## 📊 Understanding the Code Structure

### Configuration (`config.py`)

This file contains **all parameters** in one place:

- **Network structure**: Number of neurons, connectivity
- **Neuron model**: AdEx parameters (capacitance, conductances, thresholds)
- **Synaptic dynamics**: Time constants, reversal potentials
- **Simulation settings**: Duration, timestep, analysis windows
- **RC parameters**: Input encoding, training samples, regularization
- **Experimental design**: Conditions to test, repetitions

**Tip for students**: Start here! Modify one parameter at a time to see how it affects results.

### Network Model (`network_model.py`)

Contains the mathematical equations for:

1. **AdEx neuron dynamics**:
   ```
   dV/dt = (I_leak + I_exp - I_syn + I_noise - A + I_stim) / C
   ```
   - `V`: Membrane potential
   - `A`: Adaptation current
   - `I_syn`: Synaptic currents (excitatory + inhibitory)
   - `I_noise`: Background fluctuations
   - `I_stim`: External stimulus (for RC task)

2. **Synaptic dynamics**: Double-exponential synapses with rise and decay

3. **Network creation**: Builds E/I populations with random connectivity

**Key function**: `create_network()` - Returns a complete, ready-to-use network

### Data Utilities (`data_utils.py`)

Handles MNIST dataset:
- Downloads and caches data (~50 MB, only first run)
- Binarizes images (pixel > 0.5 → ON, else → OFF)
- Splits into train/test sets (75%/25%)
- One-hot encodes labels

### Analysis (`analysis.py`)

Core analysis functions:

1. **Coefficient of Variation (CV)**:
   - Measures regularity of spiking
   - CV < 1: Regular firing
   - CV ≈ 1: Poisson-like (random)
   - CV > 1: Irregular/bursty

2. **Inter-Event Interval (IEI)**:
   - Average time between network events
   - Used for adaptive bin width in avalanche analysis

3. **Avalanche Analysis**:
   - Detects neuronal avalanches
   - Fits power-law distributions
   - Calculates scaling exponents (α for size, τ for duration, γ for scaling)

4. **Branching Parameter (σ)**:
   - σ = ⟨n_{t+1}⟩ / ⟨n_t⟩
   - **σ < 1**: Subcritical (activity dies out)
   - **σ ≈ 1**: Critical (balanced, power-law dynamics)
   - **σ > 1**: Supercritical (activity grows)

### Reservoir Computing (`reservoir.py`)

Uses the spiking network as a computational reservoir:

1. **Input encoding**: Maps MNIST pixels → neurons via random projection
2. **State extraction**: Reads out smoothed firing rates at specific times
3. **Readout training**: Linear Ridge regression classifier
4. **Evaluation**: Tests accuracy on unseen digits

**Key insight**: The recurrent network doesn't learn - only the readout weights are trained!

### Plotting (`plotting.py`)

All visualization functions:
- `plot_basic_activity()`: 6-panel comprehensive activity plot
- `plot_initial_raster()`: Detailed raster of first 5 seconds
- `plot_detailed_stimulus_raster()`: RC trial visualization
- `plot_neural_manifold()`: PCA visualization of digit representations
- `generate_summary_plots()`: Phase diagrams (heatmaps or line plots)
- `plot_all_learning_accuracy_curves()`: Learning curves across conditions

### Statistics (`statistics.py`)

Statistical hypothesis testing:
- Checks assumptions (normality, homogeneity of variance)
- Runs appropriate tests:
  - **ANOVA** + Tukey HSD (if assumptions met)
  - **Kruskal-Wallis** + Mann-Whitney U (if assumptions violated)
- Compares conditions across metrics

### Main Simulation (`main_simulation.py`)

Orchestrates the entire workflow:
1. Loads MNIST data
2. For each condition (Imid × E/I ratio × repetition):
   - Creates network
   - Runs intrinsic dynamics
   - Analyzes activity
   - Runs RC task
   - Generates plots
3. Aggregates results
4. Generates summary plots
5. Runs statistical tests
6. Saves to Excel



## 📈 Output Files and Plots

### After running `main_simulation.py`:

**Summary directory** (`results_phase_diagram_summary/`):
- `simulation_summary.xlsx`: All metrics in tabular format
- `phase_diagram_*.png`: Heatmaps/line plots of metrics across conditions
- `comparative_learning_accuracy_curves.png`: RC performance comparison
- `neural_manifold_pca.png`: PCA visualization of digit classes
- `aggregated_avalanche_ccdf_*.png`: Power-law distributions

**Individual run directories** (`results_phase_diagram_runs/[condition]_Rep[X]/`):
- `basic_activity_plot.png`: 6-panel overview of network dynamics
- `initial_5s_raster.png`: Detailed raster plot
- `detailed_stimulus_raster.png`: RC trial visualization
- `individual_avalanche_plots/`: Avalanche distributions



**Quick commands to get started:**
```bash
# Install dependencies
pip install brian2 numpy scipy matplotlib scikit-learn pandas powerlaw statsmodels

# Test installation
python simple_example.py

# Run small test
# (first edit config.py to reduce N_TOTAL_NEURONS, NUM_REPETITIONS)
python main_simulation.py
```


