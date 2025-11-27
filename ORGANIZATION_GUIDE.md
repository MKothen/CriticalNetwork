# Code Organization Guide

## Overview

The original monolithic script (~3500 lines) has been successfully reorganized into modular files. **All modules are now complete and ready to use!** ✓

---

## Complete File Structure

### ✅ Core Modules (ALL COMPLETE)

1. **config.py** (~150 lines)
   - All parameters in one place
   - Network architecture settings
   - Neuron model parameters (AdEx)
   - Synaptic parameters
   - Simulation settings
   - RC parameters  
   - Experimental design
   - Plotting style configuration

2. **data_utils.py** (~140 lines)
   - `load_and_preprocess_mnist()`: Downloads and prepares MNIST dataset
   - `calculate_samples_to_reach_threshold()`: Learning curve analysis
   - `get_accuracy_at_fixed_samples()`: Performance at fixed sample size
   - Data caching for efficiency

3. **network_model.py** (~200 lines)
   - `get_neuron_equations()`: AdEx neuron model equations
   - `get_synapse_equations()`: Double-exponential synaptic dynamics
   - `initialize_neuron_population()`: Set up heterogeneous neurons
   - `create_network()`: Build complete E/I network with all connections

4. **analysis.py** (~400 lines)
   - `calculate_average_iei()`: Inter-event interval calculation
   - `calculate_cv()`: Coefficient of variation for spike regularity
   - `calculate_live_cv()`: Time-resolved CV dynamics
   - `calculate_branching_parameter()`: Criticality measure (σ)
   - `analyze_bin_width()`: Avalanche detection and power-law fitting
   - `analyze_model_spikes()`: Complete avalanche analysis pipeline

5. **reservoir.py** (~350 lines)
   - `create_input_projection_map()`: Random pixel-to-neuron mapping
   - `calculate_per_neuron_smoothed_rates()`: Gaussian-smoothed firing rates
   - `run_rc_simulation_for_input()`: Present stimulus and extract state
   - `train_readout_weights()`: Ridge regression classifier training
   - `evaluate_readout_performance()`: Test set accuracy evaluation

6. **plotting.py** (~1200 lines)
   - `plot_basic_activity()`: 6-panel comprehensive activity plot
   - `plot_initial_raster()`: Detailed raster of first 5 seconds
   - `plot_detailed_stimulus_raster()`: RC trial visualization with digit labels
   - `plot_neural_manifold()`: PCA visualization of network states
   - `plot_all_learning_accuracy_curves()`: Learning curves across conditions
   - `generate_summary_plots()`: Phase diagrams (wrapper function)
   - `_plot_1d_graphs()`: Line plots for 1D parameter sweeps
   - `_plot_2d_heatmaps()`: Heatmaps for 2D parameter spaces

7. **statistics.py** (~250 lines)
   - `run_and_print_statistical_tests()`: 
     - Checks assumptions (Shapiro-Wilk, Levene's test)
     - Runs ANOVA or Kruskal-Wallis as appropriate
     - Post-hoc tests (Tukey HSD or Mann-Whitney U)
   - `run_learning_curve_statistics()`: 
     - Compares learning curves at each sample size
     - Identifies when conditions diverge

8. **main_simulation.py** (~650 lines)
   - Main execution script that orchestrates everything
   - Parameter sweep loops (Imid × E/I ratio × repetitions)
   - Network creation and initialization
   - Intrinsic dynamics simulation (no input)
   - Avalanche analysis calls
   - Reservoir computing task execution
   - Results aggregation across conditions
   - Excel export with all metrics
   - Final plotting and statistical testing

---

## Complete Documentation Files

9. **README.md**
   - Installation instructions
   - Quick start guide
   - Module explanations
   - Example experiments
   - Troubleshooting guide
   - Thesis writing tips

10. **SETUP_INSTRUCTIONS.md**
   - Step-by-step installation
   - Dependency installation
   - Quick test procedures
   - Performance benchmarks
   - Detailed troubleshooting

11. **CONSOLE_OUTPUT_GUIDE.md** ⭐ NEW
   - Explains every line of console output
   - Metric interpretation
   - Progress tracking
   - Common warnings and their meanings
   - Expected runtimes

12. **FUNCTION_REFERENCE.md** ⭐ NEW
   - Detailed documentation for ALL functions
   - Function signatures and parameters
   - Return value explanations
   - Code examples with outputs
   - Usage patterns

13. **ORGANIZATION_GUIDE.md** (this file)
   - File-by-file breakdown
   - Benefits of organization
   - Workflow recommendations

14. **ARCHITECTURE.md**
   - Dependency graph
   - Data flow diagram
   - Function call hierarchy
   - Memory usage breakdown
   - Extension points

15. **FILE_ORGANIZATION_TREE.md**
   - Visual file structure
   - Module interconnections
   - Quick reference table

---

## What Each Module Does

### Configuration Layer

**config.py**
- **Purpose**: Single source of truth for all parameters
- **No dependencies**: Uses only Brian2 and NumPy
- **Used by**: ALL other modules
- **Key insight**: Change parameters here, not in code

### Data Layer

**data_utils.py**
- **Purpose**: Handles external data (MNIST)
- **Dependencies**: NumPy, Pandas, scikit-learn
- **Used by**: main_simulation.py, reservoir.py
- **Key insight**: Data loading is independent of simulation

### Model Layer

**network_model.py**
- **Purpose**: Defines neural network equations and structure
- **Dependencies**: Brian2, NumPy, config
- **Used by**: main_simulation.py, simple_example.py
- **Key insight**: Clean separation between model definition and analysis

### Analysis Layer

**analysis.py**
- **Purpose**: Analyzes neural activity patterns
- **Dependencies**: NumPy, SciPy, Pandas, powerlaw
- **Used by**: main_simulation.py
- **Key insight**: Reusable analysis functions work on any spike data

### Application Layer

**reservoir.py**
- **Purpose**: Applies network to computational task
- **Dependencies**: Brian2, scikit-learn, config, data_utils
- **Used by**: main_simulation.py
- **Key insight**: RC task is cleanly separated from network dynamics

### Visualization Layer

**plotting.py**
- **Purpose**: All figure generation
- **Dependencies**: Matplotlib, NumPy, Pandas, scikit-learn, config
- **Used by**: main_simulation.py
- **Key insight**: Plotting is separate from computation

### Statistical Layer

**statistics.py**
- **Purpose**: Hypothesis testing
- **Dependencies**: SciPy, statsmodels, NumPy, Pandas
- **Used by**: main_simulation.py
- **Key insight**: Statistical tests are optional and modular

### Orchestration Layer

**main_simulation.py**
- **Purpose**: Ties everything together
- **Dependencies**: ALL other modules
- **Used by**: User (run directly)
- **Key insight**: Shows high-level workflow clearly

---




### Recommended Reading Order

1. **README.md** - Overall understanding
2. **SETUP_INSTRUCTIONS.md** - Get it running
3. **CONSOLE_OUTPUT_GUIDE.md** - Understand output
4. **config.py** - See all parameters
5. **simple_example.py** - Understand basic workflow
6. **FUNCTION_REFERENCE.md** - Deep dive into functions
7. **ARCHITECTURE.md** - System design

---



---

## Design Principles Applied

### 1. Separation of Concerns
Each module has one responsibility:
- config.py: Parameters only
- network_model.py: Network construction only
- analysis.py: Analysis only
- etc.

### 2. DRY (Don't Repeat Yourself)
- Common parameters in config.py (used everywhere)
- Reusable functions (e.g., calculate_cv)
- No duplicated code

### 3. Single Source of Truth
- All parameters in config.py
- No "magic numbers" in code
- Easy to ensure consistency

### 4. Clear Naming
- Functions named after what they do
- Variables have descriptive names
- No cryptic abbreviations

### 5. Comprehensive Documentation
- Docstrings for all functions
- Inline comments for complex logic
- Separate documentation files
- Examples in README

### 6. Progressive Complexity
- simple_example.py: Minimal
- main_simulation.py: Full featured
- Clear learning path





---

## Dependency Management

### Import Structure

```
main_simulation.py
│
├── imports config
├── imports network_model (which imports config)
├── imports data_utils
├── imports analysis
├── imports reservoir (which imports config, data_utils)
├── imports plotting (which imports config)
└── imports statistics

simple_example.py
│
├── imports config
├── imports network_model
├── imports analysis (minimal)
├── imports data_utils (for RC, optional)
└── imports reservoir (for RC, optional)
```

**Key insight**: config.py has no dependencies on other project files, so it's safe to import everywhere.

---

