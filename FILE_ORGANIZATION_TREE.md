# Complete File Organization Structure

## Project Directory Layout

```
neural_network_simulation/
│
├── 📄 Core Modules (ALL COMPLETE ✅)
│   ├── config.py                    # All parameters and configuration
│   ├── data_utils.py                # MNIST loading and preprocessing  
│   ├── network_model.py             # AdEx network equations and setup
│   ├── analysis.py                  # CV, IEI, avalanche, branching analysis
│   ├── reservoir.py                 # Reservoir computing functions
│   ├── plotting.py                  # All visualization functions
│   ├── statistics.py                # Statistical testing (ANOVA, Kruskal-Wallis)
│   └── main_simulation.py           # Main execution script
│
├── 🧪 Examples (COMPLETE ✅)
│   └── simple_example.py            # Minimal working example
│
├── 📖 Documentation (COMPLETE ✅)
│   ├── README.md                    # Main documentation
│   ├── SETUP_INSTRUCTIONS.md        # Installation guide
│   ├── CONSOLE_OUTPUT_GUIDE.md      # Understanding simulation output ⭐ NEW
│   ├── FUNCTION_REFERENCE.md        # Detailed function docs ⭐ NEW
│   ├── ORGANIZATION_GUIDE.md        # Code structure explanation
│   ├── ARCHITECTURE.md              # Technical architecture details
│   └── FILE_ORGANIZATION_TREE.md    # This file
│
├── 📁 Generated Directories (Created at runtime)
│   ├── results_phase_diagram_runs/
│   │   ├── Subcritical_Rep0/
│   │   │   ├── basic_activity_plot.png
│   │   │   ├── initial_5s_raster.png
│   │   │   ├── detailed_stimulus_raster.png
│   │   │   └── (other plots...)
│   │   ├── Critical_Rep0/
│   │   ├── Supercritical_Rep0/
│   │   └── (additional repetitions...)
│   │
│   └── results_phase_diagram_summary/
│       ├── simulation_summary.xlsx
│       ├── phase_diagram_firing_rate.png
│       ├── phase_diagram_overall_cv.png
│       ├── phase_diagram_sigma.png
│       ├── phase_diagram_rc_accuracy.png
│       ├── comparative_learning_accuracy_curves.png
│       ├── neural_manifold_pca.png
│       └── aggregated_avalanche_ccdf_*.png
│
└── 📦 Data Cache (Created at runtime)
    └── ~/sklearn_datasets/
        └── openml/
            └── mnist_784/
                └── (MNIST data files)
```

---

## Module Interconnections

```
                    ┌─────────────────────────────────────┐
                    │      main_simulation.py             │
                    │  (Orchestrates everything)          │
                    └──────────────┬──────────────────────┘
                                   │
                   ┌───────────────┼───────────────┐
                   │               │               │
                   ▼               ▼               ▼
         ┌─────────────┐   ┌──────────┐   ┌──────────┐
         │ network_    │   │ analysis.│   │reservoir.│
         │ model.py    │   │ py       │   │ py       │
         └──────┬──────┘   └────┬─────┘   └────┬─────┘
                │               │              │
                │               │              ▼
                │               │        ┌──────────┐
                │               │        │ data_    │
                │               │        │ utils.py │
                │               │        └────┬─────┘
                │               │             │
                └───────────────┴─────────────┼──────────┐
                                │             │          │
                                ▼             ▼          ▼
                         ┌────────────────────────────────┐
                         │       config.py                │
                         │  (Configuration Layer)         │
                         └────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────┐
                    │                  │              │
                    ▼                  ▼              ▼
              ┌──────────┐       ┌──────────┐   ┌──────────┐
              │plotting. │       │statistics│   │  simple_ │
              │ py       │       │ .py      │   │  example │
              └──────────┘       └──────────┘   └──────────┘
```

---

## Execution Flow

### Full Simulation (`main_simulation.py`)

```
1. User edits config.py
   ↓
2. User runs: python main_simulation.py
   ↓
3. Initialize data storage arrays
   ↓
4. Load MNIST data (data_utils.py)
   ↓
5. FOR each parameter combination:
   │
   ├─→ Set random seed
   │   ↓
   ├─→ Create network (network_model.py)
   │   ├─ Create excitatory neurons
   │   ├─ Create inhibitory neurons
   │   ├─ Create synapses (E→E, E→I, I→E, I→I)
   │   └─ Initialize with heterogeneous parameters
   │   ↓
   ├─→ Run intrinsic dynamics simulation (Brian2)
   │   ├─ Initial settling (0.3s)
   │   └─ Main simulation (20s)
   │   ↓
   ├─→ Analyze network activity (analysis.py)
   │   ├─ Calculate firing rate
   │   ├─ Calculate CV
   │   ├─ Calculate IEI
   │   ├─ Detect avalanches
   │   ├─ Fit power laws
   │   └─ Calculate branching parameter σ
   │   ↓
   ├─→ Run RC task (reservoir.py)
   │   ├─ Create pixel-to-neuron projection map
   │   ├─ FOR each training image:
   │   │   ├─ Apply input currents
   │   │   └─ Extract network state (firing rates)
   │   ├─ FOR each test image:
   │   │   ├─ Apply input currents
   │   │   └─ Extract network state
   │   ├─ Train readout weights (Ridge regression)
   │   │   └─ Test multiple training set sizes
   │   └─ Evaluate on test set
   │   ↓
   ├─→ Generate individual plots (plotting.py)
   │   ├─ Basic activity plot (6 panels)
   │   ├─ Initial 5s raster
   │   └─ Detailed stimulus raster
   │   ↓
   └─→ Store results
   ↓
6. Aggregate results across all conditions
   ↓
7. Generate summary plots (plotting.py)
   ├─ Phase diagrams (1D or 2D)
   ├─ Learning curves
   ├─ Neural manifold (PCA)
   └─ Aggregated avalanche distributions
   ↓
8. Run statistical tests (statistics.py)
   ├─ Check assumptions
   ├─ ANOVA or Kruskal-Wallis
   └─ Post-hoc comparisons
   ↓
9. Save results to Excel
   ↓
10. Done! ✅
```

### Simple Example (`simple_example.py`)

```
1. User runs: python simple_example.py
   ↓
2. Define parameters (locally, not sweep)
   ↓
3. Create network (network_model.py)
   ↓
4. Add monitors
   ↓
5. Run simulation (20s + brief stimulus)
   ↓
6. Basic analysis (CV, firing rate)
   ↓
7. Create 3-panel plot
   ├─ Raster
   ├─ Voltage trace
   └─ Population rate
   ↓
8. Save plot and exit
```




---

## Quick Reference: Where to Find Things

| What you need | Where to look | File |
|--------------|---------------|------|
| Change network size | N_TOTAL_NEURONS | config.py |
| Change simulation time | SIM_RUNTIME | config.py |
| Change E/I balance | EI_ratio_values | config.py |
| Understand neuron model | AdEx equations | network_model.py |
| Understand CV metric | calculate_cv() | analysis.py |
| Understand avalanches | analyze_bin_width() | analysis.py |
| Understand RC task | Docstrings | reservoir.py |
| Change plot style | THESIS_STYLE | config.py |
| Find a bug | Function docstrings | All files |
| Speed up simulation | Reduce parameters | config.py |
| Add new analysis | New function | analysis.py |
| Understand console output | Line-by-line guide | CONSOLE_OUTPUT_GUIDE.md |
| Function details | Complete reference | FUNCTION_REFERENCE.md |

---

## File Size and Complexity

### Python Modules

| File | Lines | Functions | Complexity | Purpose |
|------|-------|-----------|------------|---------|
| config.py | ~150 | 0 | Simple | Parameters |
| data_utils.py | ~140 | 3 | Low | Data loading |
| network_model.py | ~200 | 4 | Medium | Network creation |
| analysis.py | ~400 | 6 | High | Activity analysis |
| reservoir.py | ~350 | 5 | Medium | RC task |
| plotting.py | ~1200 | 10+ | Medium | Visualization |
| statistics.py | ~250 | 2 | Medium | Hypothesis testing |
| main_simulation.py | ~650 | 1 large | High | Orchestration |
| simple_example.py | ~100 | 0 | Low | Tutorial |


### Documentation

| File | Content | Target |
|------|---------|--------|
| README.md | Overview, quick start | First read |
| SETUP_INSTRUCTIONS.md | Installation | Setup phase |
| CONSOLE_OUTPUT_GUIDE.md | Output interpretation | During runs |
| FUNCTION_REFERENCE.md | Function details | Deep dives |
| ORGANIZATION_GUIDE.md | Code structure | Understanding |
| ARCHITECTURE.md | System design | Advanced |
| FILE_ORGANIZATION_TREE.md | Visual structure | Reference |


---



## Output File Structure

### Individual Runs

```
results_phase_diagram_runs/
├── Subcritical_Rep0/
│   ├── basic_activity_plot.png         (6-panel overview)
│   ├── initial_5s_raster.png           (detailed raster)
│   └── detailed_stimulus_raster.png    (RC trials)
│
├── Critical_Rep0/
│   └── (same structure)
│
└── Supercritical_Rep0/
    └── (same structure)
```

### Summary Results

```
results_phase_diagram_summary/
├── simulation_summary.xlsx                         (all metrics)
├── phase_diagram_firing_rate.png                   (FR heatmap/line)
├── phase_diagram_overall_cv.png                    (CV heatmap/line)
├── phase_diagram_sigma.png                         (σ heatmap/line)
├── phase_diagram_rc_accuracy.png                   (accuracy heatmap/line)
├── comparative_learning_accuracy_curves.png        (learning curves)
├── neural_manifold_pca.png                         (PCA visualization)
└── aggregated_avalanche_ccdf_*.png                 (power-law plots)
```

---



## Common Modification Patterns

### To Add a New Parameter

1. Add to `config.py`:
```python
NEW_PARAM = 0.5
```

2. Use in relevant module:
```python
from config import NEW_PARAM
# Use NEW_PARAM in your code
```

3. Document in README.md

### To Add a New Analysis Function

1. Add to `analysis.py`:
```python
def calculate_new_metric(spike_monitor, ...):
    """Docstring"""
    # Implementation
    return result
```

2. Call from `main_simulation.py`:
```python
from analysis import calculate_new_metric
new_value = calculate_new_metric(SpikeMon_exc, ...)
```

3. Store result and plot

### To Add a New Plot

1. Add to `plotting.py`:
```python
def plot_new_visualization(data, ...):
    """Docstring"""
    fig, ax = plt.subplots()
    # Plotting code
    plt.savefig(...)
```

2. Call from `main_simulation.py`:
```python
from plotting import plot_new_visualization
plot_new_visualization(results, ...)
```



