# Setup and Installation Instructions

## Prerequisites

- Python 3.7 or higher
- pip package manager
- At least 4 GB RAM (8 GB recommended)
- ~500 MB free disk space (for dependencies and data)

## Installation Steps

### 1. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv neural_sim_env

# Activate it (on Linux/Mac)
source neural_sim_env/bin/activate

# Activate it (on Windows)
neural_sim_env\Scripts\activate
```

### 2. Install Dependencies

```bash
# Core simulation
pip install brian2

# Scientific computing
pip install numpy scipy matplotlib pandas

# Machine learning
pip install scikit-learn

# Statistical analysis
pip install powerlaw statsmodels

# Optional: Jupyter for interactive exploration
pip install jupyter
```

Or install all at once:

```bash
pip install brian2 numpy scipy matplotlib pandas scikit-learn powerlaw statsmodels
```

### 3. Verify Installation

```python
# Test Brian2
python -c "import brian2; print('Brian2 version:', brian2.__version__)"

# Test other packages
python -c "import numpy, scipy, matplotlib, sklearn, powerlaw, statsmodels; print('All packages OK!')"
```

## Quick Test Run

### Option 1: Run the Simple Example

```bash
python simple_example.py
```

This will:
- Create a small network (1000 neurons)
- Run for ~20 seconds of simulation time
- Generate basic plots
- Save output as `simple_example_output.png`

**Expected runtime**: 2-5 minutes

### Option 2: Run a Minimal Main Simulation

First, edit `config.py` to reduce computation:

```python
# In config.py, change these lines:
N_TOTAL_NEURONS = 500              # Reduced from 1000
SIM_RUNTIME = 10 * second          # Reduced from 20 seconds
NUM_REPETITIONS = 2                # Reduced from 8
NUM_TRAIN_SAMPLES_MAX = 100        # Reduced from 200
NUM_TEST_SAMPLES_MAX = 50          # Reduced from 100
```

Then run:

```bash
python main_simulation.py
```

**Expected runtime**: 20-40 minutes (for reduced parameters)

## Troubleshooting

### Issue: Brian2 Installation Fails

**Solution**:
```bash
# Try installing with conda instead
conda install -c conda-forge brian2
```

### Issue: "No module named 'config'"

**Solution**: Make sure you're running Python from the directory containing all the .py files:
```bash
cd /path/to/your/code/directory
python simple_example.py
```

### Issue: Out of Memory Error

**Solution**: Reduce network size in `config.py`:
```python
N_TOTAL_NEURONS = 250  # Much smaller
SIM_RUNTIME = 5 * second
```

### Issue: MNIST Download Fails

**Solution**: 
1. Check internet connection
2. Try downloading manually from: https://www.openml.org/d/554
3. Or use a smaller dataset for testing

### Issue: Simulation is Too Slow

**Solution**:
1. Use a smaller network (N_TOTAL_NEURONS = 200-500)
2. Reduce simulation time (SIM_RUNTIME = 5-10 seconds)
3. Reduce number of repetitions (NUM_REPETITIONS = 1-2)
4. Skip RC analysis temporarily by commenting out RC code in main script

## File Structure Check

After setup, your directory should look like:

```
your_project/
│
├── config.py                    # ✓ Created
├── data_utils.py                # ✓ Created
├── network_model.py             # ✓ Created
├── analysis.py                  # ✓ Created
├── reservoir.py                 # ✓ Created
├── plotting.py                  # ⚠️ To be created
├── statistics.py                # ⚠️ To be created
├── main_simulation.py           # To be created
│
├── simple_example.py            # ✓ Created
├── README.md                    # ✓ Created
├── ORGANIZATION_GUIDE.md        # ✓ Created
└── SETUP_INSTRUCTIONS.md        # This file
```

## Expected Output Files

After running simulations, these directories will be created:

```
results_phase_diagram_runs/      # Individual run results
│   ├── Subcritical_Rep0/
│   ├── Critical_Rep0/
│   └── Supercritical_Rep0/
│
results_phase_diagram_summary/   # Aggregated results
    ├── simulation_summary.xlsx
    ├── phase_diagram_*.png
    └── comparative_learning_accuracy_curves.png
```

## Performance Benchmarks

Approximate runtimes on a modern laptop (Intel i7, 16 GB RAM):

| Configuration | Runtime | Output Size |
|--------------|---------|-------------|
| Simple example | 2-5 min | ~5 MB |
| Minimal main (N=500, 2 reps) | 20-40 min | ~50 MB |
| Full simulation (N=1000, 8 reps) | 3-6 hours | ~500 MB |

## Next Steps After Setup

1. ✅ Verify installation
2. ✅ Run simple_example.py
3. 📖 Read README.md thoroughly
4. 🔧 Modify config.py parameters
5. 🧪 Run small test simulations
6. 📊 Analyze output plots
7. 🚀 Run full parameter sweep

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Verify all dependencies are installed
3. Try the simple example first
4. Reduce parameters if running out of memory
5. Check the README.md for common issues
6. Review docstrings in the code (`help(function_name)`)

## Tips

- **Start small**: Use N=200-500 neurons for initial testing
- **Save often**: Simulations can take hours; don't lose your work
- **Document everything**: Keep notes on which parameters you tried
- **Use version control**: Consider using git to track changes
- **Check outputs frequently**: Look at plots after each run to catch issues early

---

**Ready to start? Run this command:**

```bash
python simple_example.py
```

If it completes without errors, you're all set! 🎉
