# Network Extension Guide

## Overview

This guide explains the general principles for extending the network architecture with new connectivity patterns or features. Rather than providing exhaustive implementations, it focuses on the approach and structure for adding your own modifications.

---

## Understanding the Extension Points

The codebase is designed with clear extension points where you can add new functionality:

### 1. Configuration Layer (`config.py`)
Add new parameters here that control network behavior.

### 2. Network Model Layer (`network_model.py`)
Modify neuron equations, connectivity patterns, or add new network components.

### 3. Analysis Layer (`analysis.py`)
Add new metrics or analysis functions for network characterization.

---

## General Approach to Adding New Connectivity Patterns

### Current Implementation

The default network uses **random sparse connectivity**:
```python
# In create_network() function
Syn_exc_to_exc.connect(p=connection_prob)  # Random with probability p
```

**Properties:**
- Each neuron pair connects independently with probability `p`
- No structure or spatial organization
- Connection probability: 10% (default)
- Expected ~100 connections per neuron

### Extending to New Topologies

To add a new connectivity pattern (e.g., small-world, spatial, modular):

**Step 1: Define parameters in `config.py`**
```python
# Add topology-specific parameters
NETWORK_TOPOLOGY = 'your_topology_type'
TOPOLOGY_PARAM_1 = value1
TOPOLOGY_PARAM_2 = value2
```

**Step 2: Create connectivity function in `network_model.py`**
```python
def create_your_connectivity(N_source, N_target, param1, param2):
    """
    Generate connection indices for your topology.
    
    Returns
    -------
    tuple
        (i_indices, j_indices) - Arrays of source and target indices
    """
    # Your algorithm here
    # Create connection pairs based on your rules
    
    return i_indices, j_indices
```

**Step 3: Modify `create_network()` to use new topology**
```python
def create_network(..., topology='random', **topology_params):
    # ... (create neurons and synapses as before)
    
    if topology == 'random':
        Syn_exc_to_exc.connect(p=connection_prob)
    elif topology == 'your_topology':
        i, j = create_your_connectivity(N_exc, N_exc, **topology_params)
        Syn_exc_to_exc.connect(i=i, j=j)
    # Repeat for other synapse types (E→I, I→E, I→I)
```

**Step 4: Update `main_simulation.py`**
```python
# Use new topology
network_dict = create_network(
    N_exc, N_inh,
    current_Imid=Imid_val,
    exc_factor=exc_strength,
    inh_factor=inh_strength,
    topology='your_topology',
    param1=TOPOLOGY_PARAM_1,
    param2=TOPOLOGY_PARAM_2
)
```

---

## Common Topology Types and Principles

### Small-World Networks
**Key idea:** Combine local clustering with long-range shortcuts
```python
# Pseudocode
for each neuron:
    connect to k nearest neighbors (ring lattice)
    
for each connection:
    with probability p_rewire:
        rewire to random target
```

**When to use:** Biological realism, studying local vs global processing

### Spatial Networks
**Key idea:** Connection probability decreases with distance
```python
# Pseudocode
compute distances between all neuron pairs
connection_prob = function(distance)  # e.g., exp(-d/lambda)
sample connections based on probabilities
```

**When to use:** Cortical-inspired structure, studying spatial effects

### Modular Networks
**Key idea:** Strong within-module, weak between-module connectivity
```python
# Pseudocode
assign neurons to modules
for each neuron pair:
    if same_module:
        p_connect = high_probability
    else:
        p_connect = low_probability
```

**When to use:** Studying functional segregation, columnar organization

---

## Adding Network Analysis Functions

### General Pattern for Analysis Functions

Add to `analysis.py`:

```python
def analyze_your_metric(spike_monitor, network_params):
    """
    Calculate your custom metric.
    
    Parameters
    ----------
    spike_monitor : SpikeMonitor
        Brian2 spike monitor
    network_params : dict
        Relevant network parameters
    
    Returns
    -------
    float or dict
        Your computed metric(s)
    """
    # Extract spike data
    spike_times = spike_monitor.t[:]
    spike_neurons = spike_monitor.i[:]
    
    # Your analysis algorithm
    result = your_calculation(spike_times, spike_neurons)
    
    return result
```

### Analyzing Connectivity Structure

For graph-theoretic measures, you'll typically:
1. Extract connection indices from synapse objects
2. Build a graph representation (e.g., with NetworkX)
3. Compute metrics (clustering, path length, modularity, etc.)

**Example structure:**
```python
def analyze_topology_metric(synapse_object, N_source, N_target):
    """Compute graph-theoretic metric."""
    # Get connections
    i_sources = synapse_object.i[:]
    j_targets = synapse_object.j[:]
    
    # Build graph (if needed)
    # import networkx as nx
    # G = nx.DiGraph()
    # G.add_edges_from(zip(i_sources, j_targets))
    
    # Compute your metric
    metric = your_computation(i_sources, j_targets)
    
    return metric
```

---

## Best Practices

### 1. Maintain Backward Compatibility
Always provide defaults so existing code still works:
```python
def create_network(..., topology='random', connection_prob=0.1, **kwargs):
    # Default behavior unchanged
```

### 2. Document Expected Properties
For each topology, document:
- Connection density
- Degree distribution
- Clustering properties
- Biological motivation

### 3. Validate Parameters
```python
# In config.py or at function start
assert 0 <= rewiring_prob <= 1, "Rewiring probability must be in [0, 1]"
assert k_neighbors < N_neurons, "k_neighbors must be less than network size"
```

### 4. Test Incrementally
- First test connectivity generation alone
- Then integrate with network creation
- Verify properties match expectations
- Compare dynamics with baseline (random topology)

### 5. Profile Performance
Some topologies are computationally expensive:
- Small-world: O(N×k), fast
- Spatial: O(N²), can be slow for large N
- Graph analysis (path length): O(N²), very slow for N > 5,000

---

## Typical Workflow

### When Adding a New Feature

1. **Identify extension point** (config, network_model, analysis)
2. **Add parameters** to config.py
3. **Implement core function** in appropriate module
4. **Integrate** with existing create_network() or analysis pipeline
5. **Test** with small network first
6. **Verify** properties match expectations
7. **Document** new functionality

### When Analyzing Network Properties

1. **Run simulation** to generate spike data
2. **Extract relevant data** (spike times, connection matrices)
3. **Compute metrics** using analysis functions
4. **Compare** to expected ranges or literature values
5. **Visualize** results for interpretation

---

## Common Modifications

### Adding Synaptic Delays
```python
# In create_network(), after creating synapses:
Syn_exc_to_exc.delay = '1*ms + randn()*0.5*ms'  # Mean 1ms, std 0.5ms
```

### Heterogeneous Synaptic Weights
```python
# Instead of uniform weights:
weights = base_weight * (1 + weight_std * np.random.randn(len(synapse.i)))
synapse.peak_conductance = np.clip(weights, min_weight, max_weight)
```

### Multiple Neuron Types
```python
# Create separate populations with different parameters
Pop_exc_RS = NeuronGroup(N_RS, eqs, ...)  # Regular spiking
Pop_exc_IB = NeuronGroup(N_IB, eqs, ...)  # Intrinsic bursting
# Initialize with different parameter distributions
```

---

## Verification Checklist

After adding a new feature:

- [ ] Code runs without errors
- [ ] Network properties match expectations
- [ ] Results are reproducible (seed control)
- [ ] Documentation updated
- [ ] Backward compatibility maintained
- [ ] Performance acceptable

---

## Resources and Tools

### Required Dependencies
- **Brian2**: Simulation engine
- **NumPy**: Numerical operations
- **NetworkX** (optional): Graph analysis

### Useful Patterns
- **Brian2 connection syntax**: `synapse.connect(i=sources, j=targets)`
- **NetworkX graphs**: For topology analysis
- **SciPy spatial**: For distance-dependent connectivity

### Example Topology Implementations
For detailed implementations of specific topologies (Watts-Strogatz, Barabási-Albert, etc.), refer to standard network science libraries or papers.

---

## Summary

To extend the network:

1. **Define what you want to add** (topology, analysis metric, neuron type)
2. **Choose the right module** (config, network_model, analysis)
3. **Implement the core function** following existing patterns
4. **Integrate** with create_network() or analysis pipeline
5. **Test and verify** properties
6. **Document** your addition

The codebase structure makes extensions straightforward - each module has a clear purpose and standard patterns to follow. Start simple, test incrementally, and gradually add complexity.

**Key principle:** The existing random topology provides a working baseline. Any extension should maintain compatibility with this baseline while adding new functionality through optional parameters.
