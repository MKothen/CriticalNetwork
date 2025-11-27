# Network Characterization Summary

## Overview

This document provides a comprehensive characterization of the neural network architecture implemented in this simulation. It consolidates all structural, dynamical, and computational properties in one reference guide.

---

## Table of Contents

1. [Network Architecture Overview](#network-architecture-overview)
2. [Neuron Model Properties](#neuron-model-properties)
3. [Connectivity Structure](#connectivity-structure)
4. [Synaptic Properties](#synaptic-properties)
5. [Heterogeneity Specifications](#heterogeneity-specifications)
6. [Network Dynamics and Timescales](#network-dynamics-and-timescales)
7. [Expected Dynamic Ranges](#expected-dynamic-ranges)
8. [Biological Realism Assessment](#biological-realism-assessment)
9. [Computational Properties](#computational-properties)
10. [How to Verify Network Properties](#how-to-verify-network-properties)

---

## Network Architecture Overview

### Population Structure

| Property | Value | Biological Comparison |
|----------|-------|----------------------|
| **Total neurons** | 1000 (default) | 1 cortical column ≈ 10,000-100,000 neurons |
| **Excitatory neurons** | 800 (80%) | Cortex: ~80% pyramidal/excitatory |
| **Inhibitory neurons** | 200 (20%) | Cortex: ~20% interneurons |
| **E/I ratio** | 4:1 | Matches cortical proportions |

### Network Topology

| Property | Value | Notes |
|----------|-------|-------|
| **Topology type** | Random sparse (Erdős-Rényi) | Default; extensible to small-world, spatial |
| **Connection probability** | 0.1 (10%) | Each pair connects with p=0.1 |
| **Expected connections** | ~100,000 total | For N=1000, 10% of 1,000,000 possible |
| **Mean in-degree** | ~100 per neuron | p × N_total |
| **Mean out-degree** | ~100 per neuron | Same as in-degree for random networks |
| **Clustering coefficient** | ~0.1 | Low (≈ p); can increase with small-world topology |
| **Average path length** | ~2-3 hops | Short due to random connectivity |
| **Spatial structure** | None | Abstract network; no geometric embedding |

**Key insight:** The network is **non-spatial** and **uniformly random** by default, prioritizing computational exploration over anatomical realism. Can be extended to biologically realistic topologies (see NETWORK_EXTENSION_GUIDE.md).

---

## Neuron Model Properties

### Model Type: Adaptive Exponential Integrate-and-Fire (AdEx)

The AdEx model combines:
- **Exponential spike generation** (captures spike initiation dynamics)
- **Spike-triggered adaptation** (mimics calcium-activated potassium currents)
- **Linear leak current** (subthreshold integration)

**Mathematical formulation:**

```
dV/dt = [I_leak + I_exp - I_syn + I_noise - A + I_stim] / C_mem
dA/dt = [g_A(V - V_L) - A] / τ_A
```

Where:
- **I_leak** = -g_L(V - V_L): Leak current
- **I_exp** = g_L ΔT exp[(V - V_T)/ΔT]: Exponential spike term
- **I_syn** = I_syn_exc + I_syn_inh: Synaptic inputs
- **I_noise**: Ornstein-Uhlenbeck background noise
- **A**: Adaptation current
- **I_stim**: External stimulus (for RC tasks)

### Core Parameters

| Parameter | Symbol | Value | Biological Range | Notes |
|-----------|--------|-------|------------------|-------|
| **Membrane capacitance** | C_m | 200 pF | 100-300 pF | Typical for pyramidal neurons |
| **Leak conductance** | g_L | 12 nS (mean) | 10-30 nS | Controls input resistance |
| **Leak potential** | V_L | -70 mV | -70 to -60 mV | Resting potential |
| **Spike threshold** | V_T | -50 mV (mean) | -55 to -45 mV | Spike initiation |
| **Threshold slope** | ΔT | 2 mV | 1-5 mV | Sharpness of spike onset |
| **Reset potential** | V_r | -60 mV | -70 to -55 mV | Post-spike reset |
| **Adaptation conductance** | g_A | 4 nS | 2-20 nS | Strength of adaptation |
| **Adaptation time constant** | τ_A | 100-150 ms (mean) | 50-500 ms | Adaptation decay rate |
| **Adaptation step** | b | 0.1-0.5 nA | 0-2 nA | Per-spike adaptation increment |

### Timescales

| Process | Time Constant | Calculation | Functional Role |
|---------|--------------|-------------|-----------------|
| **Membrane integration** | τ_m ≈ 16.7 ms | C_m / g_L = 200 pF / 12 nS | Integration window for inputs |
| **Adaptation decay** | τ_A ≈ 100-150 ms | Heterogeneous | Post-burst hyperpolarization |
| **Noise correlation** | τ_noise = 20 ms | Fixed | Background fluctuation timescale |

### Spiking Dynamics

**Spike generation:**
1. Subthreshold integration: V increases linearly with input
2. Near threshold (-50 mV): Exponential term dominates
3. Spike escape: V → ∞ (numerically detected when V > V_thresh)
4. Reset: V → V_r (-60 mV), A → A + b
5. Adaptation: Increased A reduces excitability temporarily

**Key features:**
- **No absolute refractory period** (can spike immediately if driven strongly)
- **Relative refractory period** induced by adaptation current
- **Realistic spike shape** from exponential term (though spike waveform not explicitly modeled)

---

## Connectivity Structure

### Connection Types

The network implements **four synapse groups** following Dale's principle:

| Source → Target | Synapse Type | Connection Prob | Expected # Connections |
|-----------------|--------------|-----------------|----------------------|
| **E → E** | Excitatory | 0.1 | ~64,000 |
| **E → I** | Excitatory | 0.1 | ~16,000 |
| **I → E** | Inhibitory | 0.1 | ~16,000 |
| **I → I** | Inhibitory | 0.1 | ~4,000 |

**Total:** ~100,000 synapses for N=1000 network

### Connection Rules

**Random sparse (default):**
- Each pair (i, j) connects independently with probability p = 0.1
- No self-connections (neuron cannot connect to itself)
- Connections are **unidirectional** (synapse from i→j does not imply j→i)
- No spatial constraints

**Graph properties:**
- **Erdős-Rényi random graph**
- **Weakly connected** (typically): Most neurons reachable from most others
- **Small-world coefficient** σ < 1: Not a small-world network
- **Degree distribution**: Approximately binomial (not power-law)

### Connectivity Metrics (Typical Values for N=1000, p=0.1)

| Metric | E→E | E→I | I→E | I→I |
|--------|-----|-----|-----|-----|
| **Total synapses** | 64,000 | 16,000 | 16,000 | 4,000 |
| **Connection density** | 10% | 10% | 10% | 10% |
| **Mean in-degree** | 80 | 80 | 80 | 20 |
| **Mean out-degree** | 80 | 80 | 20 | 20 |
| **Clustering coefficient** | ~0.1 | N/A | N/A | ~0.1 |
| **Path length** | ~2-3 | N/A | N/A | ~2-3 |

---

## Synaptic Properties

### Synaptic Dynamics

**Model:** Double-exponential conductance-based synapses

**Differential equations:**
```
ds_syn/dt = x_syn
dx_syn/dt = [-(τ_r + τ_d) x_syn - s_syn] / (τ_r × τ_d)

On pre-synaptic spike: x_syn += 1 Hz
Post-synaptic current: I_syn = g_syn_max × s_syn × (V_post - V_syn)
```

### Time Constants

| Parameter | Value | Biological Range | Notes |
|-----------|-------|------------------|-------|
| **τ_rise** | 5 ms | 1-10 ms | Synaptic rise time (both E and I) |
| **τ_decay** | 50 ms | 10-100 ms | Synaptic decay time (both E and I) |
| **Peak time** | ~15 ms | — | Time to reach maximum conductance |

**Response profile:**
- Spike at t=0 → s_syn rises to peak ~15 ms → decays with τ=50 ms
- Approximates **AMPA** (excitatory) and **GABA_A** (inhibitory) kinetics

### Synaptic Weights

| Connection | Base Conductance | Scaling Factor | Effective Conductance |
|------------|-----------------|----------------|----------------------|
| **E → E** | g_exc_base | exc_factor (typically 1.0) | ~nS range |
| **E → I** | g_exc_base | exc_factor (typically 1.0) | Same as E→E |
| **I → E** | g_inh_base | inh_factor (typically 5.0) | ~5× stronger than E |
| **I → I** | g_inh_base | inh_factor (typically 5.0) | Same as I→E |

**Key insight:** Inhibitory synapses are ~5× stronger than excitatory to maintain E/I balance despite 4:1 population ratio.

### Reversal Potentials

| Synapse Type | Reversal Potential | Effect |
|--------------|-------------------|--------|
| **Excitatory (AMPA)** | V_syn_exc = 0 mV | Depolarizing (excitatory) |
| **Inhibitory (GABA)** | V_syn_inh = -80 mV | Hyperpolarizing (inhibitory) |

**Driving force:**
- For excitation: (V - 0 mV) → Maximum at rest (-70 mV), decreases at depolarization
- For inhibition: (V - (-80 mV)) → Shunting near rest, hyperpolarizing when depolarized

### Synaptic Delays

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Axonal delay** | 0 ms (instantaneous) | **Not implemented in default model** |
| **Synaptic transmission** | Instantaneous | Spike at t causes g_syn jump immediately |

**Limitation:** Current implementation has **zero synaptic delays**. This is a simplification that:
- Speeds up simulation
- Reduces biological realism
- Can be extended by adding `delay=X*ms` to synapse connections

**To add delays:**
```python
Syn_exc_to_exc.connect(p=connection_prob)
Syn_exc_to_exc.delay = '1*ms + randn()*0.5*ms'  # Mean 1ms, std 0.5ms
```

---

## Heterogeneity Specifications

Neurons have **heterogeneous parameters** drawn from Gaussian distributions to increase biological realism and prevent artificial synchronization.

### Heterogeneous Parameters

| Parameter | Mean | Std Dev | Distribution | Min Clipping |
|-----------|------|---------|--------------|--------------|
| **Spike threshold** V_T | -50 mV | 2 mV | Gaussian | None |
| **Leak conductance** g_L | 12 nS | 2 nS | Gaussian | 0.1 nS |
| **Adaptation step** b | 0.3 nA | 0.1 nA | Gaussian | 0 nA |
| **Adaptation τ** τ_A | 125 ms | 25 ms | Gaussian | 50 ms |

**Effect of heterogeneity:**
- **Threshold variability** → Different neurons spike at different input levels
- **Conductance variability** → Different time constants and input resistances
- **Adaptation variability** → Some neurons adapt strongly, others weakly

### Fixed (Homogeneous) Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| **Capacitance** C_m | 200 pF | Keeps τ_m distribution manageable |
| **Leak potential** V_L | -70 mV | Reference point for all neurons |
| **Threshold slope** ΔT | 2 mV | Controls spike sharpness uniformly |
| **Reset potential** V_r | -60 mV | Consistent post-spike state |

### Background Input

| Parameter | Mean | Std Dev | Temporal Dynamics |
|-----------|------|---------|-------------------|
| **I_mid** | 0.333 nA | — | Constant per neuron |
| **I_noise** | I_mid | σ_noise = 0.1 nA | Ornstein-Uhlenbeck process |
| **τ_noise** | 20 ms | — | Noise correlation time |

**Noise properties:**
- **Colored noise** (not white): Correlations over ~20 ms
- **Additive**: Independent of voltage
- **Balanced**: Mean equals I_mid, fluctuations ± σ_noise

---

## Network Dynamics and Timescales

### Operating Timescales

| Process | Timescale | Range |
|---------|-----------|-------|
| **Spike width** | ~1 ms | Not explicitly modeled (AdEx abstracts spike) |
| **Membrane integration** | 16.7 ms | C_m / g_L |
| **Synaptic rise** | 5 ms | Fast AMPA/GABA onset |
| **Synaptic decay** | 50 ms | Slow integration window |
| **Adaptation buildup** | 100-150 ms | Post-burst hyperpolarization |
| **Noise correlation** | 20 ms | Background fluctuation timescale |

**Hierarchy:**
```
Fast: Spikes (1 ms) < Synaptic rise (5 ms) < Membrane τ (17 ms)
Medium: Noise τ (20 ms) < Synaptic decay (50 ms)
Slow: Adaptation (100-150 ms)
```

### Dynamic Regimes

The network can operate in different regimes depending on E/I balance:

#### Subcritical Regime (E/I ratio << 1)
- **Inhibition-dominated**
- Firing rate: 0.5-2 Hz (very quiet)
- Activity dies out quickly
- Branching parameter σ < 1
- Few or no avalanches

#### Critical Regime (E/I ratio ≈ 0.3-0.5)
- **Balanced E/I**
- Firing rate: 3-8 Hz (biologically realistic)
- Sustained, irregular activity
- Branching parameter σ ≈ 1
- Power-law avalanche distributions
- High CV (>1, irregular spiking)

#### Supercritical Regime (E/I ratio ≥ 1)
- **Excitation-dominated**
- Firing rate: >10 Hz (hyperactive)
- Activity grows and spreads
- Branching parameter σ > 1
- Large avalanches or runaway excitation

---

## Expected Dynamic Ranges

### Firing Rates

| Regime | Excitatory FR | Inhibitory FR | Total FR |
|--------|---------------|---------------|----------|
| **Subcritical** | 0.5-2 Hz | 1-3 Hz | 0.8-2.2 Hz |
| **Critical** | 3-8 Hz | 8-15 Hz | 4-10 Hz |
| **Supercritical** | >10 Hz | >20 Hz | >12 Hz |
| **Pathological** | >50 Hz | >100 Hz | Seizure-like |

**Biological comparison:**
- Cortex at rest: 1-5 Hz (sparse, irregular)
- Cortex during task: 10-30 Hz
- This network (critical): 3-10 Hz ✓ Matches resting cortex

### Spike Irregularity (CV)

| Regime | Mean CV | Interpretation |
|--------|---------|----------------|
| **Too regular** | CV < 0.5 | Unnatural, clock-like |
| **Biological range** | **0.8-1.5** | **Irregular, Poisson-like** ✓ |
| **Too irregular** | CV > 2.0 | Extremely bursty |

**This network (critical):** CV ≈ 1.0-1.4 ✓ Biologically realistic

### Membrane Potentials

| State | Voltage Range | Notes |
|-------|--------------|-------|
| **Resting** | -70 mV | Near V_L |
| **Subthreshold** | -65 to -50 mV | Fluctuating with input |
| **Spike threshold** | -50 mV (mean) | Varies by neuron (-52 to -48 mV) |
| **Spike peak** | ~+20 mV | Not modeled (AdEx abstracts) |
| **Reset** | -60 mV | Post-spike |
| **Adapted state** | -75 mV | After burst (hyperpolarized) |

### Avalanche Statistics (Critical Regime)

| Metric | Expected Value | Notes |
|--------|----------------|-------|
| **Branching parameter** σ | 0.95-1.05 | Close to 1 = critical |
| **Size exponent** α | 1.3-1.7 | Power-law exponent for avalanche sizes |
| **Duration exponent** τ | 1.5-2.5 | Power-law exponent for durations |
| **Scaling exponent** γ | 1.2-1.5 | Relates size to duration |
| **Avalanche count** | 500-2000 per 20s | Depends on bin width |

---

## Biological Realism Assessment

### Strengths (High Realism)

✅ **Neuron model:**
- AdEx captures spike initiation and adaptation
- Parameters within biological ranges
- Heterogeneity mimics real populations

✅ **Population structure:**
- 80/20 E/I split matches cortex
- Separate excitatory and inhibitory types

✅ **Synaptic dynamics:**
- Double-exponential kinetics realistic
- AMPA/GABA-like time constants

✅ **Background noise:**
- Ornstein-Uhlenbeck process (colored, not white)
- Realistic correlation timescale

### Limitations (Low Realism)

⚠️ **Network topology:**
- Random connectivity (not small-world or spatially organized)
- No cortical layers or columns
- No distance-dependent connectivity

⚠️ **Synaptic delays:**
- Zero delay (instantaneous transmission)
- Real cortex: 1-5 ms delays

⚠️ **Neuron types:**
- Only one excitatory type (no RS, IB, CH distinctions)
- Only one inhibitory type (no PV, SST, VIP distinctions)

⚠️ **Synaptic plasticity:**
- Fixed weights (no STDP or homeostasis)
- No short-term facilitation/depression

⚠️ **Spatial structure:**
- No geometry (non-spatial network)
- No distance-dependent properties


---

## Quick Reference Tables

### Parameter Quick Lookup

| Category | Parameter | Symbol | Value |
|----------|-----------|--------|-------|
| **Structure** | Total neurons | N | 1000 |
| | Excitatory | N_exc | 800 |
| | Inhibitory | N_inh | 200 |
| | Connection prob | p | 0.1 |
| **Membrane** | Capacitance | C_m | 200 pF |
| | Leak conductance | g_L | 12 nS |
| | Membrane τ | τ_m | 16.7 ms |
| **Threshold** | Mean threshold | V_T | -50 mV |
| | Threshold std | σ_VT | 2 mV |
| **Adaptation** | Conductance | g_A | 4 nS |
| | Time constant | τ_A | 100-150 ms |
| | Step size | b | 0.3 nA |
| **Synaptic** | Rise time | τ_r | 5 ms |
| | Decay time | τ_d | 50 ms |
| | E reversal | V_exc | 0 mV |
| | I reversal | V_inh | -80 mV |



## Summary

This network implements a **balanced excitatory-inhibitory spiking neural network** with:

✅ **Biologically realistic neuron dynamics** (AdEx model)  
✅ **Cortex-inspired population structure** (80/20 E/I)  
✅ **Heterogeneous parameters** (theoretically prevents synchronization, but doesnt?)  
✅ **Realistic synaptic kinetics** (double-exponential)  
✅ **Critical dynamics capability** (tunable via E/I balance)  
⚠️ **Abstract topology** (random, not spatial/structured)  
⚠️ **Zero delays** (can be extended)  
⚠️ **Fixed weights** (no plasticity)  


