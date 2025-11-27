"""
Neural network model definitions
Contains Brian2 equations and network setup functions
"""

from brian2 import *
import numpy as np
from config import *


def get_neuron_equations():
    """
    Returns the AdEx (Adaptive Exponential Integrate-and-Fire) neuron equations.

    Returns
    -------
    tuple
        (equations_string, reset_string)
    """
    eqs_EIF = '''
    dV/dt = 1/C_mem * (I_leak + I_exp - (I_syn_exc + I_syn_inh) + I_noise - A + I_stim) : volt (unless refractory)
    dA/dt = 1/tau_A*(g_A*(V-V_L)-A) : ampere
    dI_noise/dt = -(I_noise-Imid_val_eq)/tau_noise + sigma_noise * sqrt(2/tau_noise) * xi : ampere
    I_stim : ampere
    Imid_val_eq : ampere
    s_tot_exc : siemens
    s_tot_inh : siemens
    V_thresh_val_eq : volt (constant)
    I_syn_exc = s_tot_exc * (V - V_syn_exc) : ampere
    I_syn_inh = s_tot_inh * (V - V_syn_inh) : ampere
    I_leak = -g_mem_val_eq*(V-V_L) : ampere
    I_exp = g_mem_val_eq*D_T*exp((V-V_T_val_eq)/D_T) : ampere
    g_mem_val_eq : siemens (constant)
    b_val_eq : ampere (constant)
    tau_A : second (constant)
    V_T_val_eq : volt (constant)
    '''

    eqs_reset = 'V = Vr; A += b_val_eq'

    return eqs_EIF, eqs_reset


def get_synapse_equations():
    """
    Returns the synaptic dynamics equations.

    Returns
    -------
    tuple
        (exc_model, inh_model, exc_onpre, inh_onpre)
    """
    eqs_syn_dynamics = '''
    ds_syn/dt = x_syn : 1 (clock-driven)
    dx_syn/dt = 1/(tau_r_syn*tau_d_syn) *( -(tau_r_syn+tau_d_syn)*x_syn - s_syn ) : Hz (clock-driven)
    peak_conductance : siemens
    '''

    eqs_syn_exc_output = 's_tot_exc_post = peak_conductance * s_syn_exc : siemens (summed)'
    eqs_syn_inh_output = 's_tot_inh_post = peak_conductance * s_syn_inh : siemens (summed)'

    model_string_exc = (
        eqs_syn_dynamics.replace('s_syn', 's_syn_exc').replace('x_syn', 'x_syn_exc') 
        + eqs_syn_exc_output
    )
    model_string_inh = (
        eqs_syn_dynamics.replace('s_syn', 's_syn_inh').replace('x_syn', 'x_syn_inh') 
        + eqs_syn_inh_output
    )

    eqs_onpre_exc = 'x_syn_exc += 1*Hz'
    eqs_onpre_inh = 'x_syn_inh += 1*Hz'

    return model_string_exc, model_string_inh, eqs_onpre_exc, eqs_onpre_inh


def initialize_neuron_population(pop, current_Imid):
    """
    Initialize a neuron population with heterogeneous parameters.

    Parameters
    ----------
    pop : NeuronGroup
        The Brian2 neuron group to initialize
    current_Imid : Quantity
        The mid-point current value (with Brian2 units)
    """
    N_pop_type = len(pop)

    # Set heterogeneous parameters
    pop.V_thresh_val_eq = Vt_mean + np.random.randn(N_pop_type) * Vt_std
    pop.g_mem_val_eq = np.maximum(0.1*nS, g_mem_mean + np.random.randn(N_pop_type) * g_mem_std)
    pop.b_val_eq = np.maximum(0*nA, b_mean + np.random.randn(N_pop_type) * b_std)
    pop.tau_A = np.maximum(50*ms, tau_A_mean + np.random.randn(N_pop_type) * tau_A_std)

    # Set fixed parameters
    pop.V_T_val_eq = V_T_val
    pop.A = 0.0 * nA
    pop.Imid_val_eq = current_Imid
    pop.I_noise = current_Imid
    pop.I_stim = 0 * nA
    pop.V = V_L + np.random.randn(N_pop_type) * 5 * mV
    pop.s_tot_exc = 0 * siemens
    pop.s_tot_inh = 0 * siemens


def create_network(N_exc, N_inh, current_Imid, exc_factor, inh_factor, connection_prob=0.1):
    """
    Create a balanced excitatory-inhibitory network.

    Parameters
    ----------
    N_exc : int
        Number of excitatory neurons
    N_inh : int
        Number of inhibitory neurons
    current_Imid : Quantity
        Background input current
    exc_factor : float
        Scaling factor for excitatory synapses
    inh_factor : float
        Scaling factor for inhibitory synapses
    connection_prob : float
        Connection probability

    Returns
    -------
    dict
        Dictionary containing all network components
    """
    # Get model equations
    eqs_neurons, eqs_reset = get_neuron_equations()
    model_exc, model_inh, onpre_exc, onpre_inh = get_synapse_equations()

    # Create neuron populations
    Pop_exc = NeuronGroup(
        N_exc, eqs_neurons, 
        threshold='V>V_thresh_val_eq', 
        reset=eqs_reset,
        method='euler', 
        dt=set_dt
    )

    Pop_inh = NeuronGroup(
        N_inh, eqs_neurons,
        threshold='V>V_thresh_val_eq',
        reset=eqs_reset,
        method='euler',
        dt=set_dt
    )

    # Initialize populations
    for pop in [Pop_exc, Pop_inh]:
        initialize_neuron_population(pop, current_Imid)

    # Create synapses
    Syn_exc_to_exc = Synapses(Pop_exc, Pop_exc, model=model_exc, on_pre=onpre_exc, dt=set_dt)
    Syn_exc_to_inh = Synapses(Pop_exc, Pop_inh, model=model_exc, on_pre=onpre_exc, dt=set_dt)
    Syn_inh_to_exc = Synapses(Pop_inh, Pop_exc, model=model_inh, on_pre=onpre_inh, dt=set_dt)
    Syn_inh_to_inh = Synapses(Pop_inh, Pop_inh, model=model_inh, on_pre=onpre_inh, dt=set_dt)

    # Connect with fixed probability
    Syn_exc_to_exc.connect(p=connection_prob)
    Syn_exc_to_inh.connect(p=connection_prob)
    Syn_inh_to_exc.connect(p=connection_prob)
    Syn_inh_to_inh.connect(p=connection_prob)

    # Set synaptic weights
    Syn_exc_to_exc.peak_conductance = exc_factor * base_g_syn_max_exc_value * siemens
    Syn_exc_to_inh.peak_conductance = exc_factor * base_g_syn_max_exc_value * siemens
    Syn_inh_to_exc.peak_conductance = inh_factor * base_g_syn_max_inh_value * siemens
    Syn_inh_to_inh.peak_conductance = inh_factor * base_g_syn_max_inh_value * siemens

    return {
        'Pop_exc': Pop_exc,
        'Pop_inh': Pop_inh,
        'Syn_exc_to_exc': Syn_exc_to_exc,
        'Syn_exc_to_inh': Syn_exc_to_inh,
        'Syn_inh_to_exc': Syn_inh_to_exc,
        'Syn_inh_to_inh': Syn_inh_to_inh,
        'synapses': [Syn_exc_to_exc, Syn_exc_to_inh, Syn_inh_to_exc, Syn_inh_to_inh]
    }