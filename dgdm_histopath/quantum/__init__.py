"""
Quantum Enhancement Module for DGDM Histopath Lab.

This module provides quantum-enhanced processing capabilities for large-scale
histopathology analysis, including quantum planning, scheduling, and optimization.

Components:
- QuantumPlanner: Quantum-enhanced task planning and resource allocation
- QuantumScheduler: Advanced scheduling with quantum optimization  
- QuantumOptimizer: Quantum algorithms for model optimization
- QuantumSafety: Safety protocols for quantum operations
- QuantumDistributed: Distributed quantum processing capabilities
"""

__version__ = "0.1.0"

# Import quantum components with graceful degradation
try:
    from dgdm_histopath.quantum.quantum_planner import QuantumTaskPlanner
    from dgdm_histopath.quantum.quantum_scheduler import QuantumScheduler
    from dgdm_histopath.quantum.quantum_optimizer import QuantumOptimizer
    from dgdm_histopath.quantum.quantum_safety import QuantumSafety
    from dgdm_histopath.quantum.quantum_distributed import QuantumDistributedProcessor
    
    QUANTUM_COMPONENTS_AVAILABLE = True
    
except ImportError as e:
    import warnings
    warnings.warn(f"Quantum components not fully available: {e}")
    QUANTUM_COMPONENTS_AVAILABLE = False

# Feature flags
FEATURES = {
    "quantum_planning": QUANTUM_COMPONENTS_AVAILABLE,
    "quantum_scheduling": QUANTUM_COMPONENTS_AVAILABLE,
    "quantum_optimization": QUANTUM_COMPONENTS_AVAILABLE,
    "quantum_safety": QUANTUM_COMPONENTS_AVAILABLE,
    "quantum_distributed": QUANTUM_COMPONENTS_AVAILABLE
}

def check_quantum_availability():
    """Check which quantum components are available."""
    return FEATURES.copy()

def get_quantum_info():
    """Get quantum module information."""
    return {
        "version": __version__,
        "components_available": QUANTUM_COMPONENTS_AVAILABLE,
        "features": FEATURES
    }

__all__ = [
    "QuantumTaskPlanner",
    "QuantumScheduler", 
    "QuantumOptimizer",
    "QuantumSafety",
    "QuantumDistributedProcessor",
    "check_quantum_availability",
    "get_quantum_info",
    "QUANTUM_COMPONENTS_AVAILABLE",
    "FEATURES"
]