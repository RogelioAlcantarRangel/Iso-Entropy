# src/core/__init__.py
# Módulo central de ISO-ENTROPÍA v2.3

"""
ISO-ENTROPÍA Core Module

Componentes:
- agent: Auditor autónomo con FSM
- physics: Simulación Monte Carlo
- fsm: Máquina de estados finitos
- constraints: Validaciones duras
- grounding: Mapeo UI → Física
- telemetry: Señales de telemetría
- prompt_templates: Prompts inteligentes por fase
"""

# ============================================================================
# IMPORTS DEL AGENTE PRINCIPAL
# ============================================================================

from .agent import IsoEntropyAgent

# ============================================================================
# IMPORTS DE FÍSICA Y SIMULACIÓN
# ============================================================================

from .physics import (
    run_simulation,
    calculate_collapse_threshold
)

# ============================================================================
# IMPORTS DE FSM (Máquina de Estados)
# ============================================================================

from .fsm import (
    IsoEntropyFSM,
    AgentPhase
)

# ============================================================================
# IMPORTS DE CONSTRAINTS (Validaciones)
# ============================================================================

from .constraints import (
    apply_hard_rules,
    HardConstraintViolation
)

# ============================================================================
# IMPORTS DE GROUNDING (Mapeo de inputs)
# ============================================================================

from .grounding import ground_inputs

# ============================================================================
# IMPORTS DE TELEMETRÍA
# ============================================================================

from .telemetry import build_llm_signal

# ============================================================================
# IMPORTS DE PROMPTS
# ============================================================================

from .prompt_templates import build_prompt_for_phase

# ============================================================================
# EXPORTS PÚBLICOS
# ============================================================================

__all__ = [
    # Agent
    "IsoEntropyAgent",
    
    # Physics & Simulation
    "run_simulation",
    "calculate_collapse_threshold",
    
    # FSM
    "IsoEntropyFSM",
    "AgentPhase",
    
    # Constraints
    "apply_hard_rules",
    "HardConstraintViolation",
    
    # Grounding
    "ground_inputs",
    
    # Telemetry
    "build_llm_signal",
    
    # Prompts
    "build_prompt_for_phase"
]

# ============================================================================
# VERSIÓN Y METADATA
# ============================================================================

__version__ = "2.3.0"
__author__ = "Rogelio Alcántar Rangel"
__description__ = "ISO-ENTROPÍA: Auditor de Fragilidad Estructural"