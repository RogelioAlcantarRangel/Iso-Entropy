"""
Core: Motor científico de ISO-ENTROPÍA

Módulos principales:
- agent: Orquestador autónomo con _build_search_context()
- physics: Simulación Monte Carlo (500 runs, ±2% precisión)
- fsm: Máquina de estados finitos (ORIENT→VALIDATE→STRESS→CONCLUDE)
- constraints: Pre-control y validaciones duras
- grounding: Mapeo UI → Física
- telemetry: Señales y enriquecimiento LLM
- prompt_templates: Prompts inteligentes por fase
"""

from .agent import IsoEntropyAgent, IsoEntropySingleTurnAgent
from .physics import run_simulation, calculate_collapse_threshold
from .fsm import IsoEntropyFSM, AgentPhase
from .constraints import apply_hard_rules
from .grounding import ground_inputs
from .telemetry import build_llm_signal
from .prompt_templates import build_prompt_for_phase

__all__ = [
    "IsoEntropyAgent",
    "IsoEntropySingleTurnAgent",
    "run_simulation",
    "calculate_collapse_threshold",
    "IsoEntropyFSM",
    "AgentPhase",
    "apply_hard_rules",
    "ground_inputs",
    "build_llm_signal",
    "build_prompt_for_phase",
]
