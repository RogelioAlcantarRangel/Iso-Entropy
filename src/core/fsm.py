# fsm.py
from enum import Enum, auto
from typing import Optional


class AgentPhase(Enum):
    ORIENT = auto()
    VALIDATE = auto()
    STRESS = auto()
    CONCLUDE = auto()


class IsoEntropyFSM:
    def __init__(self):
        self.phase: AgentPhase = AgentPhase.ORIENT
        self.stable_hits: int = 0

    def update(self, collapse_rate: Optional[float], upper_ci95: Optional[float] = None):
        """
        Actualiza la FSM basándose en el colapso y validación estadística.
        
        Args:
            collapse_rate: Tasa de colapso observada
            upper_ci95: Límite superior del intervalo de confianza de Wilson (95%)
        """
        if collapse_rate is None:
            return

        stability_threshold = 0.05
        
        # Validar estabilidad estadística: colapso < 5% Y UB95 < 5% (si se proporciona)
        is_statistically_stable = collapse_rate < stability_threshold
        if upper_ci95 is not None:
            is_statistically_stable = is_statistically_stable and (upper_ci95 < stability_threshold)

        if self.phase == AgentPhase.ORIENT:
            if is_statistically_stable:
                self.phase = AgentPhase.VALIDATE
                self.stable_hits = 1
            else:
                self.phase = AgentPhase.ORIENT

        elif self.phase == AgentPhase.VALIDATE:
            if is_statistically_stable:
                self.stable_hits += 1
                if self.stable_hits >= 2:
                    self.phase = AgentPhase.STRESS
            else:
                self.phase = AgentPhase.ORIENT
                self.stable_hits = 0

        elif self.phase == AgentPhase.STRESS:
            self.phase = AgentPhase.CONCLUDE

    def allow_simulation(self) -> bool:
        return self.phase != AgentPhase.CONCLUDE

    def phase_reasoning(self) -> str:
        return {
            AgentPhase.ORIENT:
                "Diagnosticar si el sistema puede entrar en una región estable.",
            AgentPhase.VALIDATE:
                "Confirmar que la estabilidad observada no es estadística.",
            AgentPhase.STRESS:
                "Detectar fragilidad estructural bajo estrés controlado.",
            AgentPhase.CONCLUDE:
                "La ganancia informacional adicional es marginal."
        }[self.phase]

    def phase_name(self) -> str:
        return self.phase.name
