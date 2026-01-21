# constraints.py - FIXED
"""
Hard Rules & Physical Constraints
=================================

CAMBIO CRÍTICO: Ya no lanza excepción por I > 1.5*K.
Eso es exactamente el tipo de sistema que el agente debe DIAGNOSTICAR,
no rechazar.

En su lugar:
- Log de advertencia
- Flag de "situación crítica"
- Dejar que el agente intente (o determine que es irrecuperable)
"""

from typing import Dict


class HardConstraintViolation(Exception):
    """Error crítico de física: el sistema no puede operar."""
    pass


def apply_hard_rules(
    *,
    volatilidad: str,
    rigidez: str,
    colchon_meses: int,
    params: Dict[str, float]
) -> Dict[str, float]:
    """
    Aplica reglas físicas PERO PERMITE SISTEMAS CRÍTICOS.

    - Ajusta parámetros inválidos
    - MARCA situación crítica (no lanza excepción)
    - Dejar que el agente diagnostique
    """

    I = params.get("I", 1.0)
    K = params.get("K", 1.0)
    liquidity = params.get("liquidity", 0.5)

    adjustments = []
    warnings = []

    # ------------------------------------------------------------------
    # 1️⃣ ENTROPÍA MÍNIMA SEGÚN VOLATILIDAD
    # ------------------------------------------------------------------
    if volatilidad == "Alta (Caótica)":
        if I < 4.5:
            I = 4.5
            adjustments.append("I ajustado a mínimo físico (4.5) por volatilidad alta")

    elif volatilidad == "Media (Estacional)":
        if I < 1.0:
            I = 1.0
            adjustments.append("I ajustado a mínimo físico (1.0) por volatilidad media")

    elif volatilidad == "Baja (Estable)":
        if I < 0.5:
            I = 0.5
            adjustments.append("I ajustado a mínimo físico (0.5) por volatilidad baja")

    # ------------------------------------------------------------------
    # 2️⃣ CAPACIDAD INICIAL SEGÚN RIGIDEZ
    # ------------------------------------------------------------------
    if rigidez == "Alta (Manual/Burocrático)":
        if K > 3.0:
            K = 3.0
            adjustments.append("K limitado a 3.0 por rigidez manual/burocrática")

    # ------------------------------------------------------------------
    # 3️⃣ LIQUIDEZ BAJA → CRECIMIENTO DE K MÁS AGRESIVO
    # ------------------------------------------------------------------
    if liquidity < 0.5:
        params["low_liquidity_penalty"] = True
        warnings.append("⚠️ Liquidez baja: ajustes de K serán menos efectivos")

    # ------------------------------------------------------------------
    # 4️⃣ DETECCIÓN DE CRÍTICA (pero permitir)
    # ------------------------------------------------------------------
    # CHANGE: En lugar de lanzar excepción, marcamos y prevenimos
    if I > K * 2.0:
        warnings.append(
            f"⚠️ CRÍTICA ESTRUCTURAL: I ({I:.2f}) >> 2.0 × K ({K:.2f}). "
            "El sistema está en riesgo extremo. El agente intentará diagnosticar."
        )
        params["critical_structural_warning"] = True
    
    elif I > K * 1.5:
        warnings.append(
            f"⚠️ SITUACIÓN MARGINAL: I ({I:.2f}) > 1.5 × K ({K:.2f}). "
            "Sistema frágil. Requiere intervención urgente."
        )
        params["marginal_situation"] = True

    # ------------------------------------------------------------------
    # 5️⃣ CLAMP FINAL ABSOLUTO
    # ------------------------------------------------------------------
    I = max(0.1, min(10.0, I))
    K = max(0.1, min(10.0, K))
    liquidity = max(0.0, min(1.0, liquidity))

    # ------------------------------------------------------------------
    # 6️⃣ PREVENCIÓN DE DIVISIÓN POR CERO
    # ------------------------------------------------------------------
    if K == 0:
        K = 0.1
        warnings.append("K clamped a 0.1 para prevenir división por cero")

    params.update({
        "I": I,
        "K": K,
        "liquidity": liquidity,
        "adjustments": adjustments,
        "warnings": warnings
    })

    return params