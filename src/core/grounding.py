# grounding.py
# ============================================
# COMPONENTE 2 — Grounding Físico
# Convierte inputs humanos estructurados
# en variables físicas canónicas.
# ============================================

from typing import Dict


def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


def ground_inputs(volatilidad: str, rigidez: str, colchon_meses: int) -> Dict[str, float]:
    """
    Grounding físico determinista.
    CALIBRADO PARA DEMO: Valores ajustados para generar tensión narrativa.
    """

    # 1. ENTROPÍA EXTERNA (I) - El "Caos"
    # Alta = 5.0 (Peligroso pero sobrevivible si el agente es listo)
    volatilidad_map = {
        "Baja (Estable)": 0.6,
        "Media (Estacional)": 1.5,
        "Alta (Caótica)": 5.0 
    }
    # Si no encuentra la clave, usa 5.0 por defecto
    I = volatilidad_map.get(volatilidad, 5.0)

    # 2. CAPACIDAD INICIAL (K0) - La "Respuesta"
    # Rigidez Alta = K bajo (0.8). Ratio I/K = 5.0/0.8 = 6.25 (CRÍTICO)
    rigidez_map = {
        "Baja (Automatizada)": 3.0,
        "Media (Estándar)": 1.5,
        "Alta (Manual/Burocrático)": 0.8
    }
    K0 = rigidez_map.get(rigidez, 0.8)

    # 3. BUFFER FÍSICO (STOCK)
    # Normalizamos meses a un ratio (ej: 6 meses = 0.25)
    stock = clamp(colchon_meses / 24.0, 0.05, 1.0)

    # 4. LIQUIDEZ (Fricción)
    # Alta rigidez suele implicar baja liquidez operativa
    if "Alta" in rigidez:
        liquidity = 0.3
    elif "Media" in rigidez:
        liquidity = 0.6
    else:
        liquidity = 0.9

    # 5. CAPITAL (Valor base)
    capital = 1.0

    return {
        "I": I,
        "K0": K0,
        "stock": stock,
        "liquidity": liquidity,
        "capital": capital
    }