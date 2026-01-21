# telemetry.py
from typing import List, Dict, Optional


def build_llm_signal(experiment_log: List[Dict]) -> Dict:
    """
    Construye una señal resumida para el LLM (Context Pruning).
    Resume el historial completo en estadísticas clave para evitar sobrecarga de contexto.
    Maneja estados comprimidos.
    """

    if not experiment_log:
        return {
            "experiments": 0,
            "trend": "none"
        }

    # Verificar si está comprimido
    if len(experiment_log) == 1 and experiment_log[0].get("compressed"):
        summary = experiment_log[0]["summary"]
        if isinstance(summary, dict):
            return {
                "experiments": "comprimido",
                "compressed_summary": summary.get("resumen_ejecutivo", "Comprimido"),
                "deuda_entropia": summary.get("deuda_entropia_acumulada", 0.0),
                "incertidumbre": summary.get("incertidumbre_sistema", 0.0),
                "estado_actual": summary.get("estado_actual", "Desconocido"),
                "tendencias": summary.get("tendencias", "N/A"),
                "recomendaciones": summary.get("recomendaciones", "N/A")
            }
        else:
            return {
                "experiments": "comprimido",
                "compressed_summary": str(summary)
            }

    # Filtrar entradas inválidas (sin resultado o sin hipotesis)
    valid_experiments = [
        exp for exp in experiment_log 
        if exp.get("resultado") and exp.get("hipotesis")
    ]
    
    if not valid_experiments:
        return {
            "experiments": len(experiment_log),
            "min_collapse_rate": 0.0,
            "max_collapse_rate": 0.0,
            "avg_collapse_rate": 0.0,
            "last_collapse_rate": 0.0,
            "last_K": 0.0,
            "k_range": "0.00 - 0.00",
            "theta_max_range": "0.00 - 0.00",
            "entropy_debt_accumulated": 0.0,
            "last_theta_max": 0.0,
            "overall_trend": "none"
        }
    
    # Extraer tasas de colapso con validación defensiva
    collapse_rates = [
        exp.get("resultado", {}).get("tasa_de_colapso", 0.0) 
        for exp in valid_experiments
    ]
    k_values = [
        exp.get("hipotesis", {}).get("K", 0.0) 
        for exp in valid_experiments
    ]
    theta_max_values = [
        exp.get("parametros_completos", {}).get("theta_max", 0.0) 
        for exp in valid_experiments 
        if exp.get("parametros_completos")
    ]
    
    # Calcular deuda de entropía acumulada (I - K no disipada)
    entropy_debt = 0.0
    for exp in valid_experiments:
        I = exp.get("hipotesis", {}).get("I", 0.0)
        K = exp.get("hipotesis", {}).get("K", 0.0)
        collapse_rate = exp.get("resultado", {}).get("tasa_de_colapso", 0.0)
        if I > K:
            entropy_debt += (I - K) * collapse_rate  # Ponderada por probabilidad de colapso

    # Estadísticas resumidas
    signal = {
        "experiments": len(valid_experiments),
        "min_collapse_rate": min(collapse_rates),
        "max_collapse_rate": max(collapse_rates),
        "avg_collapse_rate": sum(collapse_rates) / len(collapse_rates),
        "last_collapse_rate": collapse_rates[-1],
        "last_K": k_values[-1],
        "k_range": f"{min(k_values):.2f} - {max(k_values):.2f}",
        "theta_max_range": f"{min(theta_max_values) if theta_max_values else 0.0:.2f} - {max(theta_max_values) if theta_max_values else 0.0:.2f}",
        "entropy_debt_accumulated": entropy_debt,
        "last_theta_max": theta_max_values[-1] if theta_max_values else 0.0
    }

    # Tendencia general
    if len(collapse_rates) >= 2:
        first_half = collapse_rates[:len(collapse_rates)//2]
        second_half = collapse_rates[len(collapse_rates)//2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        delta = avg_first - avg_second
        signal["overall_trend"] = (
            "improving" if delta > 0.01
            else "worsening" if delta < -0.01
            else "stable"
        )

    return signal
