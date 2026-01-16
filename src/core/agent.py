# agent.py
import os
import json
import re
import math
from datetime import datetime

import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

from .physics import run_simulation, calculate_collapse_threshold
from .grounding import ground_inputs
from .fsm import IsoEntropyFSM, AgentPhase
from .prompt_templates import build_prompt_for_phase
from .telemetry import build_llm_signal

load_dotenv()


class IsoEntropyAgent:
    """
    Agente Iso-Entropy con:
    - Pre-Control duro
    - FSM canónica
    - Prompts por fase
    - Telemetría mínima
    """

    # =========================================================
    # INICIALIZACIÓN
    # =========================================================

    def __init__(self, model_name="gemini-3-flash-preview", log_callback=None, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.log_callback = log_callback

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.is_mock_mode = False
            except Exception:
                self.client = None
                self.is_mock_mode = True
        else:
            self.client = None
            self.is_mock_mode = True

        self.experiment_log = []
        self.fsm = IsoEntropyFSM()
        self.prompt_cache = {}  # Cache para prompts repetitivos

    # =========================================================
    # LOGGING
    # =========================================================

    def _log(self, message: str):
        print(message)
        if self.log_callback:
            try:
                self.log_callback(message)
            except Exception:
                pass

    # =========================================================
    # JSON ROBUSTO
    # =========================================================

    def _extract_json(self, text: str) -> dict:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(text)
        except Exception:
            return {}

    def _calculate_wilson_upper_bound(self, collapses: int, runs: int, confidence: float = 0.95) -> float:
        """
        Calcula el límite superior del intervalo de confianza de Wilson (95%).
        
        Args:
            collapses: Número de simulaciones que colapsaron
            runs: Número total de simulaciones
            confidence: Nivel de confianza (default 0.95, z=1.96)
        
        Returns:
            float: Límite superior del intervalo de confianza
        """
        if runs == 0:
            return 1.0
        
        z = 1.96  # Para 95% CI
        phat = collapses / runs
        n = runs
        
        denom = 1 + (z**2 / n)
        centre = phat + (z**2 / (2 * n))
        adj = z * math.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2)))
        upper = (centre + adj) / denom
        
        return min(1.0, upper)  # Clamp a [0, 1]

    def compress_simulation_state(self, experiment_log: list) -> dict:
        """
        Comprime el estado de simulación pidiendo a Gemini un resumen ejecutivo
        de la Deuda de Entropía (D_e) y la Incertidumbre (H(M)) acumulada.
        """
        if not experiment_log:
            return {"compressed": True, "summary": "Sin experimentos previos."}

        # Crear prompt para compresión
        log_summary = "\n".join([
            f"Ciclo {exp['ciclo']}: K={exp['hipotesis']['K']:.2f}, Colapso={exp['resultado']['tasa_de_colapso']:.1%}, Razonamiento: {exp.get('razonamiento_previo', 'N/A')}"
            for exp in experiment_log
        ])

        prompt = f"""
Eres un auditor de entropía especializado en termodinámica de la información.
Analiza el historial de experimentos y proporciona un resumen ejecutivo comprimido.

Historial de Experimentos:
{log_summary}

Calcula y resume:
- Deuda de Entropía acumulada (D_e): Basado en la acumulación de entropía no disipada.
- Incertidumbre del sistema (H(M)): Medida de la variabilidad o entropía en los resultados.

Proporciona un resumen ejecutivo conciso que capture el estado actual del sistema, tendencias y recomendaciones para continuar la simulación.

Respuesta en formato JSON:
{{
  "resumen_ejecutivo": "Texto conciso del resumen",
  "deuda_entropia_acumulada": float,
  "incertidumbre_sistema": float,
  "estado_actual": "Descripción breve",
  "tendencias": "Tendencias observadas",
  "recomendaciones": "Recomendaciones para próximos ciclos"
}}
"""

        if self.is_mock_mode:
            return {
                "compressed": True,
                "summary": {
                    "resumen_ejecutivo": "Mock: Estado comprimido simulado.",
                    "deuda_entropia_acumulada": 0.0,
                    "incertidumbre_sistema": 0.0,
                    "estado_actual": "Simulado",
                    "tendencias": "Estables",
                    "recomendaciones": "Continuar"
                }
            }

        try:
            generate_content_config = types.GenerateContentConfig(
                temperature=0.25,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=False,  # No incluir pensamientos para compresión
                    thinking_level="low"
                ),
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generate_content_config
            )

            summary = self._extract_json(response.text)
            return {"compressed": True, "summary": summary}
        except Exception as e:
            return {"compressed": True, "summary": f"Error en compresión: {e}"}

    # =========================================================
    # 🚫 PASO 1 — PRE-CONTROL (NO FSM)
    # =========================================================

    def should_call_llm(self, I, K_base, stock, liquidity, rigidez):
        self._log(
            f"DEBUG: should_call_llm | I={I:.2f}, K_base={K_base:.2f}, "
            f"stock={stock:.2f}, liquidity={liquidity:.2f}, "
            f"experimentos={len(self.experiment_log)}"
        )
        # 1️⃣ Colapso matemático
        if I > K_base * 1.5:
            return False, {
                "action": "TERMINATE",
                "reasoning": "Colapso determinista I >> K",
                "final_verdict": (
                    f"## ❌ Colapso Inevitable\n\n"
                    f"I = {I:.2f} > 1.5 × K = {K_base:.2f}\n"
                    "Fuera de región de control."
                )
            }

        # 2️⃣ Sin buffer físico
        if stock <= 0.0:
            return False, {
                "action": "TERMINATE",
                "reasoning": "Sin colchón físico",
                "final_verdict": "## ❌ Colapso por falta de buffer"
            }

        # 3️⃣ Liquidez crítica + rigidez alta
        if liquidity < 0.3 and "Alta" in rigidez:
            return False, {
                "action": "SIMULATE",
                "reasoning": "Liquidez crítica en sistema rígido",
                "parameters": {"K": K_base + 0.5}
            }

        # 4️⃣ FSM ORIENT sin grados de libertad
        if self.fsm.phase == AgentPhase.ORIENT:
            MAX_K_STEP = 0.75
            k_min = max(0.1, K_base - MAX_K_STEP)
            k_max = min(10.0, K_base + MAX_K_STEP)
            if abs(k_max - k_min) < 1e-6:
                # Si no hemos hecho ninguna simulación aún, NO terminamos: forzamos una simulación conservadora.
                if len(self.experiment_log) == 0:
                    return False, {
                        "action": "SIMULATE",
                        "reasoning": "No hay grados de libertad detectados pero no existen observaciones. Ejecutar prueba conservadora.",
                        "parameters": {"K": K_base + 0.1}
                    }
                else:
                    return False, {
                        "action": "TERMINATE",
                        "reasoning": "Sin grados de libertad en K",
                        "final_verdict": "## ⚠️ Sistema estancado"
                    }

        return True, None

    # =========================================================
    # 🤖 DECISIÓN LLM (PASO 2 + 3)
    # =========================================================

    def _decide_next_step(self, system_description: str) -> dict:
        # Telemetría mínima
        llm_signal = build_llm_signal(self.experiment_log)
        
        # 🔧 MEJORA CRÍTICA: Enriquecer signal con contexto de búsqueda
        search_context = self._build_search_context()
        if search_context:
            llm_signal.update(search_context)

        prompt = build_prompt_for_phase(
            phase=self.fsm.phase,
            phase_reasoning=self.fsm.phase_reasoning(),
            system_description=system_description,
            llm_signal=llm_signal
        )

        # Cache de prompts
        cache_key = hash(prompt)
        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]

        if self.is_mock_mode:
            # Mock mode: proporcionar decisiones inteligentes según la fase
            if self.fsm.phase == AgentPhase.ORIENT:
                decision = {"action": "SIMULATE", "parameters": {"K": 1.5}, "reasoning": "Mock: Explorando incremento de K"}
            elif self.fsm.phase == AgentPhase.VALIDATE:
                decision = {"action": "SIMULATE", "parameters": {"K": 1.5}, "reasoning": "Mock: Validando estabilidad"}
            elif self.fsm.phase == AgentPhase.STRESS:
                decision = {"action": "SIMULATE", "parameters": {"K": 1.5}, "reasoning": "Mock: Testeando fragilidad"}
            elif self.fsm.phase == AgentPhase.CONCLUDE:
                decision = {"action": "REPORT", "report_content": "Mock: Reporte de auditoría completado"}
            else:
                decision = {"action": "TERMINATE", "reasoning": "Mock: Fase desconocida"}
            self.prompt_cache[cache_key] = decision
            return decision

 # --- CONFIGURACIÓN DE THINKING Y TEMPERATURA ---
        generate_content_config = types.GenerateContentConfig(
            temperature=0.25,
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,  # Deshabilitado para reducir tokens
                thinking_level="low"  # Nivel bajo para optimizar costos
            ),
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generate_content_config
            )

            # Opcional: Si quieres ver los "thoughts" en la terminal/logs:
            if response.thoughts:
                self._log(f"\n💭 PENSAMIENTO INTERNO (Thinking):\n{response.thoughts}\n")


            if self.fsm.phase == AgentPhase.CONCLUDE:
                decision = {"action": "REPORT", "report_content": response.text}
            else:
                decision = self._extract_json(response.text)
                if "action" not in decision:
                    decision = {"action": "TERMINATE", "reasoning": "JSON response malformed or missing action."}
                
                # 🔧 VALIDACIÓN: Asegurar que decision tiene parámetros si es SIMULATE
                if decision.get("action") == "SIMULATE" and "parameters" not in decision:
                    decision["parameters"] = {"K": decision.get("K", 1.0)}

            self.prompt_cache[cache_key] = decision  # Cachear la decisión
            return decision
        except Exception as e:
            return {
                "action": "TERMINATE",
                "reasoning": f"Error técnico: {e}"
            }
    
    def _build_search_context(self) -> dict:
        """🔧 Construir contexto inteligente de búsqueda para guiar al LLM."""
        if len(self.experiment_log) == 0:
            return {}
        
        context = {}
        
        # ESTADÍSTICAS DE COLAPSO
        collapse_rates = [exp.get("resultado", {}).get("tasa_de_colapso", 0) for exp in self.experiment_log]
        context["colapso_min"] = min(collapse_rates)
        context["colapso_max"] = max(collapse_rates)
        context["colapso_promedio"] = sum(collapse_rates) / len(collapse_rates) if collapse_rates else 0
        
        # K VALORES TESTEADOS
        tested_K_values = [exp.get("parametros_completos", {}).get("K_auditor", 0) for exp in self.experiment_log]
        context["K_min_testeado"] = min(tested_K_values) if tested_K_values else 0
        context["K_max_testeado"] = max(tested_K_values) if tested_K_values else 0
        
        # TENDENCIA
        if len(collapse_rates) >= 2:
            recent_trend = collapse_rates[-1] - collapse_rates[-2]
            context["tendencia_colapso"] = "MEJORANDO" if recent_trend < 0 else "EMPEORANDO" if recent_trend > 0 else "ESTABLE"
            context["magnitud_cambio"] = abs(recent_trend)
        
        # ESTABILIDAD DETECTADA
        stable_experiments = [exp for exp in self.experiment_log if exp.get("resultado", {}).get("tasa_de_colapso", 0) < 0.05]
        context["experimentos_estables"] = len(stable_experiments)
        context["tasa_estabilidad"] = len(stable_experiments) / len(self.experiment_log) if self.experiment_log else 0
        
        return context

    # =========================================================
    # 🧠 LOOP PRINCIPAL
    # =========================================================

    def audit_system(self, user_input: str, volatilidad: str, colchon: int, rigidez: str) -> str:
        """
        LOOP AUTÓNOMO DEL AGENTE con K mínimo viable y reporte final completo.
        """
        # Reiniciar memoria
        self.experiment_log = []
        self.fsm = IsoEntropyFSM()  # 🔧 FIX: Reiniciar FSM también
        self.agent_state = {
            "phase": "INIT",
            "stable_runs": 0,
            "tested_K": set(),
            "last_collapse": None,
            "stable_hits": 0,
            "K_min_viable": None,
            "margin": None,
            "replicas_confirmadas": 0  # Contador de réplicas independientes que confirman K_min_viable
        }

        # Grounding físico
        physical_state = ground_inputs(
            volatilidad=volatilidad,
            rigidez=rigidez,
            colchon_meses=colchon
        )
        I_base = physical_state["I"]
        K_base = physical_state["K0"]
        stock_base = physical_state["stock"]
        liquidity_base = physical_state["liquidity"]
        capital_base = physical_state.get("capital", 1.0)

        # 🔧 FIX: Crear contexto enriquecido para el LLM
        calibration_context = {
            "volatilidad": volatilidad,
            "colchon": colchon,
            "rigidez": rigidez
        }

        self._log("🚀 INICIANDO AGENTE AUTÓNOMO GEMINI 3 PRO")
        self._log(f"📊 Calibración: {volatilidad} volatilidad, {rigidez} rigidez, {colchon} meses colchón")

        MAX_ITERATIONS = 10
        iteration = 0
        final_llm_report = None

        while iteration < MAX_ITERATIONS and self.fsm.phase != AgentPhase.CONCLUDE:
            iteration += 1
            self._log(f"\n{'='*60}")
            self._log(f"🧠 CICLO DE PENSAMIENTO #{iteration}")
            self._log(f"🔍 FSM_PHASE: {self.fsm.phase_name()}")
            self._log(f"{'='*60}")

            # Decidir si llamar a LLM o autoajustar
            should_call, auto_decision = self.should_call_llm(
                I=I_base, K_base=K_base, stock=stock_base, liquidity=liquidity_base, rigidez=rigidez
            )
            
            if not should_call:
                decision = auto_decision
            else:
                # 🔧 FIX: Crear prompt enriquecido con contexto completo
                system_prompt = f"""
{user_input}

Calibración del Sistema:
- Volatilidad: {volatilidad}
- Rigidez Operativa: {rigidez}
- Colchón Financiero: {colchon} meses

Parámetros Físicos Base:
- Entropía Externa (I): {I_base:.2f} bits
- Capacidad Inicial (K₀): {K_base:.2f} bits
- Stock Buffer: {stock_base:.2f}
- Liquidez: {liquidity_base:.2f}
"""
                decision = self._decide_next_step(system_prompt)

            action = decision.get("action", "UNKNOWN")
            self._log(f"DEBUG: acción recibida = {action}, experimentos = {len(self.experiment_log)}")
            # --- GUARD: no permitir TERMINATE sin al menos 1 experimento ---
            if action == "TERMINATE" and len(self.experiment_log) == 0:
                # Si la terminación fue generada por pre-control (auto_decision) y contiene 'final_verdict', permitir.
                # Para diferenciar, comprobamos si auto_decision está presente (variable 'should_call' / 'auto_decision').
                # Si no tienes acceso directo aquí a 'should_call' / 'auto_decision', consideramos prudente forzar una simulación.
                self._log("🧠 GUARD: No hay observaciones experimentales. Forzando 1ª SIMULATE.")
                # Forzar una simulación conservadora — mantenemos K_base para minimizar impacto
                decision = {
                    "action": "SIMULATE",
                    "reasoning": "Guard clause: primera simulación obligatoria para obtener evidencia empírica.",
                    "parameters": {"K": K_base}
                }
                action = "SIMULATE"
            # -------------------------------------------------------------

            reasoning = decision.get("reasoning", "Sin razonamiento proporcionado")

            self._log(f"\n💭 RAZONAMIENTO DEL AGENTE: {reasoning}")
            self._log(f"\n👉 DECISIÓN: {action}")

            if action == "SIMULATE":
                params = decision.get("parameters", {})

                # -----------------------------
                # 🌍 BASE FÍSICA (NO NEGOCIABLE)
                # -----------------------------
                I = I_base
                stock = stock_base
                liq = liquidity_base
                capital = capital_base

                # -----------------------------
                # 🤖 PROPUESTA DEL LLM (DELTA)
                # -----------------------------
                K_proposed = params.get("K", K_base)

                # -----------------------------
                # 🔒 CLAMP FÍSICO Y ACTION GATE
                # -----------------------------
                K = max(0.1, min(10.0, K_proposed))
                MAX_K_STEP = 0.75
                K = max(K_base - MAX_K_STEP, min(K, K_base + MAX_K_STEP))

                self._log(f"   🌍 Grounded State → I={I:.2f}, K_base={K_base:.2f}, stock={stock:.2f}, liquidity={liq:.2f}")
                if K != K_proposed:
                    self._log("   ⚠️ K ajustado por límites físicos (clamp/action gate)")

                # -----------------------------
                # 🔬 MOTOR FÍSICO
                # -----------------------------
                theta = calculate_collapse_threshold(stock, capital, liq)
                self._log(f"\n🧪 EXPERIMENTO #{iteration}: • I={I:.2f}, K={K:.2f}, θ_max={theta:.2f} • Ejecutando 500 simulaciones Monte Carlo...")
                sim_result = run_simulation(I, K, theta, runs=500)
                colapso_pct = sim_result["tasa_de_colapso"]
                collapses_total = sim_result.get("collapses_total", int(colapso_pct * 500))
                runs_total = sim_result.get("runs", 500)
                
                # Calcular Wilson CI Upper Bound
                ub95 = self._calculate_wilson_upper_bound(collapses_total, runs_total)

                if colapso_pct < 0.05:
                    emoji, status = "✅", "ESTABLE"
                elif colapso_pct < 0.15:
                    emoji, status = "⚠️", "MARGINAL"
                else:
                    emoji, status = "❌", "COLAPSO"

                self._log(f"\n📊 RESULTADO: {emoji} {status} • Tasa de colapso: {colapso_pct:.1%} • UB95: {ub95:.1%}")

                # Guardar memoria episódica
                self.experiment_log.append({
                    "ciclo": iteration,
                    "label": "initial",
                    "timestamp": datetime.now().isoformat(),
                    "hipotesis": {"I": I, "K": K},
                    "parametros_completos": {
                        "stock_ratio": stock,
                        "capital_ratio": capital,
                        "liquidity": liq,
                        "theta_max": theta
                    },
                    "resultado": {
                        "tasa_de_colapso": colapso_pct,
                        "upper_ci95": ub95,
                        "collapses_total": collapses_total,
                        "runs": runs_total,
                        "tiempo_promedio_colapso": sim_result.get("tiempo_promedio_colapso", float('inf'))
                    },
                    "razonamiento_previo": reasoning
                })
                
                # -----------------------------
                # 🔁 RE-EXPLORACIÓN FORZADA CON VALIDACIÓN ESTADÍSTICA
                # Si colapso ≥ 99% y K_min_viable es None, ejecutar al menos 2 simulaciones
                # adicionales con incrementos grandes de K (≥ +1.0 bits), validando estabilidad
                # con criterios estadísticos (colapso < 5% Y UB95 < 5%).
                # -----------------------------
                stability_threshold = 0.05
                marginal_threshold = 0.15
                forced_attempts = 2
                delta_K_step = 1.0
                
                try:
                    # Condición de activación: colapso ≥ 99% y K_min_viable es None
                    if colapso_pct >= 0.99 and self.agent_state.get("K_min_viable") is None:
                        self._log("\n🔎 Colapso ≥ 99% sin K_min_viable detectado. Iniciando re-exploración forzada con validación estadística...")
                        
                        marginal_K_candidates = []  # Para réplicas posteriores
                        
                        for attempt in range(1, forced_attempts + 1):
                            forced_K = max(0.1, min(10.0, K_base + attempt * delta_K_step))
                            # Permitir saltos grandes para exploración forzada
                            MAX_K_STEP = 0.75
                            forced_K = max(K_base - MAX_K_STEP, min(forced_K, K_base + MAX_K_STEP * 4))
                            
                            self._log(f"   🧪 Intento forzado #{attempt}: probando K = {forced_K:.2f}")

                            theta_f = calculate_collapse_threshold(stock, capital, liq)
                            self._log(f"      • Ejecutando 500 simulaciones Monte Carlo con I={I:.2f}, K={forced_K:.2f}, θ_max={theta_f:.2f}...")
                            sim_forced = run_simulation(I, forced_K, theta_f, runs=500)
                            forced_collapse = sim_forced.get("tasa_de_colapso", 1.0)
                            forced_collapses_total = sim_forced.get("collapses_total", int(forced_collapse * 500))
                            forced_runs = sim_forced.get("runs", 500)
                            
                            # Calcular Wilson CI Upper Bound
                            forced_ub95 = self._calculate_wilson_upper_bound(forced_collapses_total, forced_runs)

                            # Registrar experimento forzado en memoria
                            self.experiment_log.append({
                                "ciclo": f"{iteration}-f{attempt}",
                                "label": f"forced-{attempt}",
                                "timestamp": datetime.now().isoformat(),
                                "hipotesis": {"I": I, "K": forced_K},
                                "parametros_completos": {
                                    "stock_ratio": stock,
                                    "capital_ratio": capital,
                                    "liquidity": liq,
                                    "theta_max": theta_f
                                },
                                "resultado": {
                                    "tasa_de_colapso": forced_collapse,
                                    "upper_ci95": forced_ub95,
                                    "collapses_total": forced_collapses_total,
                                    "runs": forced_runs,
                                    "tiempo_promedio_colapso": sim_forced.get("tiempo_promedio_colapso", float('inf'))
                                },
                                "razonamiento_previo": f"Forced attempt #{attempt} tras colapso ≥ 99%"
                            })

                            self._log(f"      ➤ Resultado: Colapso = {forced_collapse:.1%}, UB95 = {forced_ub95:.1%}")

                            # Validar estabilidad estadística: colapso < 5% Y UB95 < 5%
                            if forced_collapse < stability_threshold and forced_ub95 < stability_threshold:
                                self._log(f"   ✅ Estabilidad estadística confirmada con K={forced_K:.2f}. Marcando K_min_viable.")
                                # Actualizar K_min_viable si aplica
                                if (self.agent_state.get("K_min_viable") is None) or (forced_K < self.agent_state.get("K_min_viable")):
                                    self.agent_state["K_min_viable"] = forced_K
                                    self.agent_state["margin"] = forced_K - I
                                # Actualizar K_base para siguientes iteraciones
                                K_base = forced_K
                                break
                            elif stability_threshold <= forced_collapse < marginal_threshold:
                                # Caso marginal: guardar para réplicas
                                marginal_K_candidates.append({
                                    "K": forced_K,
                                    "colapso": forced_collapse,
                                    "ub95": forced_ub95,
                                    "theta": theta_f
                                })
                                self._log(f"   ⚠️ Resultado marginal detectado. K={forced_K:.2f} será evaluado con réplicas.")
                            else:
                                # Si no funcionó, actualizar K_base para siguiente intento forzado
                                K_base = forced_K
                        
                        # -----------------------------
                        # LÓGICA DE RÉPLICAS PARA CASOS MARGINALES
                        # -----------------------------
                        if marginal_K_candidates and self.agent_state.get("K_min_viable") is None:
                            self._log("\n🔄 Ejecutando réplicas para casos marginales...")
                            for candidate in marginal_K_candidates:
                                replica_K = candidate["K"]
                                replica_theta = candidate["theta"]
                                
                                self._log(f"   🔁 Réplica para K={replica_K:.2f} con runs=1000...")
                                sim_replica = run_simulation(I, replica_K, replica_theta, runs=1000)
                                replica_collapse = sim_replica.get("tasa_de_colapso", 1.0)
                                replica_collapses_total = sim_replica.get("collapses_total", int(replica_collapse * 1000))
                                replica_runs = sim_replica.get("runs", 1000)
                                replica_ub95 = self._calculate_wilson_upper_bound(replica_collapses_total, replica_runs)
                                
                                # Registrar réplica
                                self.experiment_log.append({
                                    "ciclo": f"{iteration}-r{replica_K:.2f}",
                                    "label": "replica",
                                    "timestamp": datetime.now().isoformat(),
                                    "hipotesis": {"I": I, "K": replica_K},
                                    "parametros_completos": {
                                        "stock_ratio": stock,
                                        "capital_ratio": capital,
                                        "liquidity": liq,
                                        "theta_max": replica_theta
                                    },
                                    "resultado": {
                                        "tasa_de_colapso": replica_collapse,
                                        "upper_ci95": replica_ub95,
                                        "collapses_total": replica_collapses_total,
                                        "runs": replica_runs,
                                        "tiempo_promedio_colapso": sim_replica.get("tiempo_promedio_colapso", float('inf'))
                                    },
                                    "razonamiento_previo": f"Réplica para validar caso marginal K={replica_K:.2f}"
                                })
                                
                                self._log(f"      ➤ Réplica: Colapso = {replica_collapse:.1%}, UB95 = {replica_ub95:.1%}")
                                
                                # Validar estabilidad estadística en réplica
                                if replica_collapse < stability_threshold and replica_ub95 < stability_threshold:
                                    self._log(f"   ✅ Réplica confirma estabilidad estadística con K={replica_K:.2f}. Marcando K_min_viable.")
                                    if (self.agent_state.get("K_min_viable") is None) or (replica_K < self.agent_state.get("K_min_viable")):
                                        self.agent_state["K_min_viable"] = replica_K
                                        self.agent_state["margin"] = replica_K - I
                                        self.agent_state["replicas_confirmadas"] = 1  # Primera réplica
                                    elif abs(replica_K - self.agent_state.get("K_min_viable", 0)) < 0.01:
                                        # Réplica del mismo K_min_viable
                                        self.agent_state["replicas_confirmadas"] = self.agent_state.get("replicas_confirmadas", 0) + 1
                                    K_base = replica_K
                                    break
                                elif replica_collapse > marginal_threshold or replica_ub95 >= stability_threshold:
                                    self._log(f"   ❌ Réplica confirma fragilidad. K={replica_K:.2f} no es viable.")
                                    # Continuar con siguiente candidato o marcar FRÁGIL si no hay más
                                    
                except Exception as e:
                    # No romper el loop por fallo en esta etapa forzada
                    self._log(f"   ⚠️ Error durante re-exploración forzada: {e} (continuando).")
                
                # -----------------------------
                # FORZAR CONCLUDE DESDE ORIENT SI K_min_viable ES None DESPUÉS DE INTENTOS FORZADOS
                # -----------------------------
                # Contar intentos forzados ejecutados
                forced_executed = sum(1 for exp in self.experiment_log 
                                     if exp.get("label", "").startswith("forced"))
                if (self.fsm.phase == AgentPhase.ORIENT and 
                    self.agent_state.get("K_min_viable") is None and
                    forced_executed >= 2):
                    self._log("\n🔚 Forzando transición a CONCLUDE: K_min_viable no detectado tras intentos forzados en fase ORIENT.")
                    # Forzar transición a CONCLUDE para generar reporte FRÁGIL
                    self.fsm.phase = AgentPhase.CONCLUDE
                # -----------------------------


                # State Compressor: Comprimir después de 3 ciclos
                if len(self.experiment_log) > 3:
                    compressed_state = self.compress_simulation_state(self.experiment_log)
                    self.experiment_log = [compressed_state]
                    self._log("   📦 Estado de simulación comprimido para reducir tokens (80% menos).")

                # 🔧 FIX: Registrar K_min_viable en CUALQUIER fase si es estable estadísticamente
                # Validar estabilidad: colapso < 5% Y UB95 < 5%
                if colapso_pct < 0.05 and ub95 < 0.05:
                    if (self.agent_state["K_min_viable"] is None or 
                        K < self.agent_state["K_min_viable"]):
                        self.agent_state["K_min_viable"] = K
                        self.agent_state["margin"] = K - I
                        self.agent_state["replicas_confirmadas"] = 1  # Primera confirmación
                        self._log(f"   ✨ K mínimo viable detectado (estadísticamente confirmado): {K:.2f} bits (UB95={ub95:.1%})")
                    elif abs(K - self.agent_state.get("K_min_viable", 0)) < 0.01:
                        # Mismo K_min_viable, incrementar contador de réplicas
                        self.agent_state["replicas_confirmadas"] = self.agent_state.get("replicas_confirmadas", 0) + 1

                # Actualizar FSM con validación estadística
                try:
                    self.fsm.update(colapso_pct, ub95)
                except Exception:
                    pass

                # Actualizar K_base para siguiente iteración
                K_base = K

                # If FSM transitions to CONCLUDE, break the loop to generate final report
                if self.fsm.phase == AgentPhase.CONCLUDE:
                    self._log("\n🏁 FSM ha transicionado a CONCLUDE. Generando reporte final.")
                    break
                elif not self.fsm.allow_simulation():
                    self._log("\n🏁 FSM indica terminar exploración (no CONCLUDE).")
                    break

            elif action == "TERMINATE":
                break

            elif action == "REPORT":
                final_llm_report = decision.get("report_content")
                break

            else:
                self._log(f"\n⚠️ ACCIÓN NO RECONOCIDA: {action}")
                break
        
        # Si la FSM está en CONCLUDE, generar el reporte final
        if self.fsm.phase == AgentPhase.CONCLUDE:
            self._log("\n📄 GENERANDO REPORTE DE AUDITORÍA FINAL (FASE CONCLUDE)...")
            final_report_prompt = build_prompt_for_phase(
                phase=AgentPhase.CONCLUDE,
                phase_reasoning=self.fsm.phase_reasoning(),
                system_description=f"""
{user_input}

Calibración del Sistema:
- Volatilidad: {volatilidad}
- Rigidez Operativa: {rigidez}
- Colchón Financiero: {colchon} meses

Parámetros Físicos Base:
- Entropía Externa (I): {I_base:.2f} bits
- Capacidad Inicial (K₀): {K_base:.2f} bits
- Stock Buffer: {stock_base:.2f}
- Liquidez: {liquidity_base:.2f}

Historial de Experimentos (resumido):
{self._format_experiment_table()}
""",
                llm_signal=build_llm_signal(self.experiment_log)
            )
            
            if self.is_mock_mode:
                final_llm_report = "### [Critical Failure Point]\nMock: Sistema alcanzó punto crítico de fallo.\n\n### [Survival Horizon]\nMock: Horizonte de supervivencia estimado.\n\n### [Actionable Mitigation]\nMock: Propuesta de mitigación accionable."
            else:
                generate_content_config = types.GenerateContentConfig(
                    temperature=0.25,
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=False,
                        thinking_level="low"
                    ),
                )
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=final_report_prompt,
                        config=generate_content_config
                    )
                    final_llm_report = response.text
                except Exception as e:
                    self._log(f"Error al generar reporte final en fase CONCLUDE: {e}")
                    final_llm_report = f"Error al generar reporte final: {e}"

        # Si se generó un reporte desde CONCLUDE, usarlo directamente
        if final_llm_report:
            final_report = f"""# 🎯 Auditoría Forense - ISO-ENTROPÍA

## Contexto de Ejecución
- **Sistema Analizado:** {volatilidad} volatilidad, {rigidez} rigidez, {colchon} meses colchón
- **Experimentos Ejecutados:** {len(self.experiment_log)}
- **Fase FSM Final:** {self.fsm.phase_name()}

---

## 📋 Reporte Generado por Auditor (Gemini 3 Pro)

{final_llm_report}

---

## 📊 Datos de Respaldo (Historial Experimental)

{self._format_experiment_table()}

---
*Generado por Iso-Entropy Agent v2.2*  
*Powered by Gemini 3 Flash Preview*  
*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        else:
            # Si no hay reporte desde CONCLUDE, generar reporte estándar
            K_min = self.agent_state.get("K_min_viable")
            margin = self.agent_state.get("margin")
            
            # Calcular métricas estadísticas para determinar estado final
            all_experiments = [exp for exp in self.experiment_log if not exp.get("compressed")]
            
            if not all_experiments:
                # Si todos están comprimidos, usar datos comprimidos
                max_collapse = 0.0
                max_ub95 = 0.0
                avg_collapse_time = float('inf')
            else:
                max_collapse = max(
                    [exp["resultado"]["tasa_de_colapso"] for exp in all_experiments],
                    default=0.0
                )
                
                # Calcular tiempo promedio de colapso de experimentos críticos
                collapse_times = [
                    exp.get("resultado", {}).get("tiempo_promedio_colapso", 0)
                    for exp in all_experiments
                    if exp.get("resultado", {}).get("tasa_de_colapso", 0) > 0.15
                ]
                avg_collapse_time = (
                    sum(collapse_times) / len(collapse_times) 
                    if collapse_times else float('inf')
                )
            total_attempts = len(all_experiments)
            
            # Contar intentos forzados
            forced_count = sum(1 for exp in all_experiments if exp.get("label", "").startswith("forced"))
            initial_count = sum(1 for exp in all_experiments if exp.get("label") == "initial")
            
            # Verificar si se cumplieron criterios para FRÁGIL
            has_forced_attempts = forced_count >= 2
            has_initial = initial_count >= 1
            min_attempts_for_fragile = has_initial and has_forced_attempts
            
            # Calcular máximo UB95
            max_ub95 = max(
                [exp["resultado"].get("upper_ci95", exp["resultado"]["tasa_de_colapso"]) 
                 for exp in all_experiments],
                default=0.0
            )
            
            # Determinar estado final según definiciones formales
            replicas_confirmadas = self.agent_state.get("replicas_confirmadas", 0)
            fsm_phase = self.fsm.phase
            
            if K_min is not None:
                # Verificar que K_min tenga validación estadística
                k_min_experiments = [exp for exp in all_experiments 
                                    if abs(exp["hipotesis"]["K"] - K_min) < 0.01]
                k_min_statistically_stable = any(
                    exp["resultado"]["tasa_de_colapso"] < 0.05 and 
                    exp["resultado"].get("upper_ci95", 1.0) < 0.05
                    for exp in k_min_experiments
                )
                
                # ROBUSTO solo si: K_min existe + FSM no en ORIENT + ≥2 réplicas confirmadas
                if (k_min_statistically_stable and 
                    fsm_phase != AgentPhase.ORIENT and 
                    replicas_confirmadas >= 2):
                    estado_final = "✅ ROBUSTO"
                else:
                    # MARGINAL si K_min existe pero no cumple criterios de ROBUSTO
                    estado_final = "⚠️ MARGINAL"
            elif min_attempts_for_fragile:
                # FRÁGIL si: 1 inicial + ≥2 forzadas + ninguna cumple estabilidad + K_min_viable es None
                # Verificar que ninguna simulación cumple estabilidad estadística
                ninguna_estable = not any(
                    exp["resultado"]["tasa_de_colapso"] < 0.05 and 
                    exp["resultado"].get("upper_ci95", 1.0) < 0.05
                    for exp in all_experiments
                )
                if ninguna_estable:
                    estado_final = "❌ FRÁGIL"
                else:
                    estado_final = "⚠️ MARGINAL"
            elif max_collapse < 0.15 and max_ub95 < 0.15:
                estado_final = "⚠️ MARGINAL"
            else:
                # Caso por defecto: si no hay suficiente evidencia, MARGINAL
                estado_final = "⚠️ MARGINAL"
            
            final_report = f"""# 🎯 Diagnóstico de Fragilidad Estructural

## Resumen Ejecutivo
- **Sistema Analizado:** {volatilidad} volatilidad, {rigidez} rigidez, {colchon} meses colchón
- **Estado Final:** {estado_final}
- **Fase FSM Final:** {self.fsm.phase_name()}
- **Experimentos Ejecutados:** {total_attempts} (inicial: {initial_count}, forzados: {forced_count})

## 📊 Evidencia Experimental Observada
- **Máxima Probabilidad de Colapso Observada:** {max_collapse:.1%}
  - Esta es la tasa de colapso más alta medida en todas las simulaciones ejecutadas
- **UB95 Máximo:** {max_ub95:.1%}
  - Límite superior del intervalo de confianza (95%) más alto observado
"""
            
            # Agregar sección de estado final bajo K_min_viable si existe
            if K_min is not None:
                k_min_experiments = [exp for exp in all_experiments 
                                    if abs(exp["hipotesis"]["K"] - K_min) < 0.01]
                if k_min_experiments:
                    k_min_collapse = min(exp["resultado"]["tasa_de_colapso"] for exp in k_min_experiments)
                    k_min_ub95 = min(exp["resultado"].get("upper_ci95", 1.0) for exp in k_min_experiments)
                    final_report += f"""
## 🎯 Estado Final Bajo K_min_viable (Proyección)
- **Probabilidad de Colapso bajo K={K_min:.2f} bits:** {k_min_collapse:.1%}
- **UB95 bajo K={K_min:.2f} bits:** {k_min_ub95:.1%}
- **Nota:** Esta proyección asume que el sistema opera con K aumentado hasta {K_min:.2f} bits
"""
            
            final_report += """
## 🔬 Hallazgos Clave
"""

            if K_min is not None:
                # Distinguir entre robustez actual vs. potencial
                if max_collapse > 0.15:
                    # Hubo colapsos previos altos - robustez potencial
                    final_report += f"""
- **Región Estable Detectada (requiere aumentar K):** {K_min:.2f} bits
  - Aunque se observaron colapsos del {max_collapse:.1%} en configuraciones iniciales, el sistema presenta una región estable al aumentar K hasta {K_min:.2f} bits
  - Capacidad mínima requerida para mantener estabilidad (colapso < 5% y UB95 < 5%)
  - Réplicas independientes confirmadas: {replicas_confirmadas}
"""
                    # Advertencia si FSM está en ORIENT
                    if fsm_phase == AgentPhase.ORIENT:
                        final_report += f"""
  - ⚠️ **Nota:** El sistema se encuentra en fase ORIENT. La robustez requiere validación adicional (fase VALIDATE) con ≥2 réplicas independientes confirmadas.
"""
                    final_report += f"""
- **Margen de Seguridad sobre I:** {margin:.2f} bits
  - Exceso de capacidad disponible para absorber picos
"""
                else:
                    # No hubo colapsos altos - robustez actual
                    final_report += f"""
- **K Mínimo Viable Detectado (estadísticamente confirmado):** {K_min:.2f} bits
  - El sistema muestra estabilidad estructural en la configuración actual
  - Capacidad mínima requerida para mantener estabilidad (colapso < 5% y UB95 < 5%)
  - Réplicas independientes confirmadas: {replicas_confirmadas}
"""
                    # Advertencia si FSM está en ORIENT
                    if fsm_phase == AgentPhase.ORIENT:
                        final_report += f"""
  - ⚠️ **Nota:** El sistema se encuentra en fase ORIENT. La robustez requiere validación adicional (fase VALIDATE) con ≥2 réplicas independientes confirmadas.
"""
                    final_report += f"""
- **Margen de Seguridad sobre I:** {margin:.2f} bits
  - Exceso de capacidad disponible para absorber picos
"""
            else:
                # Justificación rigurosa de FRÁGIL
                k_values_tested = [exp["hipotesis"]["K"] for exp in all_experiments]
                k_values_str = ", ".join([f"{k:.2f}" for k in sorted(set(k_values_tested))])
                
                final_report += f"""
- **K Mínimo Viable:** No detectado durante auditoría
  - **Justificación estadística:** No se encontró K con UB95<5% tras probar {len(set(k_values_tested))} valores de K crecientes: [{k_values_str}] bits
  - Ninguna simulación cumplió simultáneamente: colapso < 5% **Y** UB95 < 5%
  - Sistema declarado FRÁGIL tras {total_attempts} experimentos (1 inicial + {forced_count} forzados)
"""

            if avg_collapse_time != float('inf'):
                final_report += f"- **Tiempo Promedio de Colapso:** {avg_collapse_time:.1f} semanas (en escenarios críticos)\n"

            # Recomendaciones accionables
            final_report += "\n## 💡 Recomendaciones Accionables\n\n"
            
            if "Alta" in rigidez:
                final_report += "1. **Automatizar Procesos Críticos:** La rigidez operativa limita severamente K. Invertir en automatización.\n"
            else:
                final_report += "1. **Mantener Agilidad Operativa:** El sistema muestra buena flexibilidad. Preservar cultura ágil.\n"
            
            if "Alta" in volatilidad:
                final_report += "2. **Diversificar Cadena de Suministro:** Alta entropía externa requiere redundancia en proveedores clave.\n"
            else:
                final_report += "2. **Mantener Monitoreo Proactivo:** Volatilidad moderada permite enfoque preventivo.\n"
            
            if colchon < 6:
                final_report += f"3. **Extender Runway Financiero:** {colchon} meses es insuficiente. Objetivo: 6-12 meses mínimo.\n"
            else:
                final_report += "3. **Colchón Financiero Adecuado:** Runway actual proporciona buffer razonable.\n"

            # Tabla de experimentos
            final_report += "\n## 📊 Historial de Experimentos\n\n"
            if len(self.experiment_log) == 1 and self.experiment_log[0].get("compressed"):
                summary = self.experiment_log[0]["summary"]
                if isinstance(summary, dict):
                    final_report += f"**Estado Comprimido:** {summary.get('resumen_ejecutivo', 'N/A')}\n\n"
                    final_report += f"- Deuda de Entropía Acumulada: {summary.get('deuda_entropia_acumulada', 0.0):.2f}\n"
                    final_report += f"- Incertidumbre del Sistema: {summary.get('incertidumbre_sistema', 0.0):.2f}\n"
                    final_report += f"- Estado Actual: {summary.get('estado_actual', 'N/A')}\n"
                    final_report += f"- Tendencias: {summary.get('tendencias', 'N/A')}\n"
                    final_report += f"- Recomendaciones: {summary.get('recomendaciones', 'N/A')}\n"
                else:
                    final_report += f"**Estado Comprimido:** {str(summary)}\n"
            else:
                final_report += "| Ciclo | K (bits) | Colapso (%) | UB95 (%) | Estado |\n"
                final_report += "|-------|----------|-------------|----------|--------|\n"

                for exp in all_experiments:
                    k_val = exp["hipotesis"]["K"]
                    collapse = exp["resultado"]["tasa_de_colapso"]
                    ub95 = exp["resultado"].get("upper_ci95", collapse)
                    estado = "✅" if collapse < 0.05 and ub95 < 0.05 else "⚠️" if collapse < 0.15 else "❌"
                    label_info = f" ({exp.get('label', 'N/A')})" if exp.get('label') else ""
                    final_report += f"| {exp.get('ciclo', 'N/A')}{label_info} | {k_val:.2f} | {collapse:.1%} | {ub95:.1%} | {estado} |\n"

            final_report += f"""
---
*Generado por Iso-Entropy Agent v2.2*  
*Powered by Gemini 3 Flash Preview*  
*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        return final_report

    # =========================================================
    # 📊 UTILIDADES
    # =========================================================
    
    def _format_experiment_table(self) -> str:
        """Genera tabla markdown de experimentos con métricas estadísticas."""
        if not self.experiment_log:
            return "*No hay experimentos registrados*"
        
        # Verificar si hay datos comprimidos
        if len(self.experiment_log) == 1 and self.experiment_log[0].get("compressed"):
            return "*Estado comprimido - ver detalles en reporte*"
        
        table = "| Ciclo | K (bits) | Colapso (%) | UB95 (%) | Estado |\n"
        table += "|-------|----------|-------------|----------|--------|\n"
        
        for exp in self.experiment_log:
            k_val = exp["hipotesis"]["K"]
            collapse = exp["resultado"]["tasa_de_colapso"]
            ub95 = exp["resultado"].get("upper_ci95", collapse)  # Fallback a colapso si no hay UB95
            estado = "✅" if collapse < 0.05 and ub95 < 0.05 else "⚠️" if collapse < 0.15 else "❌"
            table += f"| {exp.get('ciclo', 'N/A')} | {k_val:.2f} | {collapse:.1%} | {ub95:.1%} | {estado} |\n"
        
        return table