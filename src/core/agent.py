# agent.py
import os
import json
import re
import math
import time
from datetime import datetime

import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

from .physics import run_simulation, calculate_collapse_threshold
from .grounding import ground_inputs
from .fsm import IsoEntropyFSM, AgentPhase
from .prompt_templates import build_prompt_for_phase
from .telemetry import build_llm_signal
from .constraints import apply_hard_rules, HardConstraintViolation

load_dotenv()


class RateLimiter:
    """
    Rate limiter para respetar límite de 5 RPM de Gemini 3 Flash.
    Gestiona cola de timestamps y backoff exponencial ante errores 429.
    """
    
    def __init__(self, max_requests: int = 5, time_window: int = 60, min_interval: float = 15.0):
        """
        Args:
            max_requests: Máximo número de peticiones permitidas (default: 5)
            time_window: Ventana de tiempo en segundos (default: 60 segundos = 1 minuto)
            min_interval: Intervalo mínimo entre peticiones en segundos (default: 15.0)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.min_interval = min_interval
        self.request_timestamps = []  # Cola de timestamps de últimas peticiones
        self.backoff_multiplier = 1.0  # Multiplicador para backoff exponencial
        self.lock = False  # Simple lock (no thread-safe, pero suficiente para uso actual)
    
    def wait_if_needed(self):
        """
        Espera el tiempo necesario para respetar el rate limit.
        Calcula dinámicamente el tiempo de espera basado en las últimas peticiones.
        """
        current_time = time.time()
        
        # Limpiar timestamps fuera de la ventana de tiempo
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < self.time_window
        ]
        
        # Si ya tenemos max_requests en la ventana, esperar hasta que expire la más antigua
        if len(self.request_timestamps) >= self.max_requests:
            oldest_timestamp = min(self.request_timestamps)
            wait_time = self.time_window - (current_time - oldest_timestamp) + 1.0  # +1 segundo de margen
            if wait_time > 0:
                self._log(f"⏳ Rate limit: esperando {wait_time:.1f}s (5 RPM alcanzado)")
                time.sleep(wait_time)
                current_time = time.time()
        
        # Asegurar intervalo mínimo entre peticiones
        if self.request_timestamps:
            last_request_time = max(self.request_timestamps)
            time_since_last = current_time - last_request_time
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                time.sleep(wait_time)
                current_time = time.time()
        
        # Registrar esta petición
        self.request_timestamps.append(current_time)
        
        # Limpiar nuevamente después de agregar
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < self.time_window
        ]
    
    def handle_rate_limit_error(self):
        """
        Maneja un error 429 (Too Many Requests) aplicando backoff exponencial.
        """
        self.backoff_multiplier *= 2.0  # Duplicar tiempo de espera
        max_backoff = 120.0  # Máximo 2 minutos
        backoff_time = min(self.min_interval * self.backoff_multiplier, max_backoff)
        self._log(f"⚠️ Rate limit error (429): aplicando backoff exponencial de {backoff_time:.1f}s")
        time.sleep(backoff_time)
    
    def reset_backoff(self):
        """Resetea el multiplicador de backoff tras una petición exitosa."""
        self.backoff_multiplier = 1.0
    
    def _log(self, message: str):
        """Log interno (puede ser sobrescrito por el agente)."""
        print(message)


class IsoEntropyAgent:
    """
    Agente Iso-Entropy con:
    - Pre-Control duro
    - FSM canónica
    - Prompts por fase
    - Telemetría mínima
    """
    
    # =========================================================
    # PARÁMETROS CONFIGURABLES
    # =========================================================
    STABILITY_THRESHOLD = 0.05
    MARGINAL_THRESHOLD = 0.15
    FORCED_ATTEMPTS = 3  # Aumentado de 2 a 3 para casos críticos
    DELTA_K_STEP = 1.5  # Aumentado de 1.0 a 1.5 para incrementos más agresivos
    REPLICA_RUNS = 1000
    
    # Parámetros de accesibilidad estructural
    FACTOR_MAX = 1.5  # K_min puede ser hasta 1.5x K_base
    DELTA_K_TOLERABLE = 0.5  # Incremento absoluto máximo tolerable (bits)
    MARGIN_MIN = 0.2  # Margen mínimo sobre I para considerar "holgado"

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
        self.rate_limiter = RateLimiter(max_requests=5, time_window=60, min_interval=15.0)
        # Conectar el log del rate limiter con el del agente
        self.rate_limiter._log = self._log

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

    def _is_structural_accessible(self, K_min_viable: float, K_base_initial: float, I: float, margin: float):
        """
        Verifica si K_min_viable es estructuralmente accesible desde K_base_initial.
        
        Criterios de accesibilidad estructural:
        - K_min_viable / K_base_initial ≤ FACTOR_MAX (1.5 por defecto)
        - O K_min_viable - K_base_initial ≤ DELTA_K_TOLERABLE (0.5 bits por defecto)
        - Y margen (K_min_viable - I) debe ser suficientemente holgado (≥ 0.2 bits)
        
        Args:
            K_min_viable: Capacidad mínima viable detectada
            K_base_initial: Capacidad inicial del sistema (K0)
            I: Entropía externa
            margin: Margen de seguridad (K_min_viable - I)
        
        Returns:
            tuple: (es_accesible: bool, razon: str)
        """
        if K_base_initial <= 0:
            return False, "K_base_initial inválido"
        
        factor_ratio = K_min_viable / K_base_initial
        delta_absolute = K_min_viable - K_base_initial
        
        if factor_ratio > self.FACTOR_MAX:
            return False, f"K_min_viable {K_min_viable:.2f} requiere factor {factor_ratio:.2f}× sobre K_base {K_base_initial:.2f} (> {self.FACTOR_MAX})"
        if delta_absolute > self.DELTA_K_TOLERABLE:
            return False, f"Incremento {delta_absolute:.2f} bits sobre K_base {K_base_initial:.2f} excede tolerancia {self.DELTA_K_TOLERABLE:.2f}"
        if margin < self.MARGIN_MIN:
            return False, f"Margen {margin:.2f} bits insuficiente (< {self.MARGIN_MIN:.2f})"
        return True, "Estructuralmente accesible"

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
            # Rate limiting antes de llamada a API
            self.rate_limiter.wait_if_needed()
            
            generate_content_config = types.GenerateContentConfig(
                temperature=0.25,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=False,  # No incluir pensamientos para compresión
                    thinking_level="low"
                ),
            )

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=generate_content_config
                )
                self.rate_limiter.reset_backoff()  # Resetear backoff tras éxito
            except Exception as api_error:
                # Manejar errores 429 (Rate Limit)
                if "429" in str(api_error) or "rate limit" in str(api_error).lower():
                    self.rate_limiter.handle_rate_limit_error()
                    # Reintentar una vez después del backoff
                    self.rate_limiter.wait_if_needed()
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=generate_content_config
                    )
                    self.rate_limiter.reset_backoff()
                else:
                    raise

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
                "reasoning": "Insolvencia Informacional Persistente: Violación de la Ley de Ashby (I > K)",
                "final_verdict": (
                    f"## ❌ Colapso Inevitable\n\n"
                    f"**Insolvencia Informacional Persistente:** I = {I:.2f} > K = {K_base:.2f}\n"
                    f"**Causa raíz:** Violación de la Ley de Ashby - la entropía externa supera persistentemente la capacidad de respuesta.\n"
                    "El sistema es FRÁGIL debido a incapacidad estructural para homeostasis informacional."
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
                decision = {"action": "SIMULATE", "parameters": {"K": 1.5}, "reasoning": "Mock: Explorando incremento de K", "_internal_thoughts": "Mock thinking"}
            elif self.fsm.phase == AgentPhase.VALIDATE:
                decision = {"action": "SIMULATE", "parameters": {"K": 1.5}, "reasoning": "Mock: Validando estabilidad", "_internal_thoughts": "Mock thinking"}
            elif self.fsm.phase == AgentPhase.STRESS:
                decision = {"action": "SIMULATE", "parameters": {"K": 1.5}, "reasoning": "Mock: Testeando fragilidad", "_internal_thoughts": "Mock thinking"}
            elif self.fsm.phase == AgentPhase.CONCLUDE:
                decision = {"action": "REPORT", "report_content": "Mock: Reporte de auditoría completado", "_internal_thoughts": "Mock thinking"}
            else:
                decision = {"action": "TERMINATE", "reasoning": "Mock: Fase desconocida"}
            self.prompt_cache[cache_key] = decision
            return decision

        # --- INICIO DEL CAMBIO QUIRÚRGICO ---
        
        # 1. Rate limiting antes de llamada a API
        self.rate_limiter.wait_if_needed()
        
        # 2. Configuración para activar Thinking
        generate_content_config = types.GenerateContentConfig(
            temperature=0.25,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,  # <--- ESTO ES LO QUE FALTABA
                thinking_level="low"
            ),
        )

        try:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=generate_content_config
                )
                self.rate_limiter.reset_backoff()  # Resetear backoff tras éxito
            except Exception as api_error:
                # Manejar errores 429 (Rate Limit)
                if "429" in str(api_error) or "rate limit" in str(api_error).lower():
                    self.rate_limiter.handle_rate_limit_error()
                    # Reintentar una vez después del backoff
                    self.rate_limiter.wait_if_needed()
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=generate_content_config
                    )
                    self.rate_limiter.reset_backoff()
                else:
                    raise

            # 3. Capturar Pensamientos (Lógica segura para evitar errores)
            thoughts = "No disponible"
            try:
                # Intentar extraer thoughts de los candidatos según documentación Gemini 3
                if hasattr(response, 'candidates') and response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'thought') and part.thought:
                            # CORRECCIÓN: part.thought puede tener .text o ser directamente texto
                            if hasattr(part.thought, 'text'):
                                thoughts = part.thought.text
                            elif isinstance(part.thought, str):
                                thoughts = part.thought
                            elif hasattr(part.thought, '__str__'):
                                thoughts = str(part.thought)
                            break
            except Exception as e:
                self._log(f"⚠️ Error extrayendo thinking: {e}")

            # 4. Loguear el pensamiento para que aparezca en la UI (app.py)
            if thoughts != "No disponible" and isinstance(thoughts, str):
                self._log(f"\n🧠 PENSAMIENTO (Chain-of-Thought):\n{thoughts}\n")

        # --- FIN DEL CAMBIO QUIRÚRGICO ---

            if self.fsm.phase == AgentPhase.CONCLUDE:
                decision = {"action": "REPORT", "report_content": response.text}
            else:
                decision = self._extract_json(response.text)
                if "action" not in decision:
                    decision = {"action": "TERMINATE", "reasoning": "JSON response malformed or missing action."}
                
                # 🔧 VALIDACIÓN: Asegurar que decision tiene parámetros si es SIMULATE
                if decision.get("action") == "SIMULATE" and "parameters" not in decision:
                    decision["parameters"] = {"K": decision.get("K", 1.0)}

            # INYECTAR PENSAMIENTO EN LA DECISIÓN
            if isinstance(decision, dict):
                decision["_internal_thoughts"] = thoughts

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
        tested_K_values = [exp.get("hipotesis", {}).get("K", 0) for exp in self.experiment_log]
        context["K_min_testeado"] = min(tested_K_values) if tested_K_values else 0
        context["K_max_testeado"] = max(tested_K_values) if tested_K_values else 0
        
        # TENDENCIA
        if len(collapse_rates) >= 2:
            recent_trend = collapse_rates[-1] - collapse_rates[-2]
            context["tendencia_colapso"] = "MEJORANDO" if recent_trend < 0 else "EMPEORANDO" if recent_trend > 0 else "ESTABLE"
            context["magnitud_cambio"] = abs(recent_trend)
        
        # ESTABILIDAD DETECTADA
        stable_experiments = [exp for exp in self.experiment_log 
                             if exp.get("resultado", {}).get("tasa_de_colapso", 0) < self.STABILITY_THRESHOLD]
        context["experimentos_estables"] = len(stable_experiments)
        context["tasa_estabilidad"] = len(stable_experiments) / len(self.experiment_log) if self.experiment_log else 0
        
        return context

    # =========================================================
    # 🧠 LOOP PRINCIPAL
    # =========================================================

    def audit_system(self, user_input: str, volatilidad: str, colchon: int, rigidez: str) -> str:
        """
        Auditoría single-turn con function calling automático limitado a run_simulation.
        Ejecuta ground_inputs localmente y configura el agente con datos listos.
        """
        # Reiniciar estado
        self.experiment_log = []

        # 1. EJECUCIÓN LOCAL: Ejecutar ground_inputs directamente
        physical_state = ground_inputs(volatilidad=volatilidad, rigidez=rigidez, colchon_meses=colchon)
        I = physical_state["I"]
        K0 = physical_state["K0"]
        stock = physical_state["stock"]
        liquidity = physical_state["liquidity"]
        capital = physical_state.get("capital", 1.0)

        # Calcular theta_max una vez
        theta_max = calculate_collapse_threshold(stock, capital, liquidity)

        self._log("🚀 INICIANDO AGENTE CON FUNCTION CALLING LIMITADO")
        self._log(f"📊 Parámetros físicos: I={I:.2f}, K0={K0:.2f}, stock={stock:.2f}, liquidity={liquidity:.2f}, theta_max={theta_max:.2f}")

        # 2. INYECCIÓN DE CONTEXTO: Crear system_instruction con valores calculados
        system_instruction = f"""
Eres un Auditor de Entropía Autónoma especializado en termodinámica de la información.
Tu tarea es analizar el sistema usando ÚNICAMENTE la herramienta run_simulation para ejecutar simulaciones Monte Carlo.

CONTEXTO DEL USUARIO:
{user_input}

CALIBRACIÓN DEL SISTEMA:
- Volatilidad: {volatilidad}
- Rigidez Operativa: {rigidez}
- Colchón Financiero: {colchon} meses

PARÁMETROS FÍSICOS CALCULADOS (ya disponibles):
- Entropía Externa (I): {I:.2f} bits
- Capacidad Inicial (K₀): {K0:.2f} bits
- Stock Buffer: {stock:.2f}
- Liquidez: {liquidity:.2f}
- Umbral de Colapso (θ_max): {theta_max:.2f}

INSTRUCCIONES DE EJECUCIÓN:
1. Usa run_simulation para probar diferentes valores de K, empezando por K0.
2. Ejecuta máximo 5 simulaciones para encontrar estabilidad (colapso < 5%).
3. Si encuentras estabilidad, genera reporte ROBUSTO.
4. Si no encuentras estabilidad tras 5 simulaciones, genera reporte FRÁGIL.
5. Traduce resultados a lenguaje de negocio (urgencias médicas, stocks, riesgo operativo).

HERRAMIENTA DISPONIBLE:
- run_simulation(I, K, theta_max, runs=500): Ejecuta simulación Monte Carlo con parámetros dados.

Sé autónomo pero eficiente. No uses otras herramientas. Genera reporte final cuando tengas evidencia suficiente.
"""

        # 3. HERRAMIENTA ÚNICA: Configurar solo run_simulation
        tools = [types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="run_simulation",
                    description="Ejecuta simulación Monte Carlo para evaluar estabilidad del sistema",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "I": types.Schema(type=types.Type.NUMBER, description="Entropía externa"),
                            "K": types.Schema(type=types.Type.NUMBER, description="Capacidad del sistema"),
                            "theta_max": types.Schema(type=types.Type.NUMBER, description="Umbral de colapso"),
                            "runs": types.Schema(type=types.Type.INTEGER, default=500, description="Número de simulaciones")
                        },
                        required=["I", "K", "theta_max"]
                    )
                )
            ]
        )]

        # 4. AUTOMATIC FUNCTION CALLING: Configurar con disable=False y maximum_remote_calls=5
        config = types.GenerateContentConfig(
            temperature=0.1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=5
            ),
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_level="low"
            ),
            tools=tools
        )

        if self.is_mock_mode:
            # Mock response para pruebas
            final_report = f"""# Auditoría Iso-Entropy - Function Calling Agent

## Resumen Ejecutivo
- **Sistema Analizado:** {volatilidad} volatilidad, {rigidez} rigidez, {colchon} meses colchón
- **Estado Final:** ⚠️ MARGINAL (Mock Mode)
- **Parámetros:** I={I:.2f}, K0={K0:.2f}

## Hallazgos
- Mock: Sistema evaluado con simulaciones limitadas.
- Recomendación: Implementar mejoras operativas.

---
*Generado en Mock Mode*
"""
            return final_report

        # Ejecutar llamada a API con function calling automático
        self.rate_limiter.wait_if_needed()

        # Loop para manejar function calling
        conversation = [system_instruction]
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=conversation,
                config=config
            )

            # Verificar si hay llamadas a funciones
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    function_calls = []
                    text_parts = []

                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls.append(part.function_call)
                        elif hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)

                    # Si hay llamadas a funciones, ejecutarlas
                    if function_calls:
                        function_results = []

                        for call in function_calls:
                            result = self._execute_simulation_call(call, I, theta_max)
                            function_results.append(types.Part(
                                function_response=types.FunctionResponse(
                                    name=call.name,
                                    response=result
                                )
                            ))

                        # Agregar resultados al conversation
                        conversation.append(types.Content(
                            role="user",
                            parts=function_results
                        ))
                        continue

                    # Si no hay llamadas a funciones, es la respuesta final
                    else:
                        break

        # Procesar respuesta final
        final_report = response.text or "No se generó respuesta"

        # Agregar metadatos
        final_report = f"""# 🎯 Auditoría Forense - ISO-ENTROPÍA (Function Calling Agent)

## Contexto de Ejecución
- **Sistema Analizado:** {volatilidad} volatilidad, {rigidez} rigidez, {colchon} meses colchón
- **Experimentos Ejecutados:** {len(self.experiment_log)}
- **Parámetros Físicos:** I={I:.2f}, K0={K0:.2f}, θ_max={theta_max:.2f}

---

## 📋 Reporte Generado por Auditor

{final_report}

---

## 📊 Historial de Experimentos

{self._format_experiment_table()}

---
*Generado por Iso-Entropy Agent v2.3 (Function Calling)*
*Powered by Gemini 3 Flash Preview*
*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        return final_report

    def _execute_simulation_call(self, function_call, I, theta_max):
        """Ejecuta llamada a run_simulation y registra en experiment_log."""
        name = function_call.name
        args = function_call.args if hasattr(function_call, 'args') else {}

        if name == "run_simulation":
            K = args.get("K", 1.0)
            runs = args.get("runs", 500)

            self._log(f"🔬 Ejecutando simulación: I={I:.2f}, K={K:.2f}, θ_max={theta_max:.2f}, runs={runs}")

            result = run_simulation(I, K, theta_max, runs)

            # Registrar en experiment_log
            colapso_pct = result["tasa_de_colapso"]
            collapses_total = result.get("collapses_total", int(colapso_pct * runs))
            runs_total = result.get("runs", runs)
            ub95 = self._calculate_wilson_upper_bound(collapses_total, runs_total)

            self.experiment_log.append({
                "ciclo": len(self.experiment_log) + 1,
                "timestamp": datetime.now().isoformat(),
                "hipotesis": {"I": I, "K": K},
                "parametros_completos": {
                    "theta_max": theta_max,
                    "runs": runs_total
                },
                "resultado": {
                    "tasa_de_colapso": colapso_pct,
                    "upper_ci95": ub95,
                    "collapses_total": collapses_total,
                    "runs": runs_total,
                    "tiempo_promedio_colapso": result.get("tiempo_promedio_colapso", float('inf')),
                    "insolvencia_informacional": I / K if K > 0 else float('inf')
                }
            })

            self._log(f"📊 Resultado: Colapso = {colapso_pct:.1%}, UB95 = {ub95:.1%}")

            return result
        else:
            return {"error": f"Función desconocida: {name}"}

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
        
        table = "| Ciclo | K (bits) | Colapso (%) | UB95 (%) | II | D_e | Estado |\n"
        table += "|-------|----------|-------------|----------|----|-----|--------|\n"

        for exp in self.experiment_log:
            k_val = exp["hipotesis"]["K"]
            collapse = exp["resultado"]["tasa_de_colapso"]
            ub95 = exp["resultado"].get("upper_ci95", collapse)  # Fallback a colapso si no hay UB95
            ii = exp["resultado"].get("insolvencia_informacional", "N/A")
            de = exp["resultado"].get("deuda_entropica_residual", "N/A")
            estado = "✅" if collapse < self.STABILITY_THRESHOLD and ub95 < self.STABILITY_THRESHOLD else "⚠️" if collapse < self.MARGINAL_THRESHOLD else "❌"
            table += f"| {exp.get('ciclo', 'N/A')} | {k_val:.2f} | {collapse:.1%} | {ub95:.1%} | {ii if isinstance(ii, str) else f'{ii:.2f}'} | {de if isinstance(de, str) else f'{de:.2f}'} | {estado} |\n"
        
        return table


# Nueva clase IsoEntropyAgent para single-turn agent con function calling automático
class IsoEntropySingleTurnAgent:
    """
    Agente Iso-Entropy Single-Turn con Function Calling Automático usando Gemini 3 Flash.

    Esta clase implementa un agente autónomo que ejecuta una única llamada a la API
    con function calling automático para analizar sistemas y generar reportes.
    """

    def __init__(self, model_name="gemini-3-flash-preview", api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                raise RuntimeError(f"Error inicializando cliente Gemini: {e}")
        else:
            raise ValueError("GEMINI_API_KEY no encontrada en variables de entorno")

        # Log de telemetría
        self.telemetry_log = []

    def _log_telemetry(self, event: str, data: dict = None):
        """Registra evento en telemetría."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data or {}
        }
        self.telemetry_log.append(entry)
        print(f"[TELEMETRY] {event}: {data}")

    def audit_system(self, user_input: str, volatilidad: str, colchon: int, rigidez: str) -> dict:
        """
        Ejecuta auditoría single-turn con function calling automático.

        Args:
            user_input: Descripción del contexto del usuario
            volatilidad: Nivel de volatilidad ("Baja (Estable)", "Media (Estacional)", "Alta (Caótica)")
            colchon: Meses de colchón financiero
            rigidez: Nivel de rigidez ("Baja (Automatizada)", "Media (Estándar)", "Alta (Manual/Burocrático)")

        Returns:
            dict: {
                "reporte_final": str,
                "telemetria": list,
                "error": str (opcional)
            }
        """
        try:
            self._log_telemetry("audit_start", {
                "user_input": user_input,
                "volatilidad": volatilidad,
                "colchon": colchon,
                "rigidez": rigidez
            })

            # Definir herramientas nativas para function calling
            tools = [types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="ground_inputs",
                        description="Convierte inputs humanos a parámetros físicos",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "volatilidad": types.Schema(type=types.Type.STRING),
                                "rigidez": types.Schema(type=types.Type.STRING),
                                "colchon_meses": types.Schema(type=types.Type.INTEGER)
                            },
                            required=["volatilidad", "rigidez", "colchon_meses"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="apply_hard_rules",
                        description="Aplica reglas físicas absolutas",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "volatilidad": types.Schema(type=types.Type.STRING),
                                "rigidez": types.Schema(type=types.Type.STRING),
                                "colchon_meses": types.Schema(type=types.Type.INTEGER),
                                "params": types.Schema(type=types.Type.OBJECT)
                            },
                            required=["volatilidad", "rigidez", "colchon_meses", "params"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="calculate_collapse_threshold",
                        description="Calcula umbral de colapso θ_max",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "stock_ratio": types.Schema(type=types.Type.NUMBER),
                                "capital_ratio": types.Schema(type=types.Type.NUMBER),
                                "liquidity": types.Schema(type=types.Type.NUMBER)
                            },
                            required=["stock_ratio", "capital_ratio", "liquidity"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="run_simulation",
                        description="Ejecuta simulación Monte Carlo",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "I": types.Schema(type=types.Type.NUMBER),
                                "K": types.Schema(type=types.Type.NUMBER),
                                "theta_max": types.Schema(type=types.Type.NUMBER),
                                "runs": types.Schema(type=types.Type.INTEGER, default=500)
                            },
                            required=["I", "K", "theta_max"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="build_llm_signal",
                        description="Construye señal de telemetría resumida",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "experiment_log": types.Schema(
                                    type=types.Type.ARRAY,
                                    items=types.Schema(type=types.Type.OBJECT)
                                )
                            },
                            required=["experiment_log"]
                        )
                    )
                ]
            )]

            # Prompt maestro
            prompt = f"""
Eres un Auditor de Entropía Autónoma especializado en termodinámica de la información.
Tu tarea es analizar el sistema del usuario, ejecutar simulaciones necesarias ajustando K hasta estabilidad o máximo 5 iteraciones, y generar un reporte final completo con traducción semántica a lenguaje de negocio.

CONTEXTO DEL USUARIO:
{user_input}

CALIBRACIÓN DEL SISTEMA:
- Volatilidad: {volatilidad}
- Rigidez Operativa: {rigidez}
- Colchón Financiero: {colchon} meses

INSTRUCCIONES DE EJECUCIÓN:
1. Analiza el contexto del usuario y calibra parámetros físicos usando ground_inputs
2. Aplica reglas físicas duras con apply_hard_rules
3. Ejecuta simulación base para estado inicial
4. Itera internamente máximo 5 veces ajustando K hasta encontrar estabilidad (colapso < 5%) o insolvencia total
5. Traduce resultados técnicos a urgencias médicas, stocks de inventario, riesgo operativo
6. Genera reporte final completo con diagnóstico, recomendaciones y métricas

HERRAMIENTAS DISPONIBLES:
- ground_inputs: Convierte inputs humanos a parámetros físicos
- apply_hard_rules: Aplica constraints físicos absolutos
- calculate_collapse_threshold: Calcula umbral de colapso θ_max
- run_simulation: Ejecuta simulación Monte Carlo
- build_llm_signal: Construye señal de telemetría resumida

Sé autónomo y ejecuta todas las herramientas necesarias. Cuando tengas suficiente evidencia, genera el reporte final.
"""

            # Configuración para function calling
            config = types.GenerateContentConfig(
                temperature=0.1,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level="low"
                ),
                tools=tools
            )

            self._log_telemetry("api_call_start")

            # Loop para manejar function calling
            conversation = [prompt]
            max_iterations = 10
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                self._log_telemetry("api_call_iteration", {"iteration": iteration})

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=conversation,
                    config=config
                )

                # Verificar si hay llamadas a funciones
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        function_calls = []
                        text_parts = []

                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                function_calls.append(part.function_call)
                            elif hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)

                        # Si hay llamadas a funciones, ejecutarlas
                        if function_calls:
                            self._log_telemetry("function_calls_detected", {"count": len(function_calls)})
                            function_results = []

                            for call in function_calls:
                                result = self._execute_function_call(call)
                                function_results.append(types.Part(
                                    function_response=types.FunctionResponse(
                                        name=call.name,
                                        response=result
                                    )
                                ))

                            # Agregar resultados al conversation
                            conversation.append(types.Content(
                                role="user",
                                parts=function_results
                            ))
                            continue

                        # Si no hay llamadas a funciones, es la respuesta final
                        else:
                            break

            self._log_telemetry("api_call_end", {"response_length": len(response.text) if response.text else 0})

            # Extraer pensamientos
            thoughts = self._extract_thoughts(response)

            # Procesar respuesta final
            final_report = self._process_final_response(response, thoughts)

            self._log_telemetry("audit_complete", {"report_length": len(final_report)})

            return {
                "reporte_final": final_report,
                "telemetria": self.telemetry_log
            }

        except Exception as e:
            error_msg = f"Error en auditoría: {str(e)}"
            self._log_telemetry("audit_error", {"error": error_msg})
            return {
                "reporte_final": f"Error técnico: {error_msg}",
                "telemetria": self.telemetry_log,
                "error": error_msg
            }

    def _extract_thoughts(self, response) -> str:
        """Extrae pensamientos del response de Gemini."""
        try:
            if hasattr(response, 'candidates') and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        if hasattr(part.thought, 'text'):
                            return part.thought.text
                        elif isinstance(part.thought, str):
                            return part.thought
                        elif hasattr(part.thought, '__str__'):
                            return str(part.thought)
            return "Pensamientos no disponibles"
        except Exception as e:
            return f"Error extrayendo pensamientos: {e}"

    def _process_final_response(self, response, thoughts: str) -> str:
        """Procesa la respuesta final y genera reporte completo."""
        base_report = response.text or "No se generó respuesta"

        # Agregar sección de pensamientos si están disponibles
        if thoughts and thoughts != "Pensamientos no disponibles":
            base_report = f"## Pensamientos del Auditor\n\n{thoughts}\n\n---\n\n{base_report}"

        # Agregar metadatos
        final_report = f"""# Auditoría Iso-Entropy - Single-Turn Agent

{base_report}

---
*Generado por IsoEntropySingleTurnAgent*
*Modelo: {self.model_name}*
*Timestamp: {datetime.now().isoformat()}*
"""

        return final_report

    def _execute_function_call(self, function_call):
        """Ejecuta una llamada a función basada en el nombre y argumentos."""
        name = function_call.name
        args = function_call.args if hasattr(function_call, 'args') else {}

        self._log_telemetry("executing_function", {"name": name, "args": args})

        try:
            if name == "ground_inputs":
                result = self._tool_ground_inputs(**args)
            elif name == "apply_hard_rules":
                result = self._tool_apply_hard_rules(**args)
            elif name == "calculate_collapse_threshold":
                result = self._tool_calculate_collapse_threshold(**args)
            elif name == "run_simulation":
                result = self._tool_run_simulation(**args)
            elif name == "build_llm_signal":
                result = self._tool_build_llm_signal(**args)
            else:
                result = {"error": f"Función desconocida: {name}"}

            self._log_telemetry("function_executed", {"name": name, "success": True})
            return result

        except Exception as e:
            error_result = {"error": f"Error ejecutando {name}: {str(e)}"}
            self._log_telemetry("function_executed", {"name": name, "success": False, "error": str(e)})
            return error_result

    # =================================================================
    # HERRAMIENTAS NATIVAS PARA FUNCTION CALLING
    # =================================================================

    def _tool_ground_inputs(self, volatilidad: str, rigidez: str, colchon_meses: int):
        """Herramienta: Convierte inputs humanos a parámetros físicos."""
        self._log_telemetry("tool_call", {"tool": "ground_inputs", "args": locals()})
        result = ground_inputs(volatilidad, rigidez, colchon_meses)
        self._log_telemetry("tool_result", {"tool": "ground_inputs", "result": result})
        return result

    def _tool_apply_hard_rules(self, volatilidad: str, rigidez: str, colchon_meses: int, params: dict):
        """Herramienta: Aplica reglas físicas absolutas."""
        self._log_telemetry("tool_call", {"tool": "apply_hard_rules", "args": locals()})
        try:
            result = apply_hard_rules(
                volatilidad=volatilidad,
                rigidez=rigidez,
                colchon_meses=colchon_meses,
                params=params
            )
            self._log_telemetry("tool_result", {"tool": "apply_hard_rules", "result": result})
            return result
        except HardConstraintViolation as e:
            error_result = {"error": str(e), "action": "TERMINATE"}
            self._log_telemetry("tool_result", {"tool": "apply_hard_rules", "result": error_result})
            return error_result

    def _tool_calculate_collapse_threshold(self, stock_ratio: float, capital_ratio: float, liquidity: float):
        """Herramienta: Calcula umbral de colapso θ_max."""
        self._log_telemetry("tool_call", {"tool": "calculate_collapse_threshold", "args": locals()})
        result = calculate_collapse_threshold(stock_ratio, capital_ratio, liquidity)
        self._log_telemetry("tool_result", {"tool": "calculate_collapse_threshold", "result": result})
        return result

    def _tool_run_simulation(self, I: float, K: float, theta_max: float, runs: int = 500):
        """Herramienta: Ejecuta simulación Monte Carlo."""
        self._log_telemetry("tool_call", {"tool": "run_simulation", "args": locals()})
        result = run_simulation(I, K, theta_max, runs)
        self._log_telemetry("tool_result", {"tool": "run_simulation", "result": result})
        return result

    def _tool_build_llm_signal(self, experiment_log: list):
        """Herramienta: Construye señal de telemetría resumida."""
        self._log_telemetry("tool_call", {"tool": "build_llm_signal", "args": {"experiment_log_length": len(experiment_log)}})
        result = build_llm_signal(experiment_log)
        self._log_telemetry("tool_result", {"tool": "build_llm_signal", "result": result})
        return result