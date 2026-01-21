# agent.py - ISO-ENTROPY Autonomous Auditor (google-genai)
import os
import time
import json
import hashlib
import math
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# ✓ IMPORTACIÓN CORRECTA (google-genai)
from google import genai

from .physics import run_simulation, calculate_collapse_threshold
from .grounding import ground_inputs
from .constraints import apply_hard_rules, HardConstraintViolation
from .fsm import IsoEntropyFSM, AgentPhase
from .prompt_templates import build_prompt_for_phase
from .telemetry import build_llm_signal

load_dotenv()


# ============================================================================
# RATE LIMITER - Respeta 5 RPM
# ============================================================================

class RateLimiter:
    """Maneja rate limit de 5 RPM para Gemini."""
    
    def __init__(self, max_rpm: int = 5):
        self.max_rpm = max_rpm
        self.min_interval = 60.0 / max_rpm  # 12 segundos
        self.request_timestamps = []
        self.total_requests = 0
    
    def wait_if_needed(self, verbose: bool = True) -> float:
        """Espera si es necesario para respetar 5 RPM."""
        now = time.time()
        
        # Limpiar timestamps viejos (fuera de ventana de 60 segundos)
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if now - ts < 60.0
        ]
        
        # Si hay 5 requests en la ventana, esperar
        if len(self.request_timestamps) >= self.max_rpm:
            oldest = self.request_timestamps[0]
            wait_time = 60.0 - (now - oldest) + 0.5
            if verbose:
                print(f"⏳ Rate limit (5 RPM): esperando {wait_time:.1f}s")
            time.sleep(wait_time)
            now = time.time()
        
        # Asegurar intervalo mínimo
        if self.request_timestamps:
            last = self.request_timestamps[-1]
            elapsed = now - last
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                if verbose:
                    print(f"⏳ Intervalo mínimo: esperando {wait_time:.1f}s")
                time.sleep(wait_time)
                now = time.time()
        
        # Registrar request
        self.request_timestamps.append(now)
        self.total_requests += 1
        
        return now


# ============================================================================
# AUDITOR AUTÓNOMO PRINCIPAL
# ============================================================================

class IsoEntropyAgent:
    """
    Auditor Autónomo con FSM completa.
    - FSM: ORIENT → VALIDATE → STRESS → CONCLUDE
    - Loops inteligentes ajustando K
    - Simulación Monte Carlo (500 runs)
    - Function calling a Gemini
    - Rate limit respetado
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        mock_mode: bool = False,
        verbose: bool = True,
        max_iterations: int = 10
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.mock_mode = mock_mode or os.getenv("ISO_MOCK_MODE", "false").lower() == "true"
        self.verbose = verbose
        self.max_iterations = max_iterations
        
        if not self.mock_mode and not self.api_key:
            raise ValueError("❌ GEMINI_API_KEY no encontrada")
        
        if not self.mock_mode:
            # ✓ INICIALIZACIÓN CORRECTA (google-genai)
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
        
        # Estado del agente
        self.fsm = IsoEntropyFSM()
        self.experiment_log: List[Dict[str, Any]] = []
        self.rate_limiter = RateLimiter(max_rpm=5)
        self.cache = {}
    
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def _get_cache_key(self, user_input: str, volatilidad: str, rigidez: str) -> str:
        """Clave de cache."""
        key = f"{user_input}|{volatilidad}|{rigidez}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _calculate_wilson_upper_bound(self, collapses: int, runs: int) -> float:
        """Calcula límite superior intervalo Wilson (95%)."""
        if runs == 0:
            return 1.0
        
        z = 1.96  # 95% confidence
        phat = collapses / runs
        denom = 1 + (z**2 / runs)
        centre = phat + (z**2 / (2 * runs))
        adj = z * math.sqrt((phat * (1 - phat) / runs) + (z**2 / (4 * runs**2)))
        upper = (centre + adj) / denom
        
        return min(1.0, upper)
    
    # ========================================================================
    # PASO 1: GROUND INPUTS (Cálculo local)
    # ========================================================================
    
    def _ground_inputs_and_validate(
        self,
        user_input: str,
        volatilidad: str,
        rigidez: str,
        colchon: int
    ) -> Dict[str, float]:
        """Calcula parámetros localmente y aplica hard rules."""
        
        self._log("\n" + "="*70)
        self._log("🚀 AUDITORÍA AUTÓNOMA ISO-ENTROPY")
        self._log("="*70)
        
        # 1. Ground inputs
        params = ground_inputs(volatilidad, rigidez, colchon)
        self._log(f"📊 Parámetros iniciales: I={params['I']:.2f}, K={params['K0']:.2f}")
        
        # 2. Calcular theta_max
        theta_max = calculate_collapse_threshold(
            params['stock'],
            params['capital'],
            params['liquidity']
        )
        params['theta_max'] = theta_max
        self._log(f"📊 Umbral colapso: θ_max={theta_max:.2f}")
        
        # 3. Aplicar hard rules
        try:
            apply_hard_rules(
                volatilidad=volatilidad,
                rigidez=rigidez,
                colchon_meses=colchon,
                params=params
            )
            self._log("✅ Hard rules aplicadas")
        except HardConstraintViolation as e:
            self._log(f"🚫 Violación de constraint: {e}")
            raise
        
        return params
    
    # ========================================================================
    # PASO 2: LOOP PRINCIPAL CON FSM
    # ========================================================================
    
    def audit_system(
        self,
        user_input: str,
        volatilidad: str,
        rigidez: str,
        colchon: int
    ) -> str:
        """
        Auditoría completa con FSM y loops.
        """
        
        # Verificar cache
        cache_key = self._get_cache_key(user_input, volatilidad, rigidez)
        if cache_key in self.cache:
            self._log("✅ Reporte obtenido del cache")
            return self.cache[cache_key]
        
        # Ground inputs
        physical_params = self._ground_inputs_and_validate(
            user_input, volatilidad, rigidez, colchon
        )
        
        I = physical_params['I']
        K_base = physical_params['K0']
        stock = physical_params['stock']
        liquidity = physical_params['liquidity']
        capital = physical_params['capital']
        theta_max = physical_params['theta_max']
        
        # Mock mode
        if self.mock_mode:
            self._log("🎭 MOCK MODE")
            report = self._generate_mock_report(
                user_input, I, K_base, theta_max, stock, liquidity, capital
            )
            self.cache[cache_key] = report
            return report
        
        # ====================================================================
        # LOOP PRINCIPAL: ORIENT → VALIDATE → STRESS → CONCLUDE
        # ====================================================================
        
        current_K = K_base
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            self._log(f"\n📍 Iteración {iteration}/{self.max_iterations} - Fase: {self.fsm.phase_name()}")
            
            # 1. Ejecutar simulación
            self._log(f"🔬 Simulando: I={I:.2f}, K={current_K:.2f}, θ_max={theta_max:.2f}")
            
            sim_result = run_simulation(I, current_K, theta_max, runs=500)
            collapse_rate = sim_result['tasa_de_colapso']
            collapses = sim_result.get('collapses_total', int(collapse_rate * 500))
            ub95 = self._calculate_wilson_upper_bound(collapses, 500)
            
            self._log(f"📊 Resultado: Colapso={collapse_rate:.1%}, UB95={ub95:.1%}")
            
            # 2. Registrar experimento
            self.experiment_log.append({
                'ciclo': iteration,
                'fase': self.fsm.phase_name(),
                'hipotesis': {'I': I, 'K': current_K},
                'resultado': {
                    'tasa_de_colapso': collapse_rate,
                    'upper_ci95': ub95,
                    'collapses_total': collapses,
                    'runs': 500
                }
            })
            
            # 3. Actualizar FSM
            self.fsm.update(collapse_rate, ub95)
            self._log(f"🔄 FSM actualizada → {self.fsm.phase_name()}")
            
            # 4. Decisión basada en fase
            if self.fsm.phase == AgentPhase.CONCLUDE:
                self._log("✅ Pasamos a CONCLUDE - Generando reporte final")
                break
            
            elif self.fsm.phase == AgentPhase.ORIENT:
                # Ajustar K para encontrar estabilidad
                if collapse_rate < 0.05:
                    self._log("✅ Estabilidad encontrada en ORIENT")
                    current_K = current_K  # Mantener K
                else:
                    # Incrementar K
                    delta_k = 0.2 if collapse_rate > 0.5 else 0.1
                    current_K = min(current_K + delta_k, 10.0)
                    self._log(f"📈 Incrementando K a {current_K:.2f}")
            
            elif self.fsm.phase == AgentPhase.VALIDATE:
                # Confirmar reproducibilidad
                self._log("✓ En fase VALIDATE")
            
            elif self.fsm.phase == AgentPhase.STRESS:
                # Mantener K constante
                self._log("⚠️ En fase STRESS (K constante)")
        
        # ====================================================================
        # GENERAR REPORTE FINAL CON GEMINI
        # ====================================================================
        
        self._log("\n📝 Generando reporte final con Gemini...")
        
        # Construir prompt maestro con historial de experimentos
        llm_signal = build_llm_signal(self.experiment_log)
        prompt = build_prompt_for_phase(
            phase=AgentPhase.CONCLUDE,  # Forzar formato de reporte ejecutivo
            phase_reasoning=self.fsm.phase_reasoning(),
            system_description=user_input,
            llm_signal=llm_signal
        )
        
        # Agregar instrucciones finales
        final_prompt = f"""{prompt}

HISTORIAL DE EXPERIMENTOS REALIZADOS:
{json.dumps(self.experiment_log, indent=2)}

PARÁMETROS FINALES DEL SISTEMA:
- Entropía Externa (I): {I:.2f} bits
- Capacidad Óptima (K): {current_K:.2f} bits
- Ratio I/K: {I/current_K:.2f}
- Umbral Colapso: {theta_max:.2f} bits
- Stock Buffer: {stock:.2f} meses
- Liquidez: {liquidity:.2f}

GENERA UN REPORTE EJECUTIVO COMPLETO EN FORMATO MARKDOWN CON LA SIGUIENTE ESTRUCTURA EXACTA:

### Reporte de Auditoría Forense: [Nombre del Sistema]

**A la atención del Director General:**

[Introducción breve sobre el propósito de la auditoría]

---

### 1. Diagnóstico de Insolvencia Informacional
[Análisis detallado del ratio I/K y su significado en términos de negocio]

### 2. Punto Crítico de Fallo
[Descripción del umbral de colapso y vulnerabilidades identificadas]

### 3. Horizonte de Supervivencia
[Estimación temporal de estabilidad bajo condiciones actuales]

### 4. Acciones de Mitigación Concretas
[Lista numerada de 3 acciones específicas y accionables]

---

**Dictamen Final:** [Conclusión ejecutiva]

---

IMPORTANTE: Usa un tono profesional, explica términos técnicos en lenguaje de negocio, y asegura que el reporte sea completo y accionable.
"""
        
        # Hacer llamada a Gemini
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=final_prompt
            )
            
            report = f"""# 🎯 Auditoría Forense - ISO-ENTROPÍA

## 📊 Contexto de Ejecución
- **Sistema Analizado:** {volatilidad} volatilidad, {rigidez} rigidez, {colchon} meses colchón
- **Experimentos Realizados:** {len(self.experiment_log)}
- **Parámetros Finales:** I={I:.2f}, K={current_K:.2f}, θ_max={theta_max:.2f}
- **Fase Final:** {self.fsm.phase_name()}

---

{response.text}

---

## 📈 Historial de Experimentos

| Ciclo | Fase | K (bits) | Colapso (%) | UB95 (%) |
|-------|------|----------|-------------|----------|
"""
            
            for exp in self.experiment_log:
                k = exp['hipotesis']['K']
                colapso = exp['resultado']['tasa_de_colapso']
                ub = exp['resultado']['upper_ci95']
                fase = exp['fase']
                report += f"| {exp['ciclo']} | {fase} | {k:.2f} | {colapso:.1%} | {ub:.1%} |\n"
            
            report += f"""
---
*Generado por Iso-Entropy Agent v2.3*
*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            # Cachear resultado
            self.cache[cache_key] = report
            
            self._log("✅ Auditoría completada exitosamente")
            return report
        
        except Exception as e:
            error_str = str(e)
            self._log(f"❌ Error en Gemini: {error_str[:100]}")
            
            # Fallback a mock si quota agotada
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                self._log("💾 Quota agotada. Generando mock report...")
                report = self._generate_mock_report(
                    user_input, I, current_K, theta_max, stock, liquidity, capital
                )
                self.cache[cache_key] = report
                return report
            else:
                raise
    
    # ========================================================================
    # GENERADOR DE MOCK REPORT
    # ========================================================================
    
    def _generate_mock_report(
        self,
        user_input: str,
        I: float,
        K: float,
        theta_max: float,
        stock: float,
        liquidity: float,
        capital: float
    ) -> str:
        """Genera reporte mock cuando API no disponible."""
        
        ratio = I / K if K > 0 else float('inf')
        
        if ratio > 5:
            estado = "🔴 CRÍTICO"
            diagnostico = f"Sistema en colapso informacional. I/K = {ratio:.2f}"
            horizonte = "7-14 días"
        elif ratio > 2:
            estado = "🟠 MARGINAL"
            diagnostico = f"Sistema frágil. I/K = {ratio:.2f}"
            horizonte = "30-60 días"
        else:
            estado = "🟢 ESTABLE"
            diagnostico = f"Sistema robusto. I/K = {ratio:.2f}"
            horizonte = "6+ meses"
        
        return f"""# 🎯 Auditoría Forense - ISO-ENTROPÍA

**Estado: {estado}**

## 📋 Resumen Ejecutivo
Sistema bajo análisis con parámetros: I={I:.2f} bits (Entropía), K={K:.2f} bits (Capacidad).

## 🔍 Diagnóstico de Insolvencia Informacional

**Ratio I/K: {ratio:.2f}**

{diagnostico}

## ⚠️ Punto Crítico de Fallo Estructural

El sistema colapsa cuando deuda de entropía ≥ {theta_max:.2f} bits.

Factores limitantes:
- Entropía externa (I): {I:.2f} bits
- Capacidad de respuesta (K): {K:.2f} bits
- Buffer disponible (Stock): {stock:.2f} meses
- Liquidez: {liquidity:.2f}

## ⏱️ Horizonte de Supervivencia

**{horizonte}** sin intervención correctiva.

## 🛡️ Mitigación Estratégica

### Acción 1: Aumentar Capacidad (K)
- Automatizar procesos manuales
- Timeline: 4-6 semanas
- Inversión: $50K-150K
- Impacto: Reducir I/K en 20-30%

### Acción 2: Reducir Volatilidad (I)
- Diversificar ingresos/servicios
- Timeline: 2-3 meses
- Inversión: $100K-300K
- Impacto: Estabilizar mercado 15-25%

### Acción 3: Fortalecer Buffer
- Línea de crédito emergencia
- Timeline: Inmediato (2-3 semanas)
- Inversión: Bajo (0% si no utilizada)
- Impacto: +60% horizonte supervivencia

---
*Generado en Mock Mode*
*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""