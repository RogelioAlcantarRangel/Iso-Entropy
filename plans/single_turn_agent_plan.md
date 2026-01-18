# Plan de Reconstrucción: Iso-Entropy como Single-Turn Autonomous Agent con Gemini 3

## Objetivo
Transformar Iso-Entropy de un sistema multi-turn con FSM y lógica de control Python a un **Single-Turn Autonomous Agent** que utiliza Gemini 3 con function calling automático, eliminando la lógica de control iterativa de Python.

## Arquitectura Actual vs. Nueva

### Arquitectura Actual (Multi-Turn)
```
Usuario Input → Grounding → Constraints → FSM Loop:
  ORIENT → LLM Decide → Simulate → VALIDATE → LLM Decide → Simulate → STRESS → LLM Decide → Simulate → CONCLUDE → LLM Report
```

### Nueva Arquitectura (Single-Turn)
```
Usuario Input → Gemini 3 (con herramientas) → Function Calls Automáticas → Reporte Final
```

## Componentes a Eliminar/Reemplazar

### Eliminar
- **FSM (Finite State Machine)**: `fsm.py` - Lógica de fases ORIENT/VALIDATE/STRESS/CONCLUDE
- **Loop de Control Python**: El `while` loop en `agent.py` que itera hasta CONCLUDE
- **Decisiones Iterativas**: `_decide_next_step()` que llama al LLM en cada fase
- **Rate Limiter**: Ya no necesario con single-turn
- **Cache de Prompts**: No aplicable

### Reemplazar
- **Agente Multi-Turn** → **Agente Single-Turn** con function calling
- **Prompts por Fase** → **Prompt Maestro** que instruye al LLM a ser autónomo
- **Lógica de Control Python** → **Lógica de Herramientas** manejada por Gemini 3

## Herramientas Nativas para Function Calling

### 1. `ground_inputs`
```python
def ground_inputs(volatilidad: str, rigidez: str, colchon_meses: int) -> Dict[str, float]
```
- Convierte inputs humanos a parámetros físicos (I, K0, stock, liquidity, capital)

### 2. `apply_hard_rules`
```python
def apply_hard_rules(volatilidad: str, rigidez: str, colchon_meses: int, params: Dict[str, float]) -> Dict[str, float]
```
- Aplica constraints físicas antes del razonamiento
- Detecta colapso inevitable
- Ajusta parámetros inválidos

### 3. `calculate_collapse_threshold`
```python
def calculate_collapse_threshold(stock_ratio: float, capital_ratio: float, liquidity: float) -> float
```
- Calcula θ_max = log₂(1 + stock) + log₂(1 + capital) + log₂(1 + liquidity)

### 4. `run_simulation`
```python
def run_simulation(I: float, K: float, theta_max: float, runs: int = 500, time_steps: int = 52, alpha: float = 0.15) -> Dict
```
- Ejecuta simulación Monte Carlo
- Retorna tasa_de_colapso, tiempo_promedio_colapso, etc.

### 5. `build_llm_signal`
```python
def build_llm_signal(experiment_log: List[Dict]) -> Dict
```
- Construye telemetría resumida del historial de experimentos

## Flujo del Nuevo Agente

### 1. Input del Usuario
- user_input: str (descripción del sistema)
- volatilidad: str ("Alta", "Media", "Baja")
- rigidez: str ("Alta", "Media", "Baja")
- colchon: int (meses)

### 2. Llamada Única a Gemini 3
- Prompt maestro que instruye al LLM a:
  - Ser autónomo y tomar todas las decisiones
  - Usar herramientas para investigar el sistema
  - Ejecutar simulaciones necesarias
  - Analizar resultados
  - Generar reporte final completo

### 3. Function Calls Automáticas
El LLM decide qué herramientas llamar:
```
ground_inputs() → apply_hard_rules() → calculate_collapse_threshold() → run_simulation() x N → build_llm_signal() → Reporte
```

### 4. Reporte Final
- Estructura idéntica al actual (Critical Failure Point, Survival Horizon, Actionable Mitigation)
- Basado en análisis autónomo del LLM

## Prompt Maestro

El prompt debe incluir:
- Instrucciones de autonomía
- Descripción de herramientas disponibles
- Objetivo: Investigar fragilidad y generar reporte
- Formato de salida requerido
- Contexto de termodinámica de información

## Beneficios de la Nueva Arquitectura

### Ventajas
- **Más Rápido**: Una sola llamada API vs. múltiples iteraciones
- **Más Simple**: Elimina complejidad del FSM y loop
- **Más Autónomo**: El LLM razona end-to-end
- **Mejor Escalabilidad**: No rate limiting entre llamadas
- **Más Natural**: Simula pensamiento humano continuo

### Desventajas
- **Menos Control**: Python no puede intervenir en decisiones intermedias
- **Más Dependiente del LLM**: Requiere prompt engineering robusto
- **Posible Mayor Costo**: Una llamada más larga vs. múltiples cortas

## Implementación Técnica

### Nuevo Archivo: `single_turn_agent.py`
```python
class SingleTurnIsoEntropyAgent:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        # Definir herramientas para function calling
        
    def audit_system(self, user_input, volatilidad, rigidez, colchon):
        # Crear prompt maestro
        # Configurar function calling
        # Hacer llamada única a Gemini 3
        # Procesar respuesta y function calls
        # Retornar reporte
```

### Configuración de Function Calling
- Usar `types.FunctionDeclaration` para definir cada herramienta
- Configurar `tools` en `GenerateContentConfig`
- Procesar `function_calls` en la respuesta del LLM

## Validación y Testing

### Casos de Prueba
- Replicar casos de uso existentes (CASO_USO_INNOVASTORE.md)
- Verificar que reportes sean equivalentes en calidad
- Medir tiempo de ejecución (debería ser más rápido)
- Validar precisión (±2% en estimaciones)

### Métricas de Éxito
- ✅ Reportes equivalentes al agente actual
- ✅ Tiempo de ejecución < 30 segundos
- ✅ Sin errores de function calling
- ✅ Backward compatibility en API pública

## Roadmap de Implementación

1. **Fase 1**: Crear `single_turn_agent.py` con herramientas básicas
2. **Fase 2**: Implementar prompt maestro y function calling
3. **Fase 3**: Testing con casos existentes
4. **Fase 4**: Optimización y refinamiento
5. **Fase 5**: Documentación y migración

## Riesgos y Mitigaciones

### Riesgo: LLM no usa herramientas correctamente
**Mitigación**: Prompt engineering detallado, ejemplos en prompt, validación de respuestas

### Riesgo: Reportes de menor calidad
**Mitigación**: Incluir instrucciones específicas de formato, comparar con baseline

### Riesgo: Function calls fallidas
**Mitigación**: Manejo de errores robusto, reintentos, fallbacks

## Conclusión

Esta reconstrucción transforma Iso-Entropy en un agente más moderno y eficiente, aprovechando las capacidades avanzadas de Gemini 3 para function calling. Mantiene la precisión científica mientras simplifica significativamente la arquitectura.

¿Apruebas este plan para proceder con la implementación?