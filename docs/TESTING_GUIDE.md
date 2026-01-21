# 🧪 Guía de Prueba - Auditoría Concreta

## Objetivo
Validar que el sistema ISO-ENTROPÍA ejecuta correctamente la auditoría concreta con generación de reportes Markdown en la fase `CONCLUDE`.

---

## Flujo de Prueba

### Paso 1: Ejecutar Auditoría
```bash
python app.py
```

### Paso 2: Configurar Parámetros en Streamlit
```
Volatilidad: Alta
Rigidez: Alta
Colchón Financiero: 3 meses
```

### Paso 3: Observar Logs

El agente debería mostrar un flujo como:

```
🚀 INICIANDO AGENTE AUTÓNOMO GEMINI 3 PRO
📊 Calibración: Alta volatilidad, Alta rigidez, 3 meses colchón

============================================================
🧠 CICLO DE PENSAMIENTO #1
🔍 FSM_PHASE: ORIENT
============================================================

[... Ciclos de ORIENT → VALIDATE → STRESS ...]

🏁 FSM ha transicionado a CONCLUDE. Generando reporte final.

📄 GENERANDO REPORTE DE AUDITORÍA FINAL (FASE CONCLUDE)...
```

---

## Puntos de Verificación

### 1. Detección de Fase CONCLUDE ✅
**Señal esperada en logs:**
```
🏁 FSM ha transicionado a CONCLUDE. Generando reporte final.
```

### 2. Llamada LLM en CONCLUDE ✅
**Señal esperada en logs:**
```
📄 GENERANDO REPORTE DE AUDITORÍA FINAL (FASE CONCLUDE)...
```

### 3. Formato Markdown en Reporte ✅
**Estructura esperada en el reporte final:**
```markdown
### [Critical Failure Point]
...contenido cuantitativo...

### [Survival Horizon]
...estimación en ciclos...

### [Actionable Mitigation]
...propuesta concreta...
```

### 4. Telemetría Enriquecida ✅
**Verificar en logs de decisión LLM:**
- `theta_max_range`: Debe mostrar rango de $H(C)$
- `entropy_debt_accumulated`: Debe mostrar valor numérico
- `last_theta_max`: Debe mostrar último valor

### 5. Integración de Reporte ✅
**En la salida final Streamlit:**
- Debe incluir sección "📋 Reporte Generado por Auditor (Gemini 3 Pro)"
- Debe contener las tres secciones: Critical Failure Point, Survival Horizon, Actionable Mitigation

---

## Caso de Prueba 1: Sistema Estable

**Parámetros:**
- Volatilidad: Baja
- Rigidez: Baja
- Colchón: 12 meses

**Expectativa:**
- FSM alcanza CONCLUDE rápidamente
- Critical Failure Point: No hay punto crítico detectado (sistema estable)
- Survival Horizon: Indefinido o muy largo
- Actionable Mitigation: Mantener configuración actual

---

## Caso de Prueba 2: Sistema Frágil

**Parámetros:**
- Volatilidad: Alta
- Rigidez: Alta
- Colchón: 1 mes

**Expectativa:**
- FSM alcanza CONCLUDE después de varios ciclos
- Critical Failure Point: Identifica ciclo y valor de $H(C)$ donde colapso es probable
- Survival Horizon: Número limitado de ciclos
- Actionable Mitigation: Aumentar capacidad K o reducir volatilidad

---

## Caso de Prueba 3: Mock Mode

**Configuración (sin API key):**
```bash
# Sin GEMINI_API_KEY en .env
python app.py
```

**Expectativa:**
- El sistema funciona en modo mock
- Genera reporte predefinido sin llamadas reales al LLM
- Logs muestran "Mock mode" y "Mock: Sistema alcanzó punto crítico de fallo"

---

## Troubleshooting

### Problema: "FSM no alcanza CONCLUDE"
**Solución:**
- Revisar `fsm.py` para asegurar que la FSM transiciona a CONCLUDE
- Verificar que `allow_simulation()` retorna False después de CONCLUDE
- Aumentar MAX_ITERATIONS si es necesario

### Problema: "Reporte Markdown mal formateado"
**Solución:**
- Verificar que el LLM recibe el prompt de CONCLUDE correctamente
- Validar que `prompt_templates.py` tiene el formato Markdown esperado
- Revisar logs para ver la respuesta exacta del LLM

### Problema: "Telemetría no incluye theta_max"
**Solución:**
- Verificar que `calculate_collapse_threshold()` en `physics.py` se ejecuta
- Asegurar que el resultado se guarda en `parametros_completos`
- Revisar `telemetry.py` para validar la extracción

---

## Métricas a Registrar

Después de cada prueba, registrar:

| Métrica | Valor |
|---------|-------|
| Volatilidad | ... |
| Rigidez | ... |
| Colchón (meses) | ... |
| Ciclos ejecutados | ... |
| Tasa de colapso máxima | ... |
| FSM fase final | ... |
| Reporte generado (Sí/No) | ... |
| Formato correcto (Sí/No) | ... |
| Tiempo total (seg) | ... |

---

## Checklist de Validación Final

- [ ] Fase CONCLUDE se activa correctamente
- [ ] LLM genera respuesta en formato Markdown
- [ ] Tres secciones presentes: Critical Failure Point, Survival Horizon, Actionable Mitigation
- [ ] Telemetría incluye theta_max y entropy_debt_accumulated
- [ ] Reporte final integra contenido del LLM
- [ ] Mock mode funciona sin API key
- [ ] Tabla de experimentos se muestra correctamente
- [ ] No hay errores de sintaxis Python
- [ ] Logs son informativos y detallados

---

## Recursos

- **Archivo Principal:** `agent.py` (`audit_system` method)
- **Templates:** `prompt_templates.py` (función `build_prompt_for_phase`)
- **Telemetría:** `telemetry.py` (función `build_llm_signal`)
- **FSM:** `fsm.py` (verificar transiciones a CONCLUDE)
- **Physics:** `physics.py` (verificar `calculate_collapse_threshold`)

---

## Notas

1. **Compressed State:** Si se activa la compresión (> 3 ciclos), la telemetría se simplifica. Esto es normal.
2. **Token Optimization:** El thinking level se mantiene en "low" para optimizar costos API.
3. **Cache:** Los prompts se cachean para evitar duplicados. No afecta la auditoría final.

---

**Última Actualización:** 15 de enero de 2026
