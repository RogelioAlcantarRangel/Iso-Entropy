# src/ui/app.py

import streamlit as st
import time
import pandas as pd
import plotly.express as px
import threading
from datetime import datetime

# Manejo de importación defensivo
try:
    import sys
    from pathlib import Path
    root_dir = Path(__file__).parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.core.agent import IsoEntropySingleTurnAgent
except ImportError as e:
    st.error(f"❌ Error de Importación: {e}")
    st.stop()

# --- CONFIGURACIÓN DE PÁGINA Y ESTILOS ---
def setup_page():
    st.set_page_config(
        page_title="Iso-Entropy | Auditoría Forense AI",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS Personalizado para look "Cyber-Professional"
    st.markdown("""
        <style>
        .main {
            background-color: #0E1117;
        }
        .stMetric {
            background-color: #262730;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #41444C;
        }
        h1, h2, h3 {
            color: #FAFAFA;
        }
        .highlight {
            color: #FF4B4B;
            font-weight: bold;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #1c4f2e;
            color: #aaffaa;
            border: 1px solid #2e7d32;
        }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/system-diagnostic.png", width=60)
        st.title("Configuración")
        st.markdown("---")

        api_key = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Necesaria para el razonamiento del agente."
        )

        st.subheader("Parámetros Físicos")
        
        volatilidad = st.selectbox(
            "🌪️ Volatilidad (Entropía I)",
            ["Baja (Estable)", "Media (Estacional)", "Alta (Caótica)"],
            index=1,
            help="Nivel de caos e incertidumbre en el entorno del sistema."
        )

        rigidez = st.selectbox(
            "🧱 Rigidez (Capacidad K)",
            ["Baja (Automatizada)", "Media (Estándar)", "Alta (Manual/Burocrático)"],
            index=2,
            help="Capacidad del sistema para procesar información y adaptarse."
        )

        colchon = st.slider(
            "💰 Colchón Financiero (Meses)",
            min_value=1, max_value=24, value=6,
            help="Define el Umbral de Colapso (Theta_max). Actúa como batería de energía."
        )

        st.markdown("---")
        st.caption("v2.3 | Powered by Gemini 3 Pro")
        
        return api_key, volatilidad, rigidez, colchon

def main():
    setup_page()
    api_key, volatilidad, rigidez, colchon = render_sidebar()

    # --- HERO SECTION ---
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.write("") # Spacer
        st.write("⚡", unsafe_allow_html=True) # Placeholder icon
    with col_title:
        st.title("ISO-ENTROPÍA")
        st.markdown("### Auditor de Resiliencia Estructural & Insolvencia Informacional")

    st.markdown("""
    <div style='background-color: #181a20; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B;'>
        <strong>🤖 Agente Autónomo:</strong> Este sistema utiliza <strong>Termodinámica de la Información</strong> + <strong>Razonamiento de IA</strong> 
        para detectar puntos de quiebre invisibles en su operación 6-12 meses antes de que ocurran.
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Spacer

    # --- INPUT SECTION ---
    with st.container():
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("1. Contexto Operativo")
            user_input = st.text_area(
                "Describa la operación a auditar:",
                height=150,
                placeholder="Ej: Hospital privado con aumento del 40% en urgencias. Sistemas IT inestables. Personal agotado..."
            )
        
        with col2:
            st.subheader("2. Iniciar Diagnóstico")
            st.info("El agente ejecutará simulaciones Monte Carlo y análisis semántico.")
            start_btn = st.button("🚀 EJECUTAR AUDITORÍA FORENSE", type="primary", use_container_width=True)

    # --- EXECUTION LOGIC ---
    if start_btn:
        if not user_input.strip():
            st.toast("⚠️ Por favor describa la operación primero.", icon="⚠️")
            return

        # Contenedores para actualización en tiempo real
        st.divider()
        st.subheader("3. Análisis en Tiempo Real")

        status_container = st.status("🧠 Inicializando Agente Iso-Entropy Single-Turn...", expanded=True)

        # Variables compartidas para el thread
        shared_state = {
            "reporte": None,
            "error": None,
            "telemetria": [],
            "completo": False
        }

        # Ejecutar agente en hilo
        def run_audit():
            try:
                agent = IsoEntropySingleTurnAgent(api_key=api_key if api_key else None)
                # Guardamos referencia al agente para sacar telemetría después
                shared_state["agent_ref"] = agent
                result = agent.audit_system(user_input, volatilidad, colchon, rigidez)
                if "error" in result:
                    shared_state["error"] = result["error"]
                else:
                    shared_state["reporte"] = result["reporte_final"]
                    shared_state["telemetria"] = result.get("telemetria", [])
            except Exception as e:
                shared_state["error"] = str(e)
            finally:
                shared_state["completo"] = True

        thread = threading.Thread(target=run_audit, daemon=True)
        thread.start()

        # Esperar a que el agente complete (single-turn es rápido)
        while not shared_state["completo"]:
            status_container.update(label="🔄 Ejecutando análisis single-turn...", state="running")
            time.sleep(1)

        thread.join()
        status_container.update(label="✅ Auditoría Completada", state="complete", expanded=False)

        # --- RESULTADOS FINALES ---
        if shared_state["error"]:
            st.error(f"❌ Error Crítico: {shared_state['error']}")

        elif shared_state["reporte"]:
            agent = shared_state.get("agent_ref")

            # 1. DASHBOARD DE MÉTRICAS (KPIs) - Simplificado para single-turn
            st.divider()
            st.subheader("4. Resultados del Diagnóstico")

            # KPIs básicos basados en telemetría
            telemetria = shared_state.get("telemetria", [])
            if telemetria:
                # Contar llamadas a funciones y tiempo de ejecución
                function_calls = sum(1 for t in telemetria if t.get("event") == "function_executed")
                audit_duration = "Completo"  # Simplificado

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)

                kpi1.metric(
                    "Funciones Ejecutadas",
                    f"{function_calls}",
                    help="Número de herramientas utilizadas por el agente"
                )
                kpi2.metric(
                    "Estado",
                    "✅ Completo",
                    delta="Exitoso",
                    delta_color="normal"
                )
                kpi3.metric(
                    "Duración",
                    audit_duration,
                    help="Tiempo total de análisis"
                )
                kpi4.metric(
                    "Modo",
                    "Single-Turn",
                    delta="Optimizado",
                    delta_color="normal"
                )

            # 2. TABS DE DETALLE
            tab_report, tab_telemetry = st.tabs(["📄 Reporte Ejecutivo", "🧠 Telemetría del Agente"])

            with tab_report:
                st.markdown(shared_state["reporte"])
                st.download_button(
                    "📥 Descargar PDF/Markdown",
                    shared_state["reporte"],
                    file_name="auditoria_iso_entropia.md"
                )

            with tab_telemetry:
                st.info("Traza completa de ejecución y telemetría del Agente Single-Turn.")
                telemetria = shared_state.get("telemetria", [])
                if telemetria:
                    for entry in telemetria:
                        with st.expander(f"📊 {entry['event']} - {entry['timestamp'][:19]}"):
                            st.json(entry.get("data", {}))
                else:
                    st.write("No hay datos de telemetría disponibles.")

if __name__ == "__main__":
    main()