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
    from src.core.agent import IsoEntropyAgent
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
        
        status_container = st.status("🧠 Inicializando Agente Iso-Entropy...", expanded=True)
        col_metrics, col_logs = st.columns([1, 2])
        
        with col_metrics:
            metric_placeholder = st.empty()
        
        with col_logs:
            log_placeholder = st.empty()
            thought_placeholder = st.empty()

        # Variables compartidas para el thread
        shared_state = {
            "logs": [],
            "thoughts": [],
            "reporte": None,
            "error": None,
            "completo": False,
            "ciclo": 0
        }

        def capturar_log(mensaje):
            shared_state["logs"].append(mensaje)
            # Detectar pensamientos en el log (si vienen del print en agent.py)
            if "🧠 PENSAMIENTO" in mensaje:
                # Limpiar un poco el mensaje para la UI
                clean_thought = mensaje.replace("🧠 PENSAMIENTO (Chain-of-Thought):", "").strip()
                shared_state["thoughts"].append(clean_thought)

        # Ejecutar agente en hilo
        def run_audit():
            try:
                agent = IsoEntropyAgent(log_callback=capturar_log, api_key=api_key if api_key else None)
                # Guardamos referencia al agente para sacar telemetría después
                shared_state["agent_ref"] = agent 
                shared_state["reporte"] = agent.audit_system(user_input, volatilidad, colchon, rigidez)
            except Exception as e:
                shared_state["error"] = str(e)
            finally:
                shared_state["completo"] = True

        thread = threading.Thread(target=run_audit, daemon=True)
        thread.start()

        # Loop de actualización de UI
        while not shared_state["completo"]:
            # Actualizar Logs
            if shared_state["logs"]:
                last_logs = shared_state["logs"][-3:] # Mostrar solo los últimos
                log_placeholder.code("\n".join(last_logs), language="text")
                
                # Detectar ciclo para la barra de estado
                for l in reversed(shared_state["logs"]):
                    if "CICLO DE PENSAMIENTO #" in l:
                        try:
                            cycle_num = l.split("#")[1].strip()
                            status_container.update(label=f"🔄 Ejecutando Ciclo {cycle_num} (Simulación + Razonamiento)...", state="running")
                        except: pass
                        break

            # Actualizar Pensamiento (El "Cerebro")
            if shared_state["thoughts"]:
                last_thought = shared_state["thoughts"][-1]
                with thought_placeholder.container():
                    st.markdown(f"""
                    <div style='background-color: #262730; padding: 10px; border-radius: 5px; border-left: 3px solid #9b59b6; font-size: 0.9em;'>
                        <span style='color: #9b59b6; font-weight: bold;'>🧠 Gemini Thinking:</span><br>
                        {last_thought[:300]}...
                    </div>
                    """, unsafe_allow_html=True)

            time.sleep(0.5)

        thread.join()
        status_container.update(label="✅ Auditoría Completada", state="complete", expanded=False)

        # --- RESULTADOS FINALES ---
        if shared_state["error"]:
            st.error(f"❌ Error Crítico: {shared_state['error']}")
        
        elif shared_state["reporte"]:
            agent = shared_state.get("agent_ref")
            
            # 1. DASHBOARD DE MÉTRICAS (KPIs)
            st.divider()
            st.subheader("4. Resultados del Diagnóstico")
            
            # Calcular métricas finales
            if agent and agent.experiment_log:
                last_exp = agent.experiment_log[-1]
                # Manejo seguro de datos comprimidos o normales
                if "resultado" in last_exp:
                    ii = last_exp["resultado"].get("insolvencia_informacional", 0)
                    deuda = last_exp["resultado"].get("deuda_entropica_residual", 0)
                    colapso = last_exp["resultado"].get("tasa_de_colapso", 0)
                else:
                    ii, deuda, colapso = 0, 0, 0 # Fallback

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                
                kpi1.metric(
                    "Insolvencia (I/K)", 
                    f"{ii:.2f}", 
                    delta="-Crítico" if ii > 1 else "Estable", 
                    delta_color="inverse"
                )
                kpi2.metric(
                    "Prob. Colapso", 
                    f"{colapso:.1%}", 
                    delta="-Alto Riesgo" if colapso > 0.1 else "Seguro",
                    delta_color="inverse"
                )
                kpi3.metric(
                    "Deuda Entrópica", 
                    f"{deuda:.2f} bits", 
                    help="Acumulación de desorden no procesado"
                )
                kpi4.metric(
                    "Horizonte", 
                    "6-12 Meses" if colapso < 0.2 else "< 3 Meses",
                    delta="Alerta" if colapso > 0.2 else "Normal",
                    delta_color="inverse"
                )

            # 2. TABS DE DETALLE
            tab_report, tab_telemetry, tab_charts = st.tabs(["📄 Reporte Ejecutivo", "🧠 Lógica del Agente", "📈 Gráficas de Simulación"])

            with tab_report:
                st.markdown(shared_state["reporte"])
                st.download_button(
                    "📥 Descargar PDF/Markdown",
                    shared_state["reporte"],
                    file_name="auditoria_iso_entropia.md"
                )

            with tab_telemetry:
                st.info("Traza completa de razonamiento y decisiones del Agente.")
                for i, thought in enumerate(shared_state["thoughts"]):
                    with st.expander(f"💭 Pensamiento Ciclo {i+1}"):
                        st.write(thought)
                
                with st.expander("🔍 JSON Crudo de Experimentos"):
                    st.json(agent.experiment_log)

            with tab_charts:
                if agent and agent.experiment_log:
                    try:
                        # Filtrar logs válidos (no comprimidos)
                        valid_logs = [e for e in agent.experiment_log if "resultado" in e]
                        if valid_logs:
                            df = pd.DataFrame([
                                {
                                    "Ciclo": str(e["ciclo"]),
                                    "K (Capacidad)": e["hipotesis"]["K"],
                                    "Colapso (%)": e["resultado"]["tasa_de_colapso"] * 100,
                                    "Deuda Entrópica": e["resultado"].get("deuda_entropica_residual", 0)
                                } for e in valid_logs
                            ])
                            
                            st.markdown("#### Evolución de la Estabilidad")
                            fig = px.line(df, x="Ciclo", y="Colapso (%)", markers=True, title="Riesgo de Colapso por Iteración")
                            fig.add_hline(y=5, line_dash="dash", line_color="green", annotation_text="Umbral Seguro")
                            st.plotly_chart(fig, use_container_width=True)

                            col_chart1, col_chart2 = st.columns(2)
                            with col_chart1:
                                fig2 = px.bar(df, x="Ciclo", y="K (Capacidad)", title="Ajustes de Capacidad (K)")
                                st.plotly_chart(fig2, use_container_width=True)
                            with col_chart2:
                                fig3 = px.area(df, x="Ciclo", y="Deuda Entrópica", title="Acumulación de Deuda", color_discrete_sequence=["#FF4B4B"])
                                st.plotly_chart(fig3, use_container_width=True)
                    except Exception as e:
                        st.warning(f"No se pudieron generar gráficas: {e}")

if __name__ == "__main__":
    main()