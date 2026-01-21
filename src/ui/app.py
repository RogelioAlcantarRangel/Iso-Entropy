# src/ui/app.py - Streamlit UI para Iso-Entropy Auditor Autónomo

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd                 
from datetime import datetime       

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

# Cargar variables de entorno
load_dotenv()

# Configuración de página (DEBE SER LO PRIMERO)
st.set_page_config(
    page_title="Iso-Entropy | Auditoría Autónoma",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/RogelioAlcantarRangel/Iso-Entropy",
        "Report a bug": "https://github.com/RogelioAlcantarRangel/Iso-Entropy/issues",
        "About": "ISO-ENTROPÍA v2.3 - Auditor de Fragilidad Estructural"
    }
)

# ============================================================================
# CSS PERSONALIZADO
# ============================================================================

st.markdown("""
    <style>
    /* Tema dark profesional */
    .main {
        background-color: #0E1117;
        color: #E6EDF3;
    }
    
    /* Títulos */
    h1, h2, h3 {
        color: #FAFAFA;
        font-weight: 700;
    }
    
    /* Métricas */
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41444C;
    }
    
    /* Boxes de éxito */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1c4f2e;
        color: #aaffaa;
        border: 1px solid #2e7d32;
    }
    
    /* Boxes de advertencia */
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #4a3a1a;
        color: #ffcc99;
        border: 1px solid #8b6f47;
    }
    
    /* Divider */
    hr {
        border: 1px solid #41444C;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #262730;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# IMPORTAR AGENTE
# ============================================================================

try:
    root_dir = Path(__file__).parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    
    from src.core.agent import IsoEntropyAgent
except ImportError as e:
    st.error(f"❌ Error de importación: {e}")
    st.stop()

# ============================================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================================

with st.sidebar:
    # Logo y título
    st.image(
        "https://img.icons8.com/fluency/96/system-diagnostic.png",
        width=80
    )
    st.title("ISO-ENTROPÍA")
    st.caption("v2.3 - Auditor Autónomo")
    
    st.markdown("---")

     # --- INPUT DE API KEY RESTAURADO ---
    api_key_input = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        placeholder="Pegar AIzaSy... aquí",
        help="Si se deja vacío, se usará la clave del archivo .env",
        value="" 
    )
    # -----------------------------------
    
    # PARÁMETROS DEL SISTEMA
    st.subheader("⚙️ Parámetros del Sistema")
    
    volatilidad = st.selectbox(
        "🌪️ Volatilidad (Entropía Externa I)",
        options=[
            "Baja (Estable)",
            "Media (Estacional)",
            "Alta (Caótica)"
        ],
        index=1,
        help="Nivel de caos e incertidumbre en el entorno del sistema. Afecta directamente a I (Entropía Externa)."
    )
    
    rigidez = st.selectbox(
        "🧱 Rigidez Operativa (Capacidad K)",
        options=[
            "Baja (Automatizada)",
            "Media (Estándar)",
            "Alta (Manual/Burocrático)"
        ],
        index=1,
        help="Capacidad del sistema para adaptarse y procesar información. Afecta directamente a K (Capacidad de Respuesta)."
    )
    
    colchon = st.slider(
        "💰 Colchón Financiero (Meses)",
        min_value=1,
        max_value=24,
        value=6,
        step=1,
        help="Buffer de tiempo antes del colapso. Define el Umbral de Colapso (θ_max)."
    )
    
    st.markdown("---")

    # CONFIGURACIÓN AVANZADA
    st.subheader("🔧 Opciones Avanzadas")
    
    advanced_options = st.expander("Mostrar opciones avanzadas", expanded=False)
    
    with advanced_options:
        mock_mode = st.checkbox(
            "🎭 Mock Mode (sin API)",
            value=False,
            help="Activa modo simulación sin consumir quota de API. Útil para testing y desarrollo."
        )
        
        verbose = st.checkbox(
            "📝 Modo Verbose",
            value=True,
            help="Muestra logs detallados de cada iteración del agente."
        )
        
        max_iterations = st.slider(
            "🔄 Máximo de Iteraciones",
            min_value=1,
            max_value=20,
            value=10,
            help="Número máximo de iteraciones que el FSM puede ejecutar."
        )
    
    st.markdown("---")
    
    # INFO
    st.info(
        "**Powered by:**\n"
        "- google-genai SDK\n"
        "- Gemini 3 Flash\n"
        "- Termodinámica de Información",
        icon="ℹ️"
    )
    
    st.caption(
        "ISO-ENTROPÍA v2.3 | "
        "[GitHub](https://github.com/RogelioAlcantarRangel/Iso-Entropy) | "
        "[Hackathon](https://gemini3.devpost.com/)"
    )

# ============================================================================
# MAIN CONTENT - HERO SECTION
# ============================================================================

col_logo, col_title = st.columns([1, 5])

with col_logo:
    st.markdown("# ⚡")

with col_title:
    st.markdown("# ISO-ENTROPÍA")
    st.markdown("### Auditoría de Fragilidad Estructural & Detección de Colapso Empresarial")

st.markdown("""
    <div style='background-color: #181a20; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B;'>
        <strong>🤖 Auditor Autónomo:</strong> Utiliza <strong>Termodinámica de la Información</strong> + 
        <strong>Razonamiento de IA (Gemini 3)</strong> para detectar puntos de quiebre invisibles en tu 
        operación <strong>6-12 meses antes</strong> de que ocurran.
    </div>
""", unsafe_allow_html=True)

st.write("")  # Spacer

# ============================================================================
# SECCIÓN 1: DESCRIPCIÓN DEL SISTEMA
# ============================================================================

st.subheader("1️⃣ Describe tu Sistema Operativo")

user_input = st.text_area(
    "Contexto operativo (incluye desafíos, cambios recientes, restricciones):",
    height=150,
    placeholder=(
        "Ejemplo: Hospital privado de alta especialidad. "
        "Demanda de urgencias creció 40% en 12 meses. "
        "Escasez de personal especializado. "
        "Sistemas IT presentan caídas intermitentes. "
        "Márgenes financieros ajustados. "
        "Cualquier interrupción tecnológica genera cascada de efectos."
    ),
    label_visibility="collapsed"
)

st.write("")  # Spacer

# ============================================================================
# SECCIÓN 2: BOTONES DE CONTROL
# ============================================================================

st.subheader("2️⃣ Iniciar Auditoría")

col_btn1, col_btn2, col_spacer = st.columns([2, 1, 2])

with col_btn1:
    start_btn = st.button(
        "🚀 EJECUTAR AUDITORÍA AUTÓNOMA",
        type="primary",
        use_container_width=True,
        help="Inicia la auditoría completa con FSM, simulaciones y análisis."
    )

with col_btn2:
    clear_btn = st.button(
        "🗑️ Limpiar",
        use_container_width=True,
        help="Borra el historial de caché."
    )

if clear_btn:
    st.session_state.clear()
    st.rerun()

st.write("")  # Spacer

# ============================================================================
# EJECUCIÓN DE AUDITORÍA
# ============================================================================

if start_btn:
    # VALIDACIONES DE LLAVE Y LOGICA DE PRIORIDAD
    env_key = os.getenv("GEMINI_API_KEY")
    final_api_key = api_key_input.strip() if api_key_input else env_key
    
    if not user_input.strip():
        st.error("⚠️ Por favor describe tu sistema operativo primero")
        st.stop()
    
    if not final_api_key and not mock_mode:
        st.warning(
            "⚠️ GEMINI_API_KEY no encontrada. "
            "Activando Mock Mode automáticamente para demostración."
        )
        mock_mode = True
    
    # INICIALIZAR AGENTE
    try:
        agent = IsoEntropyAgent(
            api_key=final_api_key if not mock_mode else None,
            mock_mode=mock_mode,
            verbose=verbose,
            max_iterations=max_iterations
        )
    except Exception as e:
        st.error(f"❌ Error inicializando agente: {e}")
        st.stop()
    
    # EJECUTAR AUDITORÍA
    status_placeholder = st.status(
        "🔄 Iniciando auditoría autónoma...",
        expanded=True
    )
    
    with status_placeholder:
        try:
            # Crear contenedor para logs
            logs_container = st.empty()
            
            # Capturar output
            import io
            from contextlib import redirect_stdout
            
            log_capture = io.StringIO()
            
            with redirect_stdout(log_capture):
                result = agent.audit_system(
                    user_input=user_input,
                    volatilidad=volatilidad,
                    rigidez=rigidez,
                    colchon=colchon
                )
            
            # Mostrar logs
            logs = log_capture.getvalue()
            if logs:
                with st.expander("📋 Logs de Ejecución"):
                    st.code(logs, language="text")
            
            status_placeholder.update(
                label="✅ Auditoría completada",
                state="complete"
            )
        
        except Exception as e:
            status_placeholder.update(
                label=f"❌ Error durante auditoría",
                state="error"
            )
            st.error(f"Error: {str(e)}")
            st.stop()
    
    # ========================================================================
    # MOSTRAR RESULTADOS
    # ========================================================================
    
    st.divider()
    st.subheader("3️⃣ Resultados del Análisis")
    st.write("")
    
    # KPIs
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    with col_kpi1:
        st.metric(
            "Experimentos",
            len(agent.experiment_log),
            help="Número de simulaciones ejecutadas por el FSM"
        )
    
    with col_kpi2:
        modo_texto = "🎭 Mock" if mock_mode else "🚀 Production"
        st.metric(
            "Modo",
            modo_texto,
            help="Modo de ejecución"
        )
    
    with col_kpi3:
        fase_actual = agent.fsm.phase_name()
        st.metric(
            "Fase Final",
            fase_actual,
            help="Última fase completada del FSM"
        )
    
    with col_kpi4:
        st.metric(
            "Estado",
            "✅ OK",
            help="Estado de la auditoría"
        )
    
    st.write("")
    
    # REPORTE PRINCIPAL
    st.subheader("📄 Reporte Ejecutivo Completo")
    st.markdown(result)
    
    # DESCARGAR
    st.download_button(
        label="📥 Descargar Reporte (Markdown)",
        data=result,
        file_name=f"auditoria_iso_entropia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )
    
    # INFORMACIÓN TÉCNICA
    with st.expander("🔬 Detalles Técnicos"):
        col_tech1, col_tech2 = st.columns(2)
        
        with col_tech1:
            st.write("**Parámetros Físicos:**")
            st.json({
                "volatilidad": volatilidad,
                "rigidez": rigidez,
                "colchon_meses": colchon,
                "mock_mode": mock_mode
            })
        
        with col_tech2:
            st.write("**Historial FSM:**")
            if agent.experiment_log:
                hist_data = []
                for exp in agent.experiment_log:
                    hist_data.append({
                        "Ciclo": exp['ciclo'],
                        "Fase": exp['fase'],
                        "K": f"{exp['hipotesis']['K']:.2f}",
                        "Colapso": f"{exp['resultado']['tasa_de_colapso']:.1%}"
                    })
                
                df = pd.DataFrame(hist_data)
                # Uso de use_container_width para que se vea bien
                st.dataframe(df, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.caption("**ISO-ENTROPÍA v2.3**")

with col_footer2:
    st.caption("*Powered by Gemini 3 Flash*")

with col_footer3:
    st.caption("[GitHub](https://github.com/RogelioAlcantarRangel/Iso-Entropy)")
