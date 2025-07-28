import streamlit as st
from streamlit_option_menu import option_menu
import os
from paginas import (
    dashboard,
    analise_moedas,
    eventos_geopoliticos,
    sentimento_social,
    previsoes_ia,
    alertas,
)

# Configuração da página
st.set_page_config(layout="wide", page_title="CoinSight", page_icon=":crystal_ball:")


st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
""", unsafe_allow_html=True)

# === CSS para centralizar a logo usando display:flex ===
st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
        background-color: #0F131A !important;
    }

    .logo-wrapper {
        display: flex   !important; 
        justify-content: center  !important;
        align-items: center !important;
        padding: 10px 0;
    }

    .logo-wrapper img {
        width: 185px;
    }
    
    .st-emotion-cache-p75nl5 {
        width: 100% !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# Caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
LOGO_PATH = os.path.join(IMG_DIR, "logo-coinsight.png")

# === SIDEBAR ===
with st.sidebar:
    # Usa st.image() dentro de um container centralizado
    st.markdown('<div class="logo-wrapper">', unsafe_allow_html=True)
    st.image(LOGO_PATH, use_container_width=False, width=185)
    st.markdown('</div>', unsafe_allow_html=True)

    selected = option_menu(
        menu_title="Navegação",
        options=[
            "Dashboard",
            "Análise por Moedas",
            "Eventos Geopolíticos",
            "Sentimento Social",
            "Previsões IA",
            "Alertas",
        ],
        icons=[
            "bar-chart-line",
            "currency-bitcoin",
            "globe",
            "chat-dots",
            "funnel",
            "bell"
        ],
        default_index=0,
        menu_icon="cast",
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#0F131A",
                "border-radius": "0px",
            },
            "icon": {"color": "#cfd8dc", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#1e1e1e",
                "color": "#cfd8dc",
            },
            "nav-link-selected": {
                "background-color": "#0996CA",
                "color": "#ffffff",
            },
        }
    )

# === Roteamento ===
page_router = {
    "Dashboard": dashboard,
    "Análise por Moedas": analise_moedas,
    "Eventos Geopolíticos": eventos_geopoliticos,
    "Sentimento Social": sentimento_social,
    "Previsões IA": previsoes_ia,
    "Alertas": alertas,
}

page_router[selected].show()
