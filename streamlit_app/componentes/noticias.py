import os
import requests
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("GNEWS_API_KEY")

def mostrar_noticias_geopoliticas(max_articles: int = 5):
    st.markdown("### 🌐 Notícias Geopolíticas", unsafe_allow_html=True)

    if not API_KEY:
        st.error("API key da GNews não encontrada. Verifique seu arquivo .env.")
        return

    query = (
        "guerra OR conflito OR sanções OR OTAN OR Rússia OR Ucrânia OR "
        "EUA OR China OR Irã OR Israel OR Hamas OR geopolítica"
    )
    params = {
        "q": query,
        "lang": "pt",
        "max": max_articles,
        # não definir country para buscar global
    }

    try:
        response = requests.get("https://gnews.io/api/v4/search", params={**params, "token": API_KEY})
        data = response.json()

        articles = data.get("articles", [])
        if not articles:
            st.markdown(
                "<div style='background-color:#1e2a3a; border-radius:8px; "
                "padding:10px 16px; margin-bottom:12px; color:#8fa3c9; "
                "font-size:0.9rem'>Nenhuma notícia encontrada no momento.</div>",
                unsafe_allow_html=True
            )
            return

        for art in articles:
            title = art.get("title", "")
            desc = art.get("description", "")
            src = art.get("source", {}).get("name", "")
            pub = art.get("publishedAt", "")
            try:
                pub = datetime.strptime(pub, "%Y-%m-%dT%H:%M:%SZ").strftime("%d/%m/%Y")
            except:
                pass
            url = art.get("url", "#")

            st.markdown(f"""
                <div style='background:#1a1e28; border-radius:10px; padding:14px; 
                            margin-bottom:14px; color:#e6f3f1; font-size:0.9rem'>
                  <b style="font-size:1rem">{title}</b><br>
                  <span style="color:#cbd5e1;">{desc}</span><br><br>
                  <i>{src} • {pub}</i><br>
                  <a href="{url}" target="_blank" style="color:#06b6d4;">🔗 Ver mais</a>
                </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Erro ao carregar notícias: {e}")
