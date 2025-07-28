import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import plotly.graph_objects as go
from componentes.noticias import mostrar_noticias_geopoliticas
from componentes.feed_social import mostrar_feed_social



def show():
    
    st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
    """, unsafe_allow_html=True)
    
    # === FONTES E ESTILO ===
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            * {
                font-family: 'Inter', sans-serif !important;
            }
            .gradient-title {
                font-size: 65px !important;
                font-weight: 800 !important;
                background: linear-gradient(90deg, #00E1D4 0%, #3A80F6 30%, #6A5BFF 60%, #C64BD9 100%) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                margin: 0   !important;
            }
            .subtitle {
                font-size: 20px !important;
                color: #C2C4C7;
                margin-bottom: 20px !important;
            }
        </style>
        <div>
            <h1 class='gradient-title'>Crypto Insights AI</h1>
            <p class='subtitle'>Análise inteligente de criptomoedas com IA</p>
        </div>
    """, unsafe_allow_html=True)

    # === CONEXÃO COM BANCO ===
    engine = create_engine('postgresql+psycopg2://coinsight_user:papabento123@localhost:5432/coinsight')

    # === DADOS DAS MOEDAS ===
    moedas = pd.read_sql("SELECT id, nome, simbolo FROM moedas WHERE ativo = TRUE ORDER BY nome", engine)
    moedas_cards = moedas.head(4)

  

    # === CARDS DE MOEDAS ===
    cards = st.columns(len(moedas_cards))
    for i, row in moedas_cards.iterrows():
        nome = row["nome"]
        moeda_id = row["id"]

        df = pd.read_sql(
            f"SELECT preco, timestamp FROM precos WHERE moeda_id = {moeda_id} ORDER BY timestamp", 
            engine
        )

        preco_atual = df["preco"].iloc[-1] if not df.empty else 0

        variacao = 0
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            ts_24h_atras = df["timestamp"].max() - timedelta(hours=24)
            df_passado = df[df["timestamp"] <= ts_24h_atras]

            if not df_passado.empty:
                preco_passado = df_passado.iloc[-1]["preco"]
                variacao = ((preco_atual - preco_passado) / preco_passado) * 100

        cor_var = "#31fc94" if variacao >= 0 else "#ff6f6f"
        arrow = "▲" if variacao >= 0 else "▼"
        txt_var = f"{arrow} {abs(variacao):.2f}%"

        with cards[i]:
            st.markdown(f"""
                <div style="background:#1a1e28; border-radius:20px; padding:20px; margin-bottom:14px; min-height:120px; position:relative;">
                    <div style="position:absolute; top:10px; right:15px; font-size:18px; color:{cor_var};">{arrow}</div>
                    <div style="font-size:20px; color:#0996ca; font-weight: bold; text-transform: uppercase; margin-bottom:6px;">
                        <b>{nome}</b>
                    </div>
                    <h3 style="margin:0; color:#06b6d4; font-size:1.8rem;">${preco_atual:,.2f}</h3>
                    <p style="color:{cor_var}; margin:0; font-size:0.9rem;">{txt_var}</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 0.5px solid #333;'>", unsafe_allow_html=True)

    # === GRÁFICO ===
    col_grafico, col_sidebar = st.columns([2, 1])

    with col_grafico:
        moedas_dict = dict(zip(moedas['nome'], moedas['id']))
        moeda_nome = st.selectbox("Selecione a moeda", moedas['nome'], index=0)
        moeda_id = moedas_dict[moeda_nome]

        st.markdown(f"""
            <h4 style='color:#bfc3c6; margin-bottom:0.5rem; display: flex; align-items:center;'>
                 Análise de Preços - {moeda_nome}
            </h4>
        """, unsafe_allow_html=True)

        # === Estilo visual dos botões personalizados ===
        st.markdown("""
            <style>
                /* Aplica a todos os botões de st.button dentro da área visível */
                div[data-testid="stButton"] > button {
                    border: 1px solid #333 !important;
                    background-color: transparent !important;
                    color: #d3d4d6 !important;
                    border-radius: 10px !important;
                    font-size: 15px !important;
                    transition: all 0.2s ease-in-out;
                }

                div[data-testid="stButton"] > button:hover {
                    background-color: #06b6d4 !important;
                }

                /* Estilo para botão selecionado (azul) */
                div[data-testid="stButton"].selected > button {
                    background-color: #06b6d4 !important;
                    color: #ffffff !important;
                    border: 1px solid #06b6d4 !important;
                    font-weight: 600 !important;
                }
            </style>
        """, unsafe_allow_html=True)


        # === Controlador de estado do filtro ===
        if "periodo" not in st.session_state:
            st.session_state["periodo"] = "7d"

        # === Botões como colunas customizadas ===
        cols = st.columns([1, 1, 1])
        with cols[0]:
            if st.button("24h", key="btn_24h"):
                st.session_state["periodo"] = "24h"
        with cols[1]:
            if st.button("7d", key="btn_7d"):
                st.session_state["periodo"] = "7d"
        with cols[2]:
            if st.button("30d", key="btn_30d"):
                st.session_state["periodo"] = "30d"

        # === Aplica dinamicamente a classe 'selected' com base na escolha ===
        st.markdown(f"""
            <script>
            const buttons = window.parent.document.querySelectorAll('button');
            buttons.forEach(btn => {{
                const parentDiv = btn.closest('div[data-testid="stButton"]');
                if (!parentDiv) return;

                if (btn.innerText === "{st.session_state['periodo']}") {{
                    parentDiv.classList.add("selected");
                }} else {{
                    parentDiv.classList.remove("selected");
                }}
            }});
            </script>
        """, unsafe_allow_html=True)



        # === Valor final do filtro ===
        filtro = st.session_state["periodo"]


        query = f"SELECT timestamp, preco FROM precos WHERE moeda_id = {moeda_id} ORDER BY timestamp"
        df = pd.read_sql(query, engine)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            max_ts = df["timestamp"].max()

            if filtro == "24h":
                min_data = max_ts - timedelta(hours=24)
                tickformat = "%H:%M"
                dtick = 3600000  # 1h
            elif filtro == "7d":
                min_data = max_ts - timedelta(days=7)
                tickformat = "%d/%m"
                dtick = 86400000  # 1d
            else:
                min_data = max_ts - timedelta(days=30)
                tickformat = "%d/%m"
                dtick = 172800000  # 2d

            df_filtrado = df[df["timestamp"] >= min_data].copy()

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_filtrado["timestamp"],
                y=df_filtrado["preco"],
                mode="lines+markers",
                line=dict(color="#06b6d4", width=3),
                marker=dict(size=6),
                hovertemplate="<b>%{x}</b><br>Preço: <b>$%{y:,.2f}</b><extra></extra>"
            ))

            fig.update_layout(
                xaxis=dict(showgrid=False, tickangle=-45),
                yaxis=dict(showgrid=True, gridcolor="#333"),
                plot_bgcolor="#0F131A",
                paper_bgcolor="#0F131A",
                font=dict(color="#cfd8dc"),
                margin=dict(l=10, r=10, t=10, b=40),
                height=350,
            )

            fig.update_xaxes(
                tickformat=tickformat,
                tickmode="linear",
                dtick=dtick,
                tickangle=-30
            )

            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            mostrar_feed_social()

        else:
            st.warning("Não há dados para essa moeda.")


    # === SIDEBAR NOTICIAS GEOPOLITICAS ===
    with col_sidebar:
        mostrar_noticias_geopoliticas()

