# streamlit_app/paginas/dashboard.py
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from datetime import timedelta

from componentes.noticias import mostrar_noticias_geopoliticas
from componentes.feed_social import mostrar_feed_social

# ========= conexão =========
@st.cache_resource(show_spinner=False)
def get_engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        url = "postgresql+psycopg2://postgres:postgres@localhost:5432/coinsight"
    return create_engine(url, pool_pre_ping=True)

def _coluna_existe(_engine, tabela: str, coluna: str) -> bool:
    q = text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = :tabela AND column_name = :coluna
        -- opcional: limita ao schema 'public'
        AND (table_schema = current_schema() OR table_schema = 'public')
        LIMIT 1
    """)
    with _engine.connect() as conn:
        return conn.execute(q, {"tabela": tabela, "coluna": coluna}).first() is not None

@st.cache_data(ttl=60, show_spinner=False)
def listar_moedas(_engine):
    try:
        q = text("""
            SELECT
              id,
              CASE
                WHEN COALESCE(simbolo,'') <> '' AND COALESCE(nome,'') <> ''
                  THEN UPPER(simbolo) || ' (' || nome || ')'
                WHEN COALESCE(simbolo,'') <> '' THEN UPPER(simbolo)
                WHEN COALESCE(nome,'') <> '' THEN nome
                ELSE id::text
              END AS label
            FROM moedas
            WHERE COALESCE(ativo, TRUE) = TRUE
            ORDER BY UPPER(simbolo) NULLS LAST, nome NULLS LAST, id
        """)
        df = pd.read_sql_query(q, _engine)
        if not df.empty:
            return df[["id", "label"]]
    except Exception:
        pass

    # fallback: ids que existem em 'precos'
    df = pd.read_sql_query(text("SELECT DISTINCT moeda_id AS id FROM precos ORDER BY 1;"), _engine)
    df["label"] = df["id"].apply(lambda x: f"Moeda {int(x)}")
    return df[["id","label"]]

@st.cache_data(ttl=60, show_spinner=False)
def carregar_series(_engine, moeda_id: int) -> pd.DataFrame:
    # Decide qual coluna usar como "preço"
    tem_close = _coluna_existe(_engine, "precos", "close")
    tem_preco = _coluna_existe(_engine, "precos", "preco")

    if tem_close:
        col_preco = "close"
    elif tem_preco:
        col_preco = "preco"
    else:
        # sem nenhuma das duas, retorna vazio
        return pd.DataFrame(columns=["timestamp", "close"])

    # Seleciona apenas timestamp + a coluna escolhida (apelidada para 'close')
    q = text(f"""
        SELECT timestamp, {col_preco} AS close
        FROM precos
        WHERE moeda_id = :m
        ORDER BY timestamp
    """)
    df = pd.read_sql_query(q, _engine, params={"m": moeda_id})
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "close"])
    return df

def show():
    # ====== título ======
    st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
        <style>
          * { font-family: 'Inter', sans-serif !important; }
          .gradient-title {
            font-size: 85px; font-weight: 800;
            background: linear-gradient(90deg, #00E1D4 0%, #3A80F6 30%, #C64BD9 50%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
          }
          .subtitle { font-size: 20px; color: #C2C4C7; margin-bottom: 20px; }
        </style>
        <div>
          <h1 class='gradient-title'>Crypto Insights AI</h1>
          <p class='subtitle'>Análise inteligente de criptomoedas com IA</p>
        </div>
    """, unsafe_allow_html=True)

    engine = get_engine()

    # ====== moedas ======
    moedas_df = listar_moedas(engine)
    if moedas_df.empty:
        st.warning("Nenhuma moeda encontrada. Rode o ETL para popular a base.")
        return

    # ====== CARDS (primeiras 4) ======
    top = moedas_df.head(4).copy()
    cols_cards = st.columns(len(top))
    for idx, (_, row) in enumerate(top.iterrows()):
        moeda_id = int(row["id"])
        label = str(row["label"])

        df = carregar_series(engine, moeda_id)
        if df.empty:
            preco_atual = 0.0
            variacao = 0.0
        else:
            df = df.sort_values("timestamp")
            preco_atual = float(df["close"].iloc[-1])
            ts_max = df["timestamp"].max()
            ref_ts = ts_max - timedelta(hours=24)
            df_passado = df[df["timestamp"] <= ref_ts]
            if not df_passado.empty:
                preco_passado = float(df_passado.iloc[-1]["close"])
                variacao = ((preco_atual - preco_passado) / preco_passado) * 100 if preco_passado else 0.0
            else:
                variacao = 0.0

        cor_var = "#31fc94" if variacao >= 0 else "#ff6f6f"
        arrow = "▲" if variacao >= 0 else "▼"
        txt_var = f"{arrow} {abs(variacao):.2f}%"

        with cols_cards[idx]:
            st.markdown(f"""
                <div style="background:#1a1e28; border-radius:20px; padding:20px; margin-bottom:14px; min-height:120px; position:relative;">
                    <div style="position:absolute; top:10px; right:15px; font-size:18px; color:{cor_var};">{arrow}</div>
                    <div style="font-size:16px; color:#0996ca; font-weight: 700; text-transform: uppercase; margin-bottom:6px;">
                        <b>{label}</b>
                    </div>
                    <h3 style="margin:0; color:#06b6d4; font-size:1.6rem;">${preco_atual:,.2f}</h3>
                    <p style="color:{cor_var}; margin:0; font-size:0.9rem;">{txt_var}</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 0.5px solid #333;'>", unsafe_allow_html=True)

    # ====== GRÁFICO + sidebar ======
    col_grafico, col_sidebar = st.columns([2, 1])

    with col_grafico:
        label_to_id = dict(zip(moedas_df["label"], moedas_df["id"]))
        moeda_label = st.selectbox("Selecione a moeda", list(label_to_id.keys()), index=0)
        moeda_id = int(label_to_id[moeda_label])

        st.markdown(f"""
            <h4 style='color:#bfc3c6; margin-bottom:0.5rem; display:flex; align-items:center;'>
                 Análise de Preços - {moeda_label}
            </h4>
        """, unsafe_allow_html=True)

        # período (sem JS/CSS bugados)
        periodo = st.radio("Período", options=["24h", "7d", "30d"], horizontal=True, index=1)

        df = carregar_series(engine, moeda_id)
        if df.empty:
            st.warning("Não há dados para essa moeda.")
        else:
            df = df.sort_values("timestamp")
            max_ts = df["timestamp"].max()

            if periodo == "24h":
                min_data = max_ts - timedelta(hours=24)
                tickformat = "%H:%M"; dtick = 3600000  # 1h
            elif periodo == "7d":
                min_data = max_ts - timedelta(days=7)
                tickformat = "%d/%m"; dtick = 86400000  # 1d
            else:
                min_data = max_ts - timedelta(days=30)
                tickformat = "%d/%m"; dtick = 172800000  # 2d

            df_filtrado = df[df["timestamp"] >= min_data].copy()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_filtrado["timestamp"],
                y=df_filtrado["close"],
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
            fig.update_xaxes(tickformat=tickformat, tickmode="linear", dtick=dtick, tickangle=-30)

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("---")
            mostrar_feed_social()

    with col_sidebar:
        st.markdown("---")
        # paginas/dashboard.py
        mostrar_noticias_geopoliticas(
            max_articles=3,
            titulo="📰 Principais notícias cripto",
            escopo="cripto",
            provedores=("gnews",),
            layout_cols=1,
        )


