import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import timedelta
import plotly.graph_objects as go

def show():
    st.markdown("""
        <style>
            * { font-family: 'Inter', sans-serif !important; }
            .gradient-title {
                font-size: 55px;
                font-weight: 800;
                background: linear-gradient(90deg, #00E1D4, #3A80F6, #C64BD9);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0;
            }
            .subtitle {
                font-size: 17px;
                color: #a6a6a6;
                margin-top: 0;
            }
            .card-box {
                background: #12151f;
                border-radius: 16px;
                padding: 20px;
                height: 120px;
            }
            .kpi-title {
                font-size: 14px;
                color: #7c7f82;
                margin-bottom: 4px;
            }
            .kpi-value {
                font-size: 28px;
                font-weight: bold;
                color: white;
                margin: 0;
            }
            .badge {
                font-size: 14px;
                margin-top: 4px;
            }
            .green { color: #31fc94; }
            .red { color: #ff6f6f; }
        </style>
        <div>
            <h1 class='gradient-title'>Análise Detalhada por Moeda</h1>
            <p class='subtitle'>Análise técnica avançada com IA para cada criptomoeda</p>
        </div>
    """, unsafe_allow_html=True)

    # Conexão
    engine = create_engine('postgresql+psycopg2://coinsight_user:papabento123@localhost:5432/coinsight')
    moedas_df = pd.read_sql("SELECT id, nome FROM moedas WHERE ativo = TRUE ORDER BY nome", engine)
    moedas_dict = dict(zip(moedas_df["nome"], moedas_df["id"]))

    moeda_nome = st.selectbox("Escolha a moeda", list(moedas_dict.keys()))
    moeda_id = moedas_dict[moeda_nome]

    df = pd.read_sql(f"SELECT timestamp, preco, volume FROM precos WHERE moeda_id = {moeda_id}", engine)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # KPIs
    preco_atual = df["preco"].iloc[-1]
    preco_ontem = df[df["timestamp"] <= df["timestamp"].max() - timedelta(days=1)]["preco"].iloc[-1]
    variacao = ((preco_atual - preco_ontem) / preco_ontem) * 100
    volume_24h = df[df["timestamp"] >= df["timestamp"].max() - timedelta(days=1)]["volume"].sum()
    volatilidade = df["preco"].pct_change().rolling(10).std().iloc[-1] * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container():
            st.markdown(f"""
                <div class="card-box">
                    <div class="kpi-title">{moeda_nome}</div>
                    <p class="kpi-value">${preco_atual:,.0f}</p>
                    <p class="badge {'green' if variacao >= 0 else 'red'}">{variacao:+.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown(f"""
                <div class="card-box">
                    <div class="kpi-title">Volume 24h</div>
                    <p class="kpi-value">${volume_24h/1e9:.1f}B</p>
                    <p class="badge green">+15.2%</p>
                </div>
            """, unsafe_allow_html=True)

    with col3:
        with st.container():
            st.markdown(f"""
                <div class="card-box">
                    <div class="kpi-title">Volatilidade</div>
                    <p class="kpi-value">{volatilidade:.1f}%</p>
                    <p class="badge red">Moderada</p>
                </div>
            """, unsafe_allow_html=True)

    with col4:
        with st.container():
            st.markdown(f"""
                <div class="card-box">
                    <div class="kpi-title">Previsão 24h</div>
                    <p class="kpi-value">${preco_atual * 1.045:,.0f}</p>
                    <p class="badge green">+3.5%</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ========== BOTÕES COM ESTADO ==========
    if "intervalo" not in st.session_state:
        st.session_state["intervalo"] = "1h"

    tempo_map = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",    
    "4h": "4h",     
    "1d": "1D"
}

    btn_cols = st.columns(len(tempo_map))

    for i, (label, rule) in enumerate(tempo_map.items()):
        is_selected = st.session_state["intervalo"] == label
        btn_html = f"""
            <button class="{'interval-button selected' if is_selected else 'interval-button'}"
                    onclick="window.location.href='#{label}'">{label}</button>
        """
        with btn_cols[i]:
            if st.button(label, key=f"interval_{label}"):
                st.session_state["intervalo"] = label

    intervalo = st.session_state["intervalo"]
    resample_rule = tempo_map[intervalo]

    # ========= GRÁFICO =========
    df.set_index("timestamp", inplace=True)
    ohlc = df["preco"].resample(resample_rule).ohlc().dropna().reset_index()

    col_graf, col_ai = st.columns([3, 1])

    with col_graf:
        fig = go.Figure(data=[go.Candlestick(
            x=ohlc["timestamp"],
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
        )])
        fig.update_layout(
            height=460,
            plot_bgcolor="#0F131A",
            paper_bgcolor="#0F131A",
            font=dict(color="#cfd8dc"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_ai:
        st.markdown("""
            <div style="background-color: #141923; padding: 25px; border-radius: 15px;">
                <h4 style="color:#9db1cb;">🤖 Previsões IA</h4>
                <p><b>1h:</b> <span style="color:#31fc94">ALTA</span> • Alvo: <b>$46,200</b> • Confiança: 85%<br>
                   <small>RSI oversold · Volume increase · Support bounce</small></p>
                <p><b>24h:</b> <span style="color:#31fc94">ALTA</span> • Alvo: <b>$47,500</b> • Confiança: 72%<br>
                   <small>Moving average crossover · Social sentiment positivo</small></p>
                <p><b>7d:</b> <span style="color:#f2e56e">NEUTRA</span> • Alvo: <b>$44,000 - $48,000</b> • Confiança: 58%<br>
                   <small>Mixed signals · Awaiting Fed decision</small></p>
            </div>
        """, unsafe_allow_html=True)
