# streamlit_app/paginas/analise_moedas.py
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

# =========================
# Conexão
# =========================
@st.cache_resource(show_spinner=False)
def get_engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        try:
            if "DATABASE_URL" in st.secrets:
                url = st.secrets["DATABASE_URL"]
            elif "db" in st.secrets:
                db = st.secrets["db"]
                host = db.get("host", "localhost")
                port = db.get("port", 5432)
                url = (
                    f"postgresql+psycopg2://{db['user']}:{db['password']}"
                    f"@{host}:{port}/{db['database']}"
                )
        except Exception:
            pass
    if not url:
        st.error("Conexão com o banco não configurada (DATABASE_URL ou secrets).")
        st.stop()
    return create_engine(url, pool_pre_ping=True)

# =========================
# Helper: garantir tabela 'moedas'
# =========================
# id -> (símbolo, nome)
COINS_MAP = {
    1: ("BTC", "Bitcoin"),
    2: ("ETH", "Ethereum"),
    3: ("ADA", "Cardano"),
    4: ("SOL", "Solana"),
}

def ensure_moedas(engine):
    """Cria a tabela 'moedas' se faltar e faz upsert dos registros base."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS moedas(
              id      INTEGER PRIMARY KEY,
              simbolo TEXT,
              nome    TEXT,
              ativo   BOOLEAN DEFAULT TRUE
            );
        """))
        # monta VALUES parametrizado
        values_sql = ",".join(f"(:id{i}, :sym{i}, :name{i}, TRUE)" for i in COINS_MAP)
        params = {}
        for i, (sym, name) in COINS_MAP.items():
            params[f"id{i}"] = i
            params[f"sym{i}"] = sym
            params[f"name{i}"] = name

        conn.execute(
            text(f"""
              INSERT INTO moedas (id, simbolo, nome, ativo) VALUES {values_sql}
              ON CONFLICT (id) DO UPDATE
                SET simbolo=EXCLUDED.simbolo, nome=EXCLUDED.nome, ativo=TRUE;
            """),
            params
        )

# =========================
# Dados
# =========================
@st.cache_data(ttl=60, show_spinner=False)
def listar_moedas(_engine):
    # garante a tabela/população antes de ler
    ensure_moedas(_engine)

    q = text("""
        SELECT
            id,
            CASE
                WHEN COALESCE(simbolo, '') <> '' AND COALESCE(nome, '') <> ''
                    THEN UPPER(simbolo) || ' (' || nome || ')'
                WHEN COALESCE(simbolo, '') <> ''
                    THEN UPPER(simbolo)
                WHEN COALESCE(nome, '') <> ''
                    THEN nome
                ELSE 'Moeda ' || id::text
            END AS label
        FROM moedas
        WHERE COALESCE(ativo, TRUE) = TRUE
        ORDER BY UPPER(simbolo) NULLS LAST, nome NULLS LAST, id;
    """)
    df = pd.read_sql_query(q, _engine)
    if not df.empty:
        return {row["label"]: int(row["id"]) for _, row in df.iterrows()}

    # fallback extremo: se por algum motivo não houver nada, usa COINS_MAP
    return {f"{sym} ({name})": i for i, (sym, name) in COINS_MAP.items()}

@st.cache_data(ttl=60, show_spinner=False)
def carregar_ohlc(_engine, moeda_id: int, dt_ini: pd.Timestamp, dt_fim: pd.Timestamp):
    q = text("""
        SELECT timestamp, open, high, low, close, volume
        FROM precos
        WHERE moeda_id = :m
          AND timestamp BETWEEN :ini AND :fim
        ORDER BY timestamp
    """)
    df = pd.read_sql_query(
        q, _engine,
        params={"m": moeda_id, "ini": dt_ini, "fim": dt_fim},
        dtype={"open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"}
    )
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp","open","high","low","close"])
    return df

def _resample_ohlc(df: pd.DataFrame, regra: str) -> pd.DataFrame:
    if regra == "1h":
        return df
    rule = "4H" if regra == "4h" else "1D"
    dfi = df.set_index("timestamp")
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    out = dfi.resample(rule).agg(agg).dropna(subset=["open","high","low","close"]).reset_index()
    return out

# =========================
# UI
# =========================
def show():
    st.markdown("""
        <style>
            .gradient-title {
               font-size: 85px !important;
                font-weight: 800 !important;
                background: linear-gradient(90deg, #00E1D4 0%, #3A80F6 30%, #C64BD9 50%) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                margin: 0   !important;
             }
            .subtitle { font-size: 17px; color: #a6a6a6; margin-top: 0; }
            .card-box { background: #12151f; border-radius: 16px; padding: 20px; height: 120px; }
            .kpi-title { font-size: 14px; color: #7c7f82; margin-bottom: 4px; }
            .kpi-value { font-size: 28px; font-weight: bold; color: white; margin: 0; }
            .badge { font-size: 14px; margin-top: 4px; }
            .green { color: #31fc94; } .red { color: #ff6f6f; }
            .st-dg { background-color: #0996CA; }
        </style>
        <div>
            <h1 class='gradient-title'>Análise Detalhada por Moeda</h1>
            <p class='subtitle'>Análise técnica por Moedas</p>
        </div>
    """, unsafe_allow_html=True)

    engine = get_engine()

    # Select de moedas (com nomes reais)
    mapa_moedas = listar_moedas(engine)
    if not mapa_moedas:
        st.warning("Nenhuma moeda encontrada na base.")
        return
    moeda_label = st.selectbox("Escolha a moeda", list(mapa_moedas.keys()))
    moeda_id = mapa_moedas[moeda_label]

    # Intervalo
    intervalo = st.radio(
        "Intervalo",
        options=["1h", "4h", "1d"],
        horizontal=True,
        index=0
    )

    # Janela automática por intervalo
    WINDOWS = {"1h": pd.Timedelta(days=8), "4h": pd.Timedelta(days=60), "1d": pd.Timedelta(days=365)}
    dt_fim = pd.Timestamp.now(tz="UTC").floor("H")
    dt_ini = dt_fim - WINDOWS[intervalo]

    df = carregar_ohlc(engine, moeda_id, dt_ini, dt_fim)
    if df.empty:
        st.info("Sem dados no período selecionado.")
        return

    ohlc = _resample_ohlc(df, intervalo)

    # Métricas
    last = ohlc.iloc[-1]
    ts_ref = ohlc["timestamp"].max() - pd.Timedelta(days=1)
    prev = ohlc.loc[ohlc["timestamp"] <= ts_ref, "close"]
    ref_close = float(prev.iloc[-1]) if not prev.empty else float(ohlc["close"].iloc[0])
    var_pct = ((float(last["close"]) / ref_close) - 1.0) * 100 if ref_close else 0.0
    volume_24h = float(df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(hours=24))]["volume"].sum() or 0)
    vol_win = min(10, max(2, len(ohlc)//20))
    volatilidade = (ohlc["close"].pct_change().rolling(vol_win).std().iloc[-1] * 100) if len(ohlc) > vol_win else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"""
        <div class="card-box">
            <div class="kpi-title">{moeda_label}</div>
            <p class="kpi-value">${float(last["close"]):,.2f}</p>
            <p class="badge {'green' if var_pct >= 0 else 'red'}">{var_pct:+.2f}%</p>
        </div>""", unsafe_allow_html=True)
    col2.markdown(f"""
        <div class="card-box">
            <div class="kpi-title">Volume 24h</div>
            <p class="kpi-value">${volume_24h:,.0f}</p>
            <p class="badge green">atualizado</p>
        </div>""", unsafe_allow_html=True)
    col3.markdown(f"""
        <div class="card-box">
            <div class="kpi-title">Volatilidade</div>
            <p class="kpi-value">{volatilidade:.1f}%</p>
            <p class="badge red">estimada</p>
        </div>""", unsafe_allow_html=True)
    col4.markdown(f"""
        <div class="card-box">
            <div class="kpi-title">Preço máx/mín (período)</div>
            <p class="kpi-value">${float(ohlc['high'].max()):,.2f}</p>
            <p class="badge">mín ${float(ohlc['low'].min()):,.2f}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Gráfico
    col_graf, col_ai = st.columns([3, 1])
    with col_graf:
        fig = go.Figure(data=[
            go.Candlestick(
                x=ohlc["timestamp"],
                open=ohlc["open"], high=ohlc["high"],
                low=ohlc["low"], close=ohlc["close"], name="OHLC"
            )
        ])
        fig.update_layout(
            title=f"Gráfico de Velas - {moeda_label} ({intervalo})",
            height=460,
            plot_bgcolor="#0F131A", paper_bgcolor="#0F131A",
            font=dict(color="#cfd8dc"),
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Volume negociado"):
            fig_v = go.Figure([go.Bar(x=ohlc["timestamp"], y=ohlc["volume"], name="Volume")])
            fig_v.update_layout(
                height=220,
                plot_bgcolor="#0F131A", paper_bgcolor="#0F131A",
                font=dict(color="#cfd8dc"),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_v, use_container_width=True)

    with col_ai:
        st.markdown("""
            <div style="background-color: #141923; padding: 25px; border-radius: 15px;">
                <h4 style="color:#9db1cb;">Previsões</h4>
                <p><b>1h:</b> Em breve</p>
                <p><b>24h:</b> Em breve</p>
                <p><b>7d:</b> Em breve</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    show()
