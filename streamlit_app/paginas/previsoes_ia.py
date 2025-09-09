# streamlit_app/paginas/previsoes_ia.py
import os
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load

# =========================
# Paleta e estilos
# =========================
COLOR_REAL = "#00E1D4"                  # série real
COLOR_PREV = "#3A80F6"                  # previsão (média ensemble)
COLOR_BAND = "rgba(58,128,246,0.18)"    # banda 90%
PAPER_BG   = "#0F131A"
PLOT_BG    = "#0F131A"
FONT_COLOR = "#cfd8dc"
CARD_BG    = "#141923"
BORDER     = "#1f2633"

def _style_fig(fig, title_text: str, height: int = 380):
    """Aplica layout padrão com margens generosas e legenda DENTRO do gráfico (embaixo)."""
    fig.update_layout(
        height=height,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR),
        # margens aumentadas para não cortar título/elementos
        margin=dict(l=6, r=6, t=70, b=60),
        title=dict(
            text=title_text,
            x=0.01, y=0.94, xanchor="left", yanchor="top",
            pad=dict(t=6, b=0, l=0, r=0)
        ),
        # legenda horizontal DENTRO do gráfico (na base)
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", yanchor="top")
    )
    return fig

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
                url = f"postgresql+psycopg2://{db['user']}:{db['password']}@{host}:{port}/{db['database']}"
        except Exception:
            pass
    if not url:
        st.error("Conexão com o banco não configurada (DATABASE_URL ou secrets).")
        st.stop()
    return create_engine(url, pool_pre_ping=True)

# =========================
# Utilitários de dados
# =========================
def _coluna_existe(_engine, tabela: str, coluna: str) -> bool:
    q = text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = :tabela AND column_name = :coluna
        AND (table_schema = current_schema() OR table_schema = 'public')
        LIMIT 1
    """)
    with _engine.connect() as conn:
        return conn.execute(q, {"tabela": tabela, "coluna": coluna}).first() is not None

@st.cache_data(ttl=60, show_spinner=False)
def listar_moedas(_engine):
    try:
        df = pd.read_sql_query(text("""
            SELECT id,
                   CASE
                     WHEN COALESCE(simbolo,'')<>'' AND COALESCE(nome,'')<>'' THEN UPPER(simbolo)||' ('||nome||')'
                     WHEN COALESCE(simbolo,'')<>'' THEN UPPER(simbolo)
                     WHEN COALESCE(nome,'')<>'' THEN nome
                     ELSE id::text
                   END AS label
            FROM moedas
            WHERE COALESCE(ativo, TRUE)=TRUE
            ORDER BY UPPER(simbolo) NULLS LAST, nome NULLS LAST, id
        """), _engine)
        if not df.empty:
            return {row["label"]: int(row["id"]) for _, row in df.iterrows()}
    except Exception:
        pass
    df = pd.read_sql_query(text("SELECT DISTINCT moeda_id AS id FROM precos ORDER BY 1;"), _engine)
    return {f"Moeda {int(i)}": int(i) for i in df["id"].tolist()}

@st.cache_data(ttl=60, show_spinner=False)
def carregar_serie(_engine, moeda_id: int) -> pd.Series:
    tem_close = _coluna_existe(_engine, "precos", "close")
    tem_preco = _coluna_existe(_engine, "precos", "preco")
    if not tem_close and not tem_preco:
        return pd.Series(dtype="float64")
    col_preco = "close" if tem_close else "preco"
    df = pd.read_sql_query(text(f"""
        SELECT timestamp, {col_preco} AS close
        FROM precos WHERE moeda_id=:m ORDER BY timestamp
    """), _engine, params={"m": moeda_id})
    if df.empty:
        return pd.Series(dtype="float64")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp","close"]).set_index("timestamp").sort_index()
    return df["close"].astype("float64")

# =========================
# Engenharia de atributos
# =========================
def make_features(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    # lags 1..24
    for L in range(1, 25):
        df[f"lag_{L}"] = df["y"].shift(L)
    # médias e volatilidade
    for W in [3, 6, 12, 24]:
        df[f"ma_{W}"]  = df["y"].rolling(W).mean()
        df[f"vol_{W}"] = df["y"].pct_change().rolling(W).std()
    return df.dropna()

def time_split(X, y, test_size=0.2):
    n = len(X); n_test = max(1, int(n*test_size))
    return X.iloc[:-n_test], X.iloc[-n_test:], y.iloc[:-n_test], y.iloc[-n_test:]

def train_rf(X_train, y_train, n_estimators=400):
    model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=2, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model

# =========================
# Intervalo de previsão (quantis do ensemble)
# =========================
def rf_predict_with_interval(model, X, alpha=0.10):
    preds = np.column_stack([t.predict(X) for t in model.estimators_])  # (n_amostras, n_arvores)
    yhat  = preds.mean(axis=1)
    low   = np.quantile(preds, alpha/2, axis=1)
    high  = np.quantile(preds, 1 - alpha/2, axis=1)
    return yhat, low, high

def forecast_recursive_with_pi(model, history: pd.Series, steps: int) -> pd.DataFrame:
    s = history.copy()
    idx, yhat_list, low_list, high_list = [], [], [], []
    for _ in range(steps):
        feats = make_features(s).drop(columns=["y"])
        x_last = feats.iloc[-1].values.reshape(1, -1)
        yhat, low, high = rf_predict_with_interval(model, x_last, alpha=0.10)
        yhat, low, high = float(yhat[0]), float(low[0]), float(high[0])
        next_ts = s.index[-1] + pd.Timedelta(hours=1)
        s.loc[next_ts] = yhat
        idx.append(next_ts); yhat_list.append(yhat); low_list.append(low); high_list.append(high)
    return pd.DataFrame({"prev": yhat_list, "low": low_list, "high": high_list}, index=idx)

# =========================
# Persistência 'previsoes'
# =========================
def ensure_schema_previsoes(engine):
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS previsoes(
          id SERIAL PRIMARY KEY,
          moeda_id     INT NOT NULL,
          trained_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          horizonte_h  INT NOT NULL,
          ts_previsto  TIMESTAMPTZ NOT NULL,
          valor        DOUBLE PRECISION,
          low          DOUBLE PRECISION,
          high         DOUBLE PRECISION,
          rmse         DOUBLE PRECISION,
          mae          DOUBLE PRECISION,
          r2           DOUBLE PRECISION
        );
        CREATE INDEX IF NOT EXISTS ix_prev_moeda_h on previsoes(moeda_id, horizonte_h, ts_previsto DESC);
        """))

def salvar_previsao(engine, moeda_id: int, horizonte_h: int, fc: pd.DataFrame, rmse: float, mae: float, r2: float):
    if fc.empty: return 0
    payload = []
    for ts, row in fc.iterrows():
        payload.append({
            "moeda_id": moeda_id,
            "horizonte_h": int(horizonte_h),
            "ts_previsto": pd.Timestamp(ts).tz_convert("UTC"),
            "valor": float(row["prev"]),
            "low": float(row["low"]),
            "high": float(row["high"]),
            "rmse": float(rmse), "mae": float(mae), "r2": float(r2),
        })
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO previsoes (moeda_id, horizonte_h, ts_previsto, valor, low, high, rmse, mae, r2)
            VALUES (:moeda_id, :horizonte_h, :ts_previsto, :valor, :low, :high, :rmse, :mae, :r2)
        """), payload)
    return len(payload)

@st.cache_data(ttl=60, show_spinner=False)
def listar_previsoes_recentes(_engine, moeda_id: int, limit: int = 20):
    q = text("""
        SELECT ts_previsto, horizonte_h, valor, low, high, rmse, mae, r2
        FROM previsoes
        WHERE moeda_id=:m
        ORDER BY ts_previsto DESC
        LIMIT :n
    """)
    return pd.read_sql_query(q, _engine, params={"m": moeda_id, "n": limit})

# =========================
# UI principal
# =========================
def show():
    st.markdown("""
        <div style="margin:4px 0 16px 0;">
          <h1 style="margin:0;
              background: linear-gradient(90deg, #00E1D4, #3A80F6, #C64BD9);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent;
              font-weight:800; font-size:48px;">Previsões IA</h1>
          <p style="color:#9db1cb;margin:6px 0 0 0;font-size:16px;">
            Modelos preditivos a partir do histórico de preços (RF baseline)
          </p>
        </div>
    """, unsafe_allow_html=True)

    engine = get_engine()
    ensure_schema_previsoes(engine)

    mapa = listar_moedas(engine)
    if not mapa:
        st.info("Nenhuma moeda na base. Rode o ETL.")
        return

    left, right = st.columns([3, 2])

    # -------- Coluna esquerda: treino + avaliação --------
    with left:
        moeda_label = st.selectbox("Moeda", list(mapa.keys()))
        moeda_id = mapa[moeda_label]
        serie = carregar_serie(engine, moeda_id)
        if serie.empty:
            st.warning("Sem dados para essa moeda (tabela 'precos').")
            return

        with st.expander("Configurações de Treino"):
            max_rows = len(serie)
            hrs = st.slider("Usar últimas horas para treino", min_value=500, max_value=min(5000, max_rows), value=min(2160, max_rows))
            n_estimators = st.slider("n_estimators (RandomForest)", 100, 1000, 400, step=50)
        serie = serie.iloc[-hrs:].copy()

        Xy = make_features(serie)
        if Xy.empty or len(Xy) < 100:
            st.warning("Poucos dados após a engenharia de atributos. Aumente a janela.")
            return
        X, y = Xy.drop(columns=["y"]), Xy["y"]
        X_tr, X_te, y_tr, y_te = time_split(X, y, test_size=0.2)

        mdl_dir = os.getenv("MODEL_DIR", os.path.join(os.getcwd(), "models")); os.makedirs(mdl_dir, exist_ok=True)
        mdl_path = os.path.join(mdl_dir, f"rf_moeda_{moeda_id}.joblib")
        retrain = st.button("🔁 Treinar/Re-treinar Modelo")
        if retrain or not os.path.isfile(mdl_path):
            with st.spinner("Treinando modelo..."):
                model = train_rf(X_tr, y_tr, n_estimators=n_estimators)
                dump(model, mdl_path)
        else:
            model = load(mdl_path)

        y_pred = pd.Series(model.predict(X_te), index=y_te.index)
        rmse = math.sqrt(mean_squared_error(y_te, y_pred))
        mae  = mean_absolute_error(y_te, y_pred)
        r2   = r2_score(y_te, y_pred)
        ref_std = max(1e-9, y_te.std())
        conf = 1 / (1 + (rmse / ref_std))   # índice heurístico 0..1

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_te.index, y=y_te.values, name="Real", mode="lines", line=dict(color=COLOR_REAL)))
        fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred.values, name="Previsto", mode="lines", line=dict(color=COLOR_PREV)))
        fig = _style_fig(fig, f"Previsões vs Realidade — {moeda_label}", height=420)
        st.plotly_chart(fig, use_container_width=True)

    # -------- Coluna direita: forecast + salvar --------
    with right:
        horizonte = st.selectbox("Horizonte de Previsão", ["1h","24h","7d"], index=1)
        steps = {"1h":1, "24h":24, "7d":168}[horizonte]

        series_hist = serie.copy()
        fc = forecast_recursive_with_pi(model, series_hist, steps=steps)
        prox_valor = float(fc["prev"].iloc[-1])

        ultimo = float(series_hist.iloc[-1]); delta = prox_valor - ultimo
        pct = (delta/ultimo)*100 if ultimo else 0
        tend = "↗︎ Alta" if delta>0 else ("↘︎ Baixa" if delta<0 else "→ Lateral")
        tend_color = "#37e39f" if delta>0 else ("#ff6b6b" if delta<0 else "#9db1cb")
        st.markdown(f"""
        <div style="background:{CARD_BG};padding:18px;border-radius:14px;border:1px solid {BORDER}">
          <div style="font-weight:700;color:#9db1cb;">Próxima Previsão</div>
          <div style="font-size:28px;color:#e9eef5;margin-top:6px;">${prox_valor:,.6f}</div>
          <div style="color:{tend_color};margin-top:6px;font-weight:700">{tend} · {delta:+.6f} ({pct:+.2f}%)</div>
          <div style="color:#9db1cb;margin-top:6px;">{moeda_label} em {horizonte}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{CARD_BG};padding:18px;border-radius:14px;border:1px solid {BORDER};">
          <div style="font-weight:700;color:#9db1cb;">Performance (hold-out)</div>
          <div style="color:{FONT_COLOR};margin-top:6px;">RMSE: {rmse:.6f}</div>
          <div style="color:{FONT_COLOR};">MAE: {mae:.6f}</div>
          <div style="color:{FONT_COLOR};">R²: {r2:.3f}</div>
          <div style="margin-top:8px;color:{FONT_COLOR};">Confiança (0–100): <b>{conf*100:.0f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        hist_tail = series_hist.tail(200)
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, name="Histórico", mode="lines", line=dict(color=COLOR_REAL)))
        # banda 90%
        fig_fc.add_trace(go.Scatter(x=fc.index, y=fc["high"], name="Intervalo (90%)", mode="lines", line=dict(width=0), showlegend=False))
        fig_fc.add_trace(go.Scatter(x=fc.index, y=fc["low"],  name="Intervalo (90%)", mode="lines",
                                    fill="tonexty", fillcolor=COLOR_BAND, line=dict(width=0), showlegend=True))
        # linha prevista
        fig_fc.add_trace(go.Scatter(x=fc.index, y=fc["prev"], name="Forecast", mode="lines", line=dict(color=COLOR_PREV, width=2)))
        fig_fc = _style_fig(fig_fc, "Projeção (média do ensemble + banda 90%)", height=320)
        st.plotly_chart(fig_fc, use_container_width=True)

        colb1, colb2 = st.columns(2)
        with colb1:
            if st.button("💾 Salvar previsão"):
                inserted = salvar_previsao(engine, moeda_id, steps, fc, rmse, mae, r2)
                st.success(f"Salvo {inserted} timestamp(s) previsto(s) para {moeda_label} ({horizonte}).")
        with colb2:
            if st.button("🗂️ Ver previsões recentes"):
                df_prev = listar_previsoes_recentes(engine, moeda_id)
                if df_prev.empty:
                    st.info("Ainda não há previsões salvas para esta moeda.")
                else:
                    st.dataframe(df_prev, use_container_width=True, height=300)

if __name__ == "__main__":
    show()
