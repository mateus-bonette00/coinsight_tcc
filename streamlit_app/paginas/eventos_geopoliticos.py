# streamlit_app/paginas/eventos_geopoliticos.py
import os
import math
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from datetime import datetime, timezone
from componentes.noticias import mostrar_noticias_geopoliticas


# ---------- conexão ----------
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

# ---------- utils ----------
def flag_emoji(country_code: str) -> str:
    """Transforma 'US' -> 🇺🇸 ; 'CN' -> 🇨🇳"""
    if not country_code or len(country_code) != 2:
        return "🌐"
    base = 127397
    return chr(ord(country_code[0].upper()) + base) + chr(ord(country_code[1].upper()) + base)

def time_ago(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    now = datetime.now(timezone.utc)
    diff = now - ts.to_pydatetime()
    s = int(diff.total_seconds())
    if s < 60:
        return f"{s}s atrás"
    m = s // 60
    if m < 60:
        return f"{m} min atrás"
    h = m // 60
    if h < 24:
        return f"{h} horas atrás"
    d = h // 24
    return f"{d} dias atrás"

def badge(texto, bg="#1E2533", fg="#E8E8E8"):
    return f"<span style='display:inline-block;padding:6px 10px;border-radius:10px;background:{bg};color:{fg};font-size:12px;'> {texto} </span>"

def sev_badge(nivel: str) -> str:
    cores = {
        "Alto": ("#43221f", "#ff9b8a"),
        "Médio": ("#3a341d", "#f6e08a"),
        "Baixo": ("#203c2c", "#7be0a2"),
    }
    bg, fg = cores.get(nivel, ("#283142", "#cfe2ff"))
    return badge(nivel, bg, fg)

def sent_badge(sent: str) -> str:
    cores = {
        "Positivo": ("#143b2b", "#31fc94"),
        "Neutro": ("#2b2f3a", "#cfd8dc"),
        "Negativo": ("#3b1e22", "#ff6f6f"),
    }
    bg, fg = cores.get(sent, ("#2b2f3a", "#cfd8dc"))
    return badge(sent, bg, fg)

# ---------- dados ----------
@st.cache_data(ttl=30, show_spinner=False)
def carregar_eventos(_engine) -> pd.DataFrame:
    try:
        q = text("""
            SELECT
                id, timestamp, pais_codigo, pais_nome, instituicao,
                titulo, descricao, categoria, severidade, sentimento,
                impacto_pct, confianca_pct, moedas
            FROM eventos_geopoliticos
            ORDER BY timestamp DESC
            LIMIT 500
        """)
        df = pd.read_sql_query(q, _engine)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df["moedas"] = df["moedas"].fillna("").apply(lambda s: [m.strip() for m in s.split(",") if m.strip()])
            return df
    except Exception:
        pass

    # >>> sem MOCK: retorna DF vazio
    return pd.DataFrame(columns=[
        "id","timestamp","pais_codigo","pais_nome","instituicao",
        "titulo","descricao","categoria","severidade","sentimento",
        "impacto_pct","confianca_pct","moedas"
    ])


# ---------- render ----------
def render_card(ev: pd.Series):
    flag = flag_emoji((ev.get("pais_codigo") or "")[:2])
    impacto = ev.get("impacto_pct")
    impacto_txt = f"{impacto:+.1f}%" if pd.notna(impacto) else "—"
    conf = ev.get("confianca_pct")
    conf_txt = f"{int(conf)}%" if pd.notna(conf) else "—"
    moedas = ev.get("moedas") or []

    st.markdown(
        f"""
        <div style="background:#141a24;border-radius:16px;padding:18px;margin-bottom:16px;border:1px solid #1f2633;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="display:flex;gap:10px;align-items:center;">
              <div style="font-size:22px">{flag}</div>
              <div>
                <div style="font-weight:700;color:#e9eef5">{ev.get('pais_nome','')}</div>
                <div style="font-size:12px;color:#9db1cb">{ev.get('instituicao','')}</div>
              </div>
            </div>
            <div>{sev_badge(ev.get('severidade',''))}</div>
          </div>

          <div style="margin-top:12px;color:#e9eef5;font-weight:600">{ev.get('titulo','')}</div>
          <div style="margin-top:6px;color:#aeb7c2;font-size:14px">{ev.get('descricao','')}</div>

          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:14px;">
            <div>{sent_badge(ev.get('sentimento','Neutro'))}</div>
            <div style="display:flex;gap:22px;color:#aeb7c2;font-size:13px;">
              <div>⏱ {time_ago(ev.get('timestamp'))}</div>
              <div>📊 {conf_txt}</div>
              <div style="color:{'#31fc94' if (isinstance(impacto,(int,float)) and impacto>=0) else '#ff6f6f'};">
                {impacto_txt}
              </div>
            </div>
          </div>

          <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">
            {"".join(badge(m, "#0f1722", "#cfe2ff") for m in moedas)}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- página ----------
def show():
    # Cabeçalho
    st.markdown("""
        <div style="margin:4px 0 16px 0;">
          <h1 style="margin:0;
              background: linear-gradient(90deg, #00E1D4, #3A80F6, #C64BD9);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent;
              font-weight:800; font-size:48px;">Eventos Geopolíticos</h1>
          <p style="color:#9db1cb;margin:6px 0 0 0;font-size:16px;">
            Acompanhe eventos mundiais que impactam o mercado de criptomoedas
          </p>
        </div>
    """, unsafe_allow_html=True)

    engine = get_engine()
    df = carregar_eventos(engine)

    # Linha de controles
    left, right = st.columns([3, 2])
    with left:
        # segmented_control está disponível no Streamlit 1.37
        try:
            categoria = st.segmented_control(
                "Categoria",
                options=["Todos","Econômico","Inovação","Político"],
                default="Todos"
            )
        except Exception:
            categoria = st.radio(
                "Categoria", ["Todos","Econômico","Inovação","Político"], horizontal=True, index=0
            )
    with right:
        q = st.text_input("Buscar Eventos...", placeholder="País, órgão, título, moeda...")

    with st.expander("Filtros"):
        colf1, colf2, colf3, colf4 = st.columns([1.2, 1, 1, 1])
        pais = colf1.multiselect("País", sorted(df["pais_nome"].dropna().unique().tolist()))
        sev = colf2.multiselect("Severidade", ["Baixo","Médio","Alto"])
        sent = colf3.multiselect("Sentimento", ["Positivo","Neutro","Negativo"])
        moedas = colf4.multiselect("Moedas", sorted({m for lst in df["moedas"].tolist() for m in lst}))
        cold1, cold2 = st.columns(2)
        dt_ini = cold1.date_input("De", value=(pd.Timestamp.utcnow()-pd.Timedelta(days=30)).date())
        dt_fim = cold2.date_input("Até", value=pd.Timestamp.utcnow().date())

    # ---- filtros ----
    work = df.copy()
    if categoria and categoria != "Todos":
        work = work[work["categoria"] == categoria]

    if q:
        qlow = q.lower()
        def _match(row):
            campos = [
                str(row.get("pais_nome","")),
                str(row.get("instituicao","")),
                str(row.get("titulo","")),
                str(row.get("descricao","")),
                ",".join(row.get("moedas") or []),
            ]
            return any(qlow in c.lower() for c in campos)
        work = work[work.apply(_match, axis=1)]

    if pais:
        work = work[work["pais_nome"].isin(pais)]
    if sev:
        work = work[work["severidade"].isin(sev)]
    if sent:
        work = work[work["sentimento"].isin(sent)]
    if moedas:
        work = work[work["moedas"].apply(lambda lst: any(m in (lst or []) for m in moedas))]
        
    extras = []
    if categoria and categoria != "Todos": extras.append(categoria)
    if q: extras.append(q)
    if pais: extras += pais
    if moedas: extras += moedas

    st.markdown("---")
    # paginas/eventos_geopoliticos.py
    mostrar_noticias_geopoliticas(
        max_articles=12,
        termos_extra=extras,
        lang=None,                 # usa PT+EN por padrão (definido no componente)
        categoria=categoria,
        paises=pais,
        moedas=moedas,
        busca_texto=q,
        layout_cols=2,
        titulo="📰 Notícias Geopolíticas (PT/EN, últimas 72h)",
        escopo="geopolitica",
        provedores=("gdelt","gnews")  # GDELT primeiro para geopolítica
    )






    # período
    dt_ini_ts = pd.Timestamp(dt_ini, tz="UTC")
    dt_fim_ts = pd.Timestamp(dt_fim, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    work = work[(work["timestamp"] >= dt_ini_ts) & (work["timestamp"] <= dt_fim_ts)]

    # ---- grid de cards ----
    if work.empty:
        st.info("Nenhum evento encontrado com os filtros atuais.")
        return

    cols = st.columns(2)
    for i, (_, ev) in enumerate(work.sort_values("timestamp", ascending=False).iterrows()):
        with cols[i % 2]:
            render_card(ev)

    # rodapé pequeno
    st.caption(f"Exibindo {len(work)} de {len(df)} eventos.")
    

# ponto de entrada
if __name__ == "__main__":
    show()
