# streamlit_app/paginas/sentimento_social.py
import os, re, pandas as pd, streamlit as st, requests
from datetime import timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from componentes.feed_social import mostrar_feed_social, TWEETS_SIMULADOS  # reaproveita teu componente
# ↑ feed_social já lê TWITTER_BEARER_TOKEN do .env — aqui vamos tentar buscar tweets tbm

ANALYZER = SentimentIntensityAnalyzer()
BEARER = os.getenv("TWITTER_BEARER_TOKEN", "")

def _score(txt: str) -> float:
    try:
        return ANALYZER.polarity_scores(txt or "")["compound"]
    except Exception:
        return 0.0

def _classify(compound: float) -> str:
    if compound >= 0.05: return "Positivo"
    if compound <= -0.05: return "Negativo"
    return "Neutro"

def _fetch_tweets(max_results: int = 60) -> list[dict]:
    """
    Busca tweets recentes via API v2. Se falhar ou não houver token → fallback simulado.
    Retorna [{autor, texto, created_at}, ...]
    """
    if not BEARER:
        return [{"autor": t["autor"], "texto": t["texto"], "created_at": t["data"]} for t in TWEETS_SIMULADOS]

    q = "(bitcoin OR btc OR ethereum OR eth OR crypto) (lang:en OR lang:pt) -is:retweet"
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {"query": q, "max_results": min(100, max_results), "tweet.fields": "created_at,text,lang,author_id"}
    headers = {"Authorization": f"Bearer {BEARER}"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", []) or []
        if not data:
            raise RuntimeError("Sem tweets")
        return [{"autor": t.get("author_id"), "texto": t.get("text", ""), "created_at": t.get("created_at")} for t in data]
    except Exception:
        return [{"autor": t["autor"], "texto": t["texto"], "created_at": t["data"]} for t in TWEETS_SIMULADOS]

def _extract_hashtags(text: str) -> list[str]:
    return [h.lower() for h in re.findall(r"#\w+", text or "")]

def show():
    # Cabeçalho
    st.markdown("""
        <div style="margin:4px 0 12px 0;">
          <h1 style="margin:0;
              background: linear-gradient(90deg, #00E1D4, #3A80F6, #C64BD9);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent;
              font-weight:800; font-size:48px;">Análise de Sentimento Social</h1>
          <p style="color:#9db1cb;margin:6px 0 0 0;font-size:16px;">
            Monitoramento em tempo quase real do sentimento sobre criptomoedas
          </p>
        </div>
    """, unsafe_allow_html=True)

    # 1) Coleta
    raw = _fetch_tweets(max_results=80)
    df = pd.DataFrame(raw)
    if df.empty:
        st.info("Sem dados de redes sociais agora.")
        return

    # 2) Processamento
    df["timestamp"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["compound"] = df["texto"].apply(_score)
    df["classe"] = df["compound"].apply(_classify)
    df["hora"] = df["timestamp"].dt.floor("H")
    df["hashtags"] = df["texto"].apply(_extract_hashtags)

    # 3) KPIs (janela 24h)
    cutoff = (df["timestamp"].max() - pd.Timedelta(hours=24)) if not df.empty else pd.Timestamp.utcnow() - pd.Timedelta(hours=24)
    last24 = df[df["timestamp"] >= cutoff]
    total = len(last24)
    pos = int((last24["classe"] == "Positivo").sum())
    neu = int((last24["classe"] == "Neutro").sum())
    neg = int((last24["classe"] == "Negativo").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentimento Positivo", f"{(pos/max(total,1))*100:,.0f}%")
    c2.metric("Neutro",               f"{(neu/max(total,1))*100:,.0f}%")
    c3.metric("Negativo",             f"{(neg/max(total,1))*100:,.0f}%")
    c4.metric("Total de Posts (24h)", f"{total:,}")

    st.markdown("---")

    # 4) Distribuição (pizza)
    import plotly.express as px
    pie_df = last24["classe"].value_counts().rename_axis("classe").reset_index(name="qtd")
    pie = px.pie(pie_df, names="classe", values="qtd", hole=0.55)
    pie.update_layout(height=320, paper_bgcolor="#0F131A", plot_bgcolor="#0F131A", font_color="#cfd8dc", margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(pie, use_container_width=True)

    # 5) Tendência 24h (área empilhada por classe)
    trend = last24.pivot_table(index="hora", columns="classe", values="texto", aggfunc="count", fill_value=0).reset_index()
    trend = trend.sort_values("hora")

    import plotly.graph_objects as go
    fig = go.Figure()
    for cls in ["Positivo", "Neutro", "Negativo"]:
        if cls in trend.columns:
            fig.add_trace(go.Scatter(x=trend["hora"], y=trend[cls], name=cls, stackgroup="one", mode="lines"))
    fig.update_layout(height=340, paper_bgcolor="#0F131A", plot_bgcolor="#0F131A", font_color="#cfd8dc", margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # 6) Hashtags em alta (top 10 nas últimas 24h)
    flat_tags = [h for lst in last24["hashtags"] for h in lst]
    if flat_tags:
        tags = (pd.Series(flat_tags).value_counts().head(10)).reset_index()
        tags.columns = ["hashtag", "qtd"]
        st.markdown("#### Hashtags em Alta (24h)")
        st.dataframe(tags, use_container_width=True, hide_index=True, height=260)
    else:
        st.caption("Sem hashtags detectadas nas últimas 24h.")

    st.markdown("---")

    # 7) Feed em tempo quase real (teu componente)
    mostrar_feed_social()
