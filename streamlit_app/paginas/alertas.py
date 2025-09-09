import os, pandas as pd, streamlit as st
from sqlalchemy import create_engine, text

@st.cache_resource(show_spinner=False)
def get_engine():
    url = os.getenv("DATABASE_URL") or "postgresql+psycopg2://postgres:postgres@localhost:5432/coinsight"
    return create_engine(url, pool_pre_ping=True)

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

def ensure_schema(engine):
    with engine.begin() as c:
        c.execute(text("""
        CREATE TABLE IF NOT EXISTS alertas (
          id SERIAL PRIMARY KEY,
          created_at TIMESTAMPTZ DEFAULT NOW(),
          moeda_id INT NOT NULL,
          tipo TEXT NOT NULL,        -- 'preco' ou 'previsao'
          condicao TEXT NOT NULL,    -- 'acima','abaixo','>=','<='
          valor DOUBLE PRECISION NOT NULL,
          horizonte_h INT,           -- obrigatório quando tipo='previsao'
          notificar_email BOOLEAN DEFAULT FALSE,
          notificar_push  BOOLEAN DEFAULT FALSE,
          ativo BOOLEAN DEFAULT TRUE
        );
        CREATE TABLE IF NOT EXISTS alertas_log (
          id SERIAL PRIMARY KEY,
          alerta_id INT REFERENCES alertas(id),
          disparado_em TIMESTAMPTZ DEFAULT NOW(),
          valor_atual DOUBLE PRECISION,
          origem TEXT,               -- 'preco' ou 'previsao(h)'
          mensagem TEXT
        );
        """))

@st.cache_data(ttl=60, show_spinner=False)
def listar_moedas(_engine):
    q = text("""
        SELECT id,
               CASE WHEN COALESCE(simbolo,'')<>'' AND COALESCE(nome,'')<>'' THEN UPPER(simbolo)||' ('||nome||')'
                    WHEN COALESCE(simbolo,'')<>'' THEN UPPER(simbolo)
                    WHEN COALESCE(nome,'')<>'' THEN nome
                    ELSE id::text END AS label
        FROM moedas
        WHERE COALESCE(ativo, TRUE)=TRUE
        ORDER BY UPPER(simbolo) NULLS LAST, nome NULLS LAST, id
    """)
    try:
        df = pd.read_sql_query(q, _engine)
        if not df.empty: return dict(zip(df["label"], df["id"]))
    except Exception:
        pass
    df = pd.read_sql_query(text("SELECT DISTINCT moeda_id AS id FROM precos ORDER BY 1"), _engine)
    df["label"] = df["id"].apply(lambda i: f"Moeda {int(i)}")
    return dict(zip(df["label"], df["id"]))

def ultimo_preco(engine, moeda_id:int) -> float|None:
    # Usa 'close' se existir, senão 'preco'
    col = "close" if _coluna_existe(engine, "precos", "close") else ("preco" if _coluna_existe(engine, "precos", "preco") else None)
    if not col: return None
    q = text(f"SELECT {col} AS px FROM precos WHERE moeda_id=:m ORDER BY timestamp DESC LIMIT 1")
    df = pd.read_sql_query(q, engine, params={"m": moeda_id})
    return float(df.iloc[0,0]) if not df.empty else None

def ultima_previsao(engine, moeda_id:int, horizonte_h:int) -> float|None:
    q = text("""
        SELECT valor FROM previsoes
        WHERE moeda_id=:m AND horizonte_h=:h
        ORDER BY trained_at DESC, ts_previsto DESC
        LIMIT 1
    """)
    df = pd.read_sql_query(q, engine, params={"m": moeda_id, "h": horizonte_h})
    return float(df.iloc[0,0]) if not df.empty else None

def _compara(cond: str, atual: float, alvo: float) -> bool:
    if cond in ("acima", ">=", "maior", "maior_igual"): return atual >= alvo
    if cond in ("abaixo","<=","menor","menor_igual"):   return atual <= alvo
    return False

def avaliar_alertas(engine):
    with engine.begin() as c:
        df = pd.read_sql_query(text("SELECT * FROM alertas WHERE ativo=TRUE ORDER BY id DESC"), c)
        disps = 0
        for _, row in df.iterrows():
            val_atual = None; origem = None
            if row["tipo"] == "preco":
                val_atual = ultimo_preco(engine, int(row["moeda_id"])); origem = "preco"
            elif row["tipo"] == "previsao":
                h = int(row.get("horizonte_h") or 24)
                val_atual = ultima_previsao(engine, int(row["moeda_id"]), h); origem = f"previsao({h}h)"
            if val_atual is None: continue
            if _compara(row["condicao"], val_atual, float(row["valor"])):
                msg = f"[{origem}] {row['condicao']} de {row['valor']}, atual={val_atual:.6f}"
                c.execute(text("""
                    INSERT INTO alertas_log(alerta_id, valor_atual, origem, mensagem)
                    VALUES (:a,:v,:o,:m)
                """), {"a": int(row["id"]), "v": float(val_atual), "o": origem, "m": msg})
                disps += 1
        return disps

def show():
    st.markdown("""
      <div style="margin:4px 0 16px 0;">
        <h1 style="margin:0;
            background: linear-gradient(90deg, #00E1D4, #3A80F6, #C64BD9);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-weight:800; font-size:48px;">Alertas</h1>
        <p style="color:#9db1cb;margin:6px 0 0 0;font-size:16px;">Monitore o mercado por preço real ou por previsões de IA</p>
      </div>
    """, unsafe_allow_html=True)

    engine = get_engine()
    ensure_schema(engine)

    moedas = listar_moedas(engine)
    if not moedas:
        st.info("Sem moedas na base.")
        return

    st.subheader("Criar Alerta")
    c1,c2,c3,c4 = st.columns([2,1,1,1])
    moeda_label = c1.selectbox("Criptomoeda", list(moedas.keys()))
    tipo  = c2.selectbox("Tipo", ["preco","previsao"], help="Use 'previsao' para disparar com base no valor previsto pelo modelo.")
    cond  = c3.selectbox("Condição", ["acima","abaixo"])
    valor = c4.number_input("Valor (USD)", min_value=0.0, step=0.0001, format="%.6f")
    colH = st.columns(4)[0]
    horizonte = colH.selectbox("Horizonte (se 'previsao')", [1,24,168], index=1)
    cA, cB = st.columns(2)
    email = cA.toggle("Notificar por Email", value=False)
    push  = cB.toggle("Notificar Push",  value=False)
    if st.button("➕ Criar Alerta", use_container_width=True):
        with engine.begin() as c:
            c.execute(text("""
                INSERT INTO alertas(moeda_id,tipo,condicao,valor,horizonte_h,notificar_email,notificar_push)
                VALUES (:m,:t,:c,:v,:h,:e,:p)
            """), {"m": moedas[moeda_label], "t": tipo, "c": cond, "v": float(valor),
                   "h": int(horizonte if tipo=="previsao" else 0),
                   "e": email, "p": push})
        st.success("Alerta criado.")

    st.markdown("---")
    if st.button("🔔 Avaliar alertas agora"):
        n = avaliar_alertas(engine)
        st.info(f"Avaliação concluída. Disparos gerados: {n}.")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Ativos")
        ativos = pd.read_sql_query(text("""
            SELECT a.*, COALESCE(UPPER(m.simbolo)||' ('||m.nome||')','Moeda '||a.moeda_id::text) AS moeda
            FROM alertas a LEFT JOIN moedas m ON m.id=a.moeda_id
            WHERE a.ativo=TRUE ORDER BY a.created_at DESC
        """), engine)
        st.dataframe(ativos, use_container_width=True, height=280)
    with colB:
        st.subheader("Histórico de Disparos")
        hist = pd.read_sql_query(text("""
            SELECT l.disparado_em, l.origem, l.valor_atual, l.mensagem, a.id AS alerta_id
            FROM alertas_log l JOIN alertas a ON a.id=l.alerta_id
            ORDER BY l.disparado_em DESC LIMIT 100
        """), engine)
        st.dataframe(hist, use_container_width=True, height=280)
