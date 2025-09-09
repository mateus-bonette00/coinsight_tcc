# etl_coins_ohlc.py
import os
from datetime import timedelta
import argparse
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, DateTime, text
from sqlalchemy.dialects.postgresql import insert

# -----------------------------
# Config
# -----------------------------
load_dotenv()

DEFAULT_DB_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/coinsight"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# "1h" (intraday, limitado a ~729d no Yahoo) ou "1d" (diário, histórico longo)
DEFAULT_INTERVAL = os.getenv("INTERVAL", "1h").lower()

# Dicionário de moedas: edite/expanda aqui
COINS = {
    # nome: (moeda_id, ticker, start_daily_backfill)
    "BTC": (1, "BTC-USD", "2014-01-01"),
    "ETH": (2, "ETH-USD", "2015-08-01"),
    "ADA": (3, "ADA-USD", "2017-10-01"),
    "SOL": (4, "SOL-USD", "2020-01-01"),
}

# Limite do Yahoo para intraday
MAX_LOOKBACK_DAYS_1H = 729

# -----------------------------
# Engine + schema
# -----------------------------
engine = create_engine(DB_URL, pool_pre_ping=True)


def ensure_schema():
    """Cria tabela e índices se não existirem."""
    meta = MetaData()
    Table(
        "precos", meta,
        Column("moeda_id", Integer, nullable=False),
        Column("timestamp", DateTime(timezone=True), nullable=False),
        Column("open", Float, nullable=False),
        Column("high", Float, nullable=False),
        Column("low", Float, nullable=False),
        Column("close", Float, nullable=False),
        Column("volume", Float),
        # algumas bases antigas têm coluna 'preco' NOT NULL; não declarar aqui para não colidir
    )
    meta.create_all(engine, checkfirst=True)
    with engine.begin() as conn:
        conn.execute(text(
            "DO $$ BEGIN "
            "IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname='pk_precos_moeda_ts') THEN "
            "  CREATE UNIQUE INDEX pk_precos_moeda_ts ON precos(moeda_id, timestamp); "
            "END IF; "
            "IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname='ix_precos_lookup') THEN "
            "  CREATE INDEX ix_precos_lookup ON precos(moeda_id, timestamp DESC); "
            "END IF; "
            "END $$;"
        ))


def last_timestamp(moeda_id: int) -> pd.Timestamp | None:
    """Último timestamp dessa moeda no banco (UTC)."""
    with engine.begin() as conn:
        ts = conn.execute(
            text("SELECT MAX(timestamp) FROM precos WHERE moeda_id = :m"),
            {"m": moeda_id}
        ).scalar()
    return pd.to_datetime(ts, utc=True) if ts else None


# -----------------------------
# Downloaders
# -----------------------------
def _normalize_yf(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza dataframe do yfinance para colunas: timestamp, open, high, low, close, volume."""
    if df.empty:
        return df

    # MultiIndex => achata para nível OHLC
    if isinstance(df.columns, pd.MultiIndex):
        # tenta pegar o nível com OHLC
        lvl0 = [str(x).lower() for x in df.columns.get_level_values(0)]
        lvl1 = [str(x).lower() for x in df.columns.get_level_values(1)]
        ohlc_keys = {"open", "high", "low", "close", "volume", "adj close"}
        if any(k in lvl0 for k in ohlc_keys):
            df.columns = df.columns.get_level_values(0)
        elif any(k in lvl1 for k in ohlc_keys):
            df.columns = df.columns.get_level_values(1)
        else:
            df.columns = ["_".join(str(p).lower() for p in col if p) for col in df.columns]

    df = df.reset_index()

    # Detecta coluna temporal
    time_col = None
    for cand in ["Datetime", "Date", "datetime", "date"]:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                time_col = c
                break
    if time_col is None:
        raise RuntimeError(f"Não encontrei coluna de tempo em: {df.columns.tolist()}")

    # Padroniza nomes
    df = df.rename(columns={
        time_col: "timestamp",
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj close", "Volume": "volume",
        "open": "open", "high": "high", "low": "low", "close": "close",
        "adj close": "adj close", "volume": "volume",
    })
    df.columns = [str(c).lower() for c in df.columns]

    # Se nomes vierem como 'btc-usd_open', tenta resolver
    if "open" not in df.columns and any("open" in c for c in df.columns):
        def pick(colname):
            cands = [c for c in df.columns if colname in c]
            for c in cands:
                if c.endswith(f"_{colname}") or c.startswith(f"{colname}_"):
                    return c
            return cands[0] if cands else None

        mapping = {k: pick(k) for k in ["open", "high", "low", "close", "volume"]}
        missing = [k for k, v in mapping.items() if v is None]
        if missing:
            raise RuntimeError(f"Colunas ausentes após normalização: {missing}. Colunas: {df.columns.tolist()}")
        df = df.rename(columns={mapping[k]: k for k in mapping if mapping[k]})

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    needed = ["timestamp", "open", "high", "low", "close"]
    opt = ["volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colunas OHLC ausentes: {missing}. Colunas: {df.columns.tolist()}")

    cols = needed + [c for c in opt if c in df.columns]
    df = df[cols].dropna(subset=["open", "high", "low", "close"])

    return df


def download_intraday_1h(ticker: str, since: pd.Timestamp | None) -> pd.DataFrame:
    """
    Baixa dados em 1h respeitando o limite de ~729 dias do Yahoo.
    - Se since is None => últimos 729d
    - Caso contrário, calcula a janela (since -> now) e cap 729d
    """
    now_utc = pd.Timestamp.utcnow().floor("H")
    if since is None:
        period = f"{MAX_LOOKBACK_DAYS_1H}d"
    else:
        delta_days = max(1, int((now_utc - since).days) + 1)
        period = f"{min(delta_days, MAX_LOOKBACK_DAYS_1H)}d"

    raw = yf.download(
        ticker,
        interval="1h",
        period=period,
        auto_adjust=False,
        progress=False,
        group_by=None,
        threads=False,
    )
    if raw.empty:
        return raw

    df = _normalize_yf(raw)
    # Se since existir, filtra para evitar repetir candles
    if since is not None:
        df = df[df["timestamp"] > since]
    return df


def download_daily_1d(ticker: str, start_date: str, since: pd.Timestamp | None) -> pd.DataFrame:
    """
    Baixa dados diários desde start_date. Se já houver dados, baixa a partir do próximo dia do since.
    """
    if since is not None:
        start_date = (since + timedelta(days=1)).date().isoformat()

    raw = yf.download(
        ticker,
        interval="1d",
        start=start_date,
        auto_adjust=False,
        progress=False,
        group_by=None,
    )
    if raw.empty:
        return raw

    df = _normalize_yf(raw)
    if since is not None:
        df = df[df["timestamp"] > since]
    return df


# -----------------------------
# Upsert
# -----------------------------
def upsert_ohlc(moeda_id: int, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["moeda_id"] = moeda_id

    meta = MetaData()
    precos = Table("precos", meta, autoload_with=engine)
    cols = {c.name for c in precos.columns}

    # Tabelas antigas com 'preco' NOT NULL: povoar com 'close'
    if "preco" in cols and "preco" not in df.columns:
        df["preco"] = df["close"]

    ordered = [c for c in ["moeda_id","timestamp","preco","open","high","low","close","volume"] if c in cols]
    payload = df[ordered].to_dict(orient="records")

    stmt = insert(precos).values(payload)
    updatable = {c: stmt.excluded[c] for c in ["preco","open","high","low","close","volume"] if c in cols}
    stmt = stmt.on_conflict_do_update(
        index_elements=["moeda_id","timestamp"],
        set_=updatable
    )

    with engine.begin() as conn:
        res = conn.execute(stmt)
    return res.rowcount or 0


# -----------------------------
# Runner
# -----------------------------
def run_one(name: str, moeda_id: int, ticker: str, start_daily: str, interval: str) -> int:
    lt = last_timestamp(moeda_id)

    if interval == "1h":
        df = download_intraday_1h(ticker, lt)
    elif interval == "1d":
        df = download_daily_1d(ticker, start_daily, lt)
    else:
        raise ValueError("INTERVAL deve ser '1h' ou '1d'.")

    inserted = upsert_ohlc(moeda_id, df)
    base_since = (lt.isoformat() if lt is not None else ("(novo) " + start_daily))
    print(f"[{name}] {inserted} linhas inseridas/atualizadas desde {base_since} (interval={interval})")
    return inserted


def run_all(interval: str):
    ensure_schema()
    total = 0
    for name, (mid, ticker, start) in COINS.items():
        total += run_one(name, mid, ticker, start, interval)
    print(f"Total inserido/atualizado: {total}")


def parse_args():
    parser = argparse.ArgumentParser(description="ETL OHLC para tabela 'precos' (coinsight).")
    parser.add_argument("symbols", nargs="*", help="Moedas a atualizar (ex.: BTC ETH). Vazio = todas.")
    parser.add_argument("--interval", "-i", choices=["1h","1d"], default=DEFAULT_INTERVAL,
                        help="Intervalo das velas (default: valor de INTERVAL no .env ou '1h').")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_schema()

    interval = args.interval
    symbols = [s.upper() for s in args.symbols] if args.symbols else list(COINS.keys())

    unknown = [s for s in symbols if s not in COINS]
    if unknown:
        print(f"Moedas não mapeadas: {', '.join(unknown)}. Disponíveis: {', '.join(COINS.keys())}")
        symbols = [s for s in symbols if s in COINS]

    total = 0
    for key in symbols:
        mid, ticker, start = COINS[key]
        total += run_one(key, mid, ticker, start, interval)
    print(f"Total inserido/atualizado: {total}")
