#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/streamlit_app/.venv/bin/python"
ETL="$DIR/scripts/etl_coins_ohlc.py"

export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/coinsight"

echo ""
echo "============================================================"
echo "  CoinSight TCC — Populando banco com dados históricos"
echo "============================================================"
echo ""
echo "Baixando dados de BTC, ETH, ADA e SOL do Yahoo Finance..."
echo "Isso pode levar alguns minutos. Aguarde."
echo ""

"$VENV" "$ETL" --interval 1h

echo ""
echo "============================================================"
echo "✅ ETL concluído! Verificando banco..."
echo "============================================================"
"$VENV" "$DIR/check_db.py"

echo ""
echo "============================================================"
echo "Para rodar o projeto:"
echo ""
echo "  cd $DIR/streamlit_app"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py"
echo "============================================================"
echo ""
