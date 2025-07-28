import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql+psycopg2://coinsight_user:papabento123@localhost:5432/coinsight')

df = yf.download('SOL-USD', start='2020-01-01', group_by=None)  # Solana só tem histórico desde 2020
df = df.reset_index()
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(-1)
df.columns = [str(c) if c != '' else 'timestamp' for c in df.columns]
df = df.rename(columns={'Close': 'preco', 'Volume': 'volume'})

df['moeda_id'] = 4  # id da SOL no seu banco
df['variacao'] = None
df = df[['moeda_id', 'timestamp', 'preco', 'volume', 'variacao']]

df.to_sql('precos', engine, if_exists='append', index=False)
print("Dados da Solana salvos no PostgreSQL com sucesso!")
print(df.head())  # Exibe as primeiras linhas para verificar se os dados foram carregados corretamente