import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine

USER = 'coinsight_user'
PASSWORD = 'papabento123'
HOST = 'localhost'
PORT = '5432'
DB = 'coinsight'

engine = create_engine(f'postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}')

df = yf.download('BTC-USD', start='2014-01-01', group_by=None)
df = df.reset_index()

# Se colunas são MultiIndex, pega só o nome do preço (último nível)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(-1)

# Corrige a coluna de data sem nome
df.columns = [str(c) if c != '' else 'timestamp' for c in df.columns]

print(df.columns)
print(df.head())

# Renomeia as colunas para garantir
df = df.rename(columns={'Close': 'preco', 'Volume': 'volume'})

df['moeda_id'] = 1
df['variacao'] = None

df = df[['moeda_id', 'timestamp', 'preco', 'volume', 'variacao']]

df.to_sql('precos', engine, if_exists='append', index=False)
print("Dados do BTC salvos no PostgreSQL com sucesso!")
