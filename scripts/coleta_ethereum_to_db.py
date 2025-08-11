import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine

# Defina o id correto da ETH na tabela moedas
ID_ETHEREUM = 2  # No seu banco, ETH tem id 2

# Conexão
engine = create_engine('postgresql+psycopg2://postgres:postgres@localhost:5432/coinsight')

# Baixa os dados da Ethereum
df = yf.download('ETH-USD', start='2014-01-01', group_by=None)
df = df.reset_index()
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(-1)
df.columns = [str(c) if c != '' else 'timestamp' for c in df.columns]
df = df.rename(columns={'Close': 'preco', 'Volume': 'volume'})

df['moeda_id'] = ID_ETHEREUM
df['variacao'] = None
df = df[['moeda_id', 'timestamp', 'preco', 'volume', 'variacao']]

df.to_sql('precos', engine, if_exists='append', index=False)
print("Dados do Ethereum (ETH) salvos no PostgreSQL com sucesso!")
