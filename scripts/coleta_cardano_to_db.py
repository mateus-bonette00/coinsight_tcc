import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine

# Altere o id abaixo para o id correto da Cardano (ADA) na sua tabela 'moedas'
ID_CARDANO = 3  # No seu print, ADA está com id = 3

# Conexão com o banco
engine = create_engine('postgresql+psycopg2://postgres:postgres@localhost:5432/coinsight')

# Baixa dados históricos da Cardano
df = yf.download('ADA-USD', start='2017-10-01', group_by=None)  # Cardano só tem histórico a partir de 2017
df = df.reset_index()

# Ajusta as colunas (tira multi-index, corrige nomes)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(-1)
df.columns = [str(c) if c != '' else 'timestamp' for c in df.columns]
df = df.rename(columns={'Close': 'preco', 'Volume': 'volume'})

# Adiciona as colunas exigidas pelo banco
df['moeda_id'] = ID_CARDANO
df['variacao'] = None
df = df[['moeda_id', 'timestamp', 'preco', 'volume', 'variacao']]

# Salva no banco
df.to_sql('precos', engine, if_exists='append', index=False)
print("Dados da Cardano (ADA) salvos no PostgreSQL com sucesso!")
