# scripts/coleta_bitcoin.py
from pycoingecko import CoinGeckoAPI
import pandas as pd

cg = CoinGeckoAPI()
btc_data = cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days='30')

prices = btc_data['prices']
df = pd.DataFrame(prices, columns=['timestamp', 'price'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.to_csv('data/bitcoin_prices.csv', index=False)
print("Coleta finalizada, dados salvos em data/bitcoin_prices.csv")
