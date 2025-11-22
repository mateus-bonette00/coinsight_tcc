"""
Engenharia de Features Avançada para Previsão de Criptomoedas
Inclui features técnicas, de volatilidade, tendência e momentum
"""
import numpy as np
import pandas as pd
from typing import Tuple

class FeatureEngine:
    """Cria features técnicas avançadas para ML"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_all_features(self, df: pd.DataFrame, events_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Cria todas as features a partir de dados OHLCV

        Args:
            df: DataFrame com colunas [timestamp, open, high, low, close, volume]
            events_df: DataFrame opcional com eventos geopolíticos para criar features

        Returns:
            DataFrame com todas as features
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Features básicas de retorno
        df = self._add_return_features(df)

        # Features de volatilidade
        df = self._add_volatility_features(df)

        # Features técnicas
        df = self._add_technical_features(df)

        # Features de momentum
        df = self._add_momentum_features(df)

        # Features de volume
        df = self._add_volume_features(df)

        # Features temporais
        df = self._add_temporal_features(df)

        # Features de eventos geopolíticos (se disponível)
        if events_df is not None and not events_df.empty:
            df = self._add_geopolitical_features(df, events_df)

        # Remove NaN das primeiras linhas (devido a lags)
        df = df.dropna().reset_index(drop=True)

        return df
    
    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de retorno simples e log"""
        # Retorno simples
        df['return_1d'] = df['close'].pct_change(1)
        df['return_3d'] = df['close'].pct_change(3)
        df['return_7d'] = df['close'].pct_change(7)
        df['return_14d'] = df['close'].pct_change(14)
        df['return_30d'] = df['close'].pct_change(30)
        
        # Log returns (mais estáveis)
        df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_7d'] = np.log(df['close'] / df['close'].shift(7))
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de volatilidade"""
        # Volatilidade histórica
        for window in [7, 14, 30]:
            df[f'volatility_{window}d'] = df['return_1d'].rolling(window).std()
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        
        # Amplitude intradiária
        df['daily_range'] = (df['high'] - df['low']) / df['close']
        df['avg_range_7d'] = df['daily_range'].rolling(7).mean()
        
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicadores técnicos clássicos"""
        # Médias móveis
        for window in [7, 14, 21, 50, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Distância das médias móveis (features de tendência)
        df['distance_sma7'] = (df['close'] - df['sma_7']) / df['sma_7']
        df['distance_sma21'] = (df['close'] - df['sma_21']) / df['sma_21']
        df['distance_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_lower'] = sma20 - (std20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de momentum"""
        # Rate of change
        for period in [3, 7, 14]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de volume"""
        # Volume relativo
        df['volume_ratio_7d'] = df['volume'] / df['volume'].rolling(7).mean()
        df['volume_ratio_30d'] = df['volume'] / df['volume'].rolling(30).mean()
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        
        # Volume Weighted Average Price
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features temporais (dia da semana, mês, etc)"""
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        df['quarter'] = pd.to_datetime(df['timestamp']).dt.quarter

        # Encoding cíclico para preservar continuidade
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def _add_geopolitical_features(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features baseadas em eventos geopolíticos

        Cria features que capturam:
        - Proximidade temporal de eventos
        - Sentimento agregado de eventos recentes
        - Severidade de eventos recentes
        - Contagem de eventos por categoria
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])

        # Inicializa features
        df['events_last_7d'] = 0
        df['events_last_30d'] = 0
        df['avg_sentiment_7d'] = 0.0
        df['avg_sentiment_30d'] = 0.0
        df['high_severity_events_7d'] = 0
        df['high_severity_events_30d'] = 0
        df['positive_events_7d'] = 0
        df['negative_events_7d'] = 0
        df['days_since_last_event'] = 999  # valor alto padrão
        df['last_event_sentiment'] = 0.0
        df['economic_events_30d'] = 0
        df['political_events_30d'] = 0
        df['innovation_events_30d'] = 0

        # Mapeia sentimentos para valores numéricos
        sentiment_map = {'Positivo': 1, 'Neutro': 0, 'Negativo': -1}
        events_df['sentiment_score'] = events_df['sentimento'].map(sentiment_map).fillna(0)

        # Para cada data nos preços, calcula features de eventos
        for idx, row in df.iterrows():
            current_date = row['timestamp']

            # Eventos nos últimos 7 dias
            events_7d = events_df[
                (events_df['timestamp'] <= current_date) &
                (events_df['timestamp'] > current_date - pd.Timedelta(days=7))
            ]

            # Eventos nos últimos 30 dias
            events_30d = events_df[
                (events_df['timestamp'] <= current_date) &
                (events_df['timestamp'] > current_date - pd.Timedelta(days=30))
            ]

            # Contagens
            df.at[idx, 'events_last_7d'] = len(events_7d)
            df.at[idx, 'events_last_30d'] = len(events_30d)

            # Sentimento médio
            if len(events_7d) > 0:
                df.at[idx, 'avg_sentiment_7d'] = events_7d['sentiment_score'].mean()
                df.at[idx, 'positive_events_7d'] = (events_7d['sentimento'] == 'Positivo').sum()
                df.at[idx, 'negative_events_7d'] = (events_7d['sentimento'] == 'Negativo').sum()
                df.at[idx, 'high_severity_events_7d'] = (events_7d['severidade'] == 'Alto').sum()

            if len(events_30d) > 0:
                df.at[idx, 'avg_sentiment_30d'] = events_30d['sentiment_score'].mean()
                df.at[idx, 'high_severity_events_30d'] = (events_30d['severidade'] == 'Alto').sum()

                # Eventos por categoria
                df.at[idx, 'economic_events_30d'] = (events_30d['categoria'] == 'Econômico').sum()
                df.at[idx, 'political_events_30d'] = (events_30d['categoria'] == 'Político').sum()
                df.at[idx, 'innovation_events_30d'] = (events_30d['categoria'] == 'Inovação').sum()

            # Dias desde o último evento
            past_events = events_df[events_df['timestamp'] <= current_date]
            if len(past_events) > 0:
                last_event = past_events.iloc[-1]
                days_diff = (current_date - last_event['timestamp']).days
                df.at[idx, 'days_since_last_event'] = min(days_diff, 999)
                df.at[idx, 'last_event_sentiment'] = last_event['sentiment_score']

        # Features de interação (preço x eventos)
        df['price_x_sentiment_7d'] = df['close'] * df['avg_sentiment_7d']
        df['volatility_x_events_7d'] = df['volatility_7d'] * df['events_last_7d']

        return df
    
    def create_target(self, df: pd.DataFrame, horizon: int = 1, 
                     target_type: str = 'regression') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Cria variável target para previsão
        
        Args:
            df: DataFrame com features
            horizon: Horizonte de previsão (1 = próximo dia)
            target_type: 'regression' (prever retorno) ou 'classification' (prever direção)
        
        Returns:
            (features_df, target_series)
        """
        if target_type == 'regression':
            # Prever retorno futuro
            target = df['close'].pct_change(horizon).shift(-horizon)
            target.name = f'target_return_{horizon}d'
        else:  # classification
            # Prever direção (1 = sobe, 0 = desce)
            future_return = df['close'].pct_change(horizon).shift(-horizon)
            target = (future_return > 0).astype(int)
            target.name = f'target_direction_{horizon}d'
        
        # Remove linhas com target NaN (últimas linhas)
        valid_idx = target.notna()
        df_clean = df[valid_idx].copy()
        target_clean = target[valid_idx].copy()
        
        return df_clean, target_clean
    
    def get_feature_names(self, df: pd.DataFrame) -> list:
        """Retorna lista de nomes das features (excluindo colunas originais e target)"""
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                   'moeda_id', 'preco', 'variacao']
        exclude += [col for col in df.columns if col.startswith('target_')]
        
        return [col for col in df.columns if col not in exclude]


def prepare_train_test_split(df: pd.DataFrame, test_size: float = 0.2, 
                             val_size: float = 0.1) -> Tuple:
    """
    Split temporal (sem embaralhar) para treino/validação/teste
    
    Args:
        df: DataFrame com features e target
        test_size: Proporção para teste
        val_size: Proporção para validação (do que sobrou após teste)
    
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n = len(df)
    
    # Índices de corte temporal
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    # Features e target
    feature_cols = [col for col in df.columns if not col.startswith('target_') 
                    and col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    target_col = [col for col in df.columns if col.startswith('target_')][0]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split temporal
    X_train = X.iloc[:val_idx]
    X_val = X.iloc[val_idx:test_idx]
    X_test = X.iloc[test_idx:]
    
    y_train = y.iloc[:val_idx]
    y_val = y.iloc[val_idx:test_idx]
    y_test = y.iloc[test_idx:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test