"""
Sistema de Backtesting para validar previsões em períodos históricos
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple


class Backtester:
    """Backtesting de estratégias baseadas em ML"""
    
    def __init__(self, initial_capital: float = 10000, transaction_cost: float = 0.001):
        """
        Args:
            initial_capital: Capital inicial em USD
            transaction_cost: Custo de transação (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = None
    
    def run_backtest(self, predictions: np.ndarray, actual_prices: pd.Series, 
                     timestamps: pd.Series, strategy: str = 'long_short') -> pd.DataFrame:
        """
        Executa backtest
        
        Args:
            predictions: Array com previsões do modelo (retornos ou direções)
            actual_prices: Série com preços reais
            timestamps: Série com timestamps
            strategy: 'long_short' (compra/venda) ou 'long_only' (só compra)
        
        Returns:
            DataFrame com resultados do backtest
        """
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': actual_prices,
            'prediction': predictions
        })
        
        df['actual_return'] = df['price'].pct_change()
        
        # Define posições baseadas nas previsões
        if strategy == 'long_short':
            # Posição: +1 se prevê alta, -1 se prevê baixa
            df['position'] = np.where(df['prediction'] > 0, 1, -1)
        else:  # long_only
            # Posição: +1 se prevê alta, 0 se prevê baixa
            df['position'] = np.where(df['prediction'] > 0, 1, 0)
        
        # Calcula retornos da estratégia
        df['strategy_return'] = df['position'].shift(1) * df['actual_return']
        
        # Aplica custos de transação quando há mudança de posição
        df['position_change'] = df['position'].diff().abs()
        df['transaction_costs'] = df['position_change'] * self.transaction_cost
        df['strategy_return_net'] = df['strategy_return'] - df['transaction_costs']
        
        # Retornos cumulativos
        df['cumulative_strategy'] = (1 + df['strategy_return_net'].fillna(0)).cumprod()
        df['cumulative_buy_hold'] = (1 + df['actual_return'].fillna(0)).cumprod()
        
        # Valor do portfólio
        df['portfolio_value'] = self.initial_capital * df['cumulative_strategy']
        df['buy_hold_value'] = self.initial_capital * df['cumulative_buy_hold']
        
        self.results = df
        return df
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calcula métricas de performance"""
        if self.results is None:
            raise RuntimeError("Execute run_backtest() primeiro")
        
        df = self.results
        
        # Retorno total
        total_return_strategy = (df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        total_return_buy_hold = (df['buy_hold_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Retorno anualizado
        n_days = len(df)
        annual_return_strategy = ((df['portfolio_value'].iloc[-1] / self.initial_capital) ** (252 / n_days) - 1) * 100
        annual_return_buy_hold = ((df['buy_hold_value'].iloc[-1] / self.initial_capital) ** (252 / n_days) - 1) * 100
        
        # Volatilidade anualizada
        volatility_strategy = df['strategy_return_net'].std() * np.sqrt(252) * 100
        volatility_buy_hold = df['actual_return'].std() * np.sqrt(252) * 100
        
        # Sharpe Ratio (assumindo taxa livre de risco = 0)
        sharpe_strategy = (df['strategy_return_net'].mean() / df['strategy_return_net'].std()) * np.sqrt(252) if df['strategy_return_net'].std() > 0 else 0
        sharpe_buy_hold = (df['actual_return'].mean() / df['actual_return'].std()) * np.sqrt(252) if df['actual_return'].std() > 0 else 0
        
        # Maximum Drawdown
        def calc_max_drawdown(cumulative_returns):
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            return drawdown.min() * 100
        
        max_dd_strategy = calc_max_drawdown(df['cumulative_strategy'])
        max_dd_buy_hold = calc_max_drawdown(df['cumulative_buy_hold'])
        
        # Win rate
        winning_trades = (df['strategy_return_net'] > 0).sum()
        total_trades = (df['position_change'] > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Custos totais
        total_costs = df['transaction_costs'].sum() * self.initial_capital
        
        metrics = {
            'total_return_strategy': total_return_strategy,
            'total_return_buy_hold': total_return_buy_hold,
            'annual_return_strategy': annual_return_strategy,
            'annual_return_buy_hold': annual_return_buy_hold,
            'volatility_strategy': volatility_strategy,
            'volatility_buy_hold': volatility_buy_hold,
            'sharpe_strategy': sharpe_strategy,
            'sharpe_buy_hold': sharpe_buy_hold,
            'max_drawdown_strategy': max_dd_strategy,
            'max_drawdown_buy_hold': max_dd_buy_hold,
            'win_rate': win_rate,
            'total_trades': int(total_trades),
            'total_transaction_costs': total_costs,
            'final_portfolio_value': df['portfolio_value'].iloc[-1],
            'final_buy_hold_value': df['buy_hold_value'].iloc[-1]
        }
        
        return metrics
    
    def plot_results(self, title: str = "Backtest Results") -> go.Figure:
        """Cria visualização interativa dos resultados"""
        if self.results is None:
            raise RuntimeError("Execute run_backtest() primeiro")
        
        df = self.results
        
        # Cria subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Valor do Portfólio', 'Retornos Diários', 'Posições')
        )
        
        # Gráfico 1: Valor do portfólio
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['portfolio_value'],
                name='Estratégia ML',
                line=dict(color='#00E1D4', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['buy_hold_value'],
                name='Buy & Hold',
                line=dict(color='#FF6B6B', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Gráfico 2: Retornos diários
        colors = ['#31fc94' if r > 0 else '#ff6f6f' for r in df['strategy_return_net']]
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['strategy_return_net'] * 100,
                name='Retornos',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Gráfico 3: Posições
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['position'],
                name='Posição',
                line=dict(color='#3A80F6', width=1),
                fill='tozeroy',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Layout
        fig.update_layout(
            title=title,
            height=800,
            hovermode='x unified',
            plot_bgcolor='#0F131A',
            paper_bgcolor='#0F131A',
            font=dict(color='#cfd8dc'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(title_text="Valor (USD)", row=1, col=1, gridcolor='#1f2633')
        fig.update_yaxes(title_text="Retorno (%)", row=2, col=1, gridcolor='#1f2633')
        fig.update_yaxes(title_text="Posição", row=3, col=1, gridcolor='#1f2633')
        fig.update_xaxes(gridcolor='#1f2633')
        
        return fig
    
    def plot_drawdown(self) -> go.Figure:
        """Plota drawdown ao longo do tempo"""
        if self.results is None:
            raise RuntimeError("Execute run_backtest() primeiro")
        
        df = self.results
        
        # Calcula drawdown
        running_max_strategy = df['cumulative_strategy'].cummax()
        dd_strategy = (df['cumulative_strategy'] - running_max_strategy) / running_max_strategy * 100
        
        running_max_bh = df['cumulative_buy_hold'].cummax()
        dd_bh = (df['cumulative_buy_hold'] - running_max_bh) / running_max_bh * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=dd_strategy,
            name='Estratégia ML',
            line=dict(color='#00E1D4', width=2),
            fill='tozeroy'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=dd_bh,
            name='Buy & Hold',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Drawdown ao Longo do Tempo',
            xaxis_title='Data',
            yaxis_title='Drawdown (%)',
            height=400,
            plot_bgcolor='#0F131A',
            paper_bgcolor='#0F131A',
            font=dict(color='#cfd8dc'),
            hovermode='x unified'
        )
        
        fig.update_yaxes(gridcolor='#1f2633')
        fig.update_xaxes(gridcolor='#1f2633')
        
        return fig
    
    def generate_report(self) -> str:
        """Gera relatório textual dos resultados"""
        metrics = self.calculate_metrics()
        
        report = f"""
📊 RELATÓRIO DE BACKTEST
{'='*50}

💰 RETORNOS
Estratégia ML:     {metrics['total_return_strategy']:>8.2f}%
Buy & Hold:        {metrics['total_return_buy_hold']:>8.2f}%
Diferença:         {metrics['total_return_strategy'] - metrics['total_return_buy_hold']:>8.2f}%

📈 RETORNO ANUALIZADO
Estratégia ML:     {metrics['annual_return_strategy']:>8.2f}%
Buy & Hold:        {metrics['annual_return_buy_hold']:>8.2f}%

📊 VOLATILIDADE ANUALIZADA
Estratégia ML:     {metrics['volatility_strategy']:>8.2f}%
Buy & Hold:        {metrics['volatility_buy_hold']:>8.2f}%

⚡ SHARPE RATIO
Estratégia ML:     {metrics['sharpe_strategy']:>8.2f}
Buy & Hold:        {metrics['sharpe_buy_hold']:>8.2f}

📉 MAXIMUM DRAWDOWN
Estratégia ML:     {metrics['max_drawdown_strategy']:>8.2f}%
Buy & Hold:        {metrics['max_drawdown_buy_hold']:>8.2f}%

🎯 ESTATÍSTICAS DE TRADING
Win Rate:          {metrics['win_rate']:>8.2f}%
Total de Trades:   {metrics['total_trades']:>8.0f}
Custos Totais:     ${metrics['total_transaction_costs']:>8.2f}

💵 VALOR FINAL
Estratégia ML:     ${metrics['final_portfolio_value']:>12,.2f}
Buy & Hold:        ${metrics['final_buy_hold_value']:>12,.2f}

{'='*50}
        """
        
        return report