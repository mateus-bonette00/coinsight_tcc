"""
Walk-Forward Analysis - Backtesting Robusto
Simula trading real com re-treinamento periódico
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple, List, Callable
from datetime import datetime, timedelta


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis para validação robusta de modelos

    Simula cenário real:
    1. Treina em janela passada
    2. Testa em janela futura
    3. Desliza janelas
    4. Re-treina periodicamente
    """

    def __init__(self,
                 train_window_size: int = 180,
                 test_window_size: int = 30,
                 retrain_frequency: int = 30,
                 initial_capital: float = 10000,
                 transaction_cost: float = 0.001):
        """
        Args:
            train_window_size: Tamanho da janela de treino (dias)
            test_window_size: Tamanho da janela de teste (dias)
            retrain_frequency: Frequência de re-treinamento (dias)
            initial_capital: Capital inicial
            transaction_cost: Custo de transação (0.001 = 0.1%)
        """
        self.train_window = train_window_size
        self.test_window = test_window_size
        self.retrain_freq = retrain_frequency
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        self.results = None
        self.fold_results = []

    def run(self,
            df: pd.DataFrame,
            model_trainer: Callable,
            feature_cols: List[str],
            target_col: str = 'target_return_1d',
            strategy: str = 'long_short') -> pd.DataFrame:
        """
        Executa Walk-Forward Analysis

        Args:
            df: DataFrame com features, target e timestamp
            model_trainer: Função que recebe (X_train, y_train) e retorna modelo treinado
            feature_cols: Lista de colunas de features
            target_col: Nome da coluna target
            strategy: 'long_short' ou 'long_only'

        Returns:
            DataFrame com resultados completos
        """
        df = df.sort_values('timestamp').reset_index(drop=True)
        n = len(df)

        all_predictions = []
        all_actuals = []
        all_timestamps = []
        all_prices = []
        fold_metrics = []

        current_idx = self.train_window
        fold_number = 1

        print(f"🚀 Iniciando Walk-Forward Analysis")
        print(f"   Janela de Treino: {self.train_window} dias")
        print(f"   Janela de Teste: {self.test_window} dias")
        print(f"   Re-treino a cada: {self.retrain_freq} dias\n")

        # Loop de Walk-Forward
        while current_idx + self.test_window <= n:
            # Define janelas
            train_start = max(0, current_idx - self.train_window)
            train_end = current_idx
            test_start = current_idx
            test_end = min(n, current_idx + self.test_window)

            # Dados de treino e teste
            train_data = df.iloc[train_start:train_end]
            test_data = df.iloc[test_start:test_end]

            if len(train_data) < 50 or len(test_data) < 5:
                current_idx += self.retrain_freq
                continue

            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]

            # Treina modelo
            try:
                print(f"   📊 Fold {fold_number}: Treino={len(train_data)}, Teste={len(test_data)}")
                model = model_trainer(X_train, y_train)

                # Previsões
                predictions = model.predict(X_test)

                # Armazena resultados
                all_predictions.extend(predictions)
                all_actuals.extend(y_test.values)
                all_timestamps.extend(test_data['timestamp'].values)
                all_prices.extend(test_data['close'].values)

                # Métricas do fold
                fold_mae = np.mean(np.abs(predictions - y_test.values))
                fold_rmse = np.sqrt(np.mean((predictions - y_test.values) ** 2))

                fold_metrics.append({
                    'fold': fold_number,
                    'train_start': train_data['timestamp'].iloc[0],
                    'train_end': train_data['timestamp'].iloc[-1],
                    'test_start': test_data['timestamp'].iloc[0],
                    'test_end': test_data['timestamp'].iloc[-1],
                    'n_train': len(train_data),
                    'n_test': len(test_data),
                    'mae': fold_mae,
                    'rmse': fold_rmse
                })

                fold_number += 1

            except Exception as e:
                print(f"   ❌ Erro no fold {fold_number}: {e}")

            # Avança janela
            current_idx += self.retrain_freq

        print(f"\n✅ Walk-Forward concluído: {fold_number-1} folds processados\n")

        # Cria DataFrame de resultados
        self.results = pd.DataFrame({
            'timestamp': all_timestamps,
            'price': all_prices,
            'actual_return': all_actuals,
            'predicted_return': all_predictions
        })

        self.fold_results = pd.DataFrame(fold_metrics)

        # Calcula métricas de trading
        self._calculate_trading_metrics(strategy)

        return self.results

    def _calculate_trading_metrics(self, strategy: str):
        """Calcula métricas de trading baseadas nas previsões"""
        df = self.results.copy()

        # Define posições
        if strategy == 'long_short':
            df['position'] = np.where(df['predicted_return'] > 0, 1, -1)
        else:  # long_only
            df['position'] = np.where(df['predicted_return'] > 0, 1, 0)

        # Calcula retornos da estratégia
        df['strategy_return'] = df['position'] * df['actual_return']

        # Custos de transação
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

    def calculate_metrics(self) -> Dict[str, float]:
        """Calcula métricas completas de performance"""
        if self.results is None:
            raise RuntimeError("Execute run() primeiro")

        df = self.results

        # Métricas de erro
        mae = np.mean(np.abs(df['predicted_return'] - df['actual_return']))
        rmse = np.sqrt(np.mean((df['predicted_return'] - df['actual_return']) ** 2))
        mape = np.mean(np.abs((df['actual_return'] - df['predicted_return']) /
                             np.where(df['actual_return'] != 0, df['actual_return'], 1))) * 100

        # Acurácia direcional
        actual_dir = np.sign(df['actual_return'])
        pred_dir = np.sign(df['predicted_return'])
        directional_acc = np.mean(actual_dir == pred_dir)

        # Retornos
        total_return_strategy = (df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        total_return_bh = (df['buy_hold_value'].iloc[-1] / self.initial_capital - 1) * 100

        # Retorno anualizado
        n_days = len(df)
        annual_return_strategy = ((df['portfolio_value'].iloc[-1] / self.initial_capital) ** (252 / n_days) - 1) * 100
        annual_return_bh = ((df['buy_hold_value'].iloc[-1] / self.initial_capital) ** (252 / n_days) - 1) * 100

        # Volatilidade anualizada
        volatility_strategy = df['strategy_return_net'].std() * np.sqrt(252) * 100
        volatility_bh = df['actual_return'].std() * np.sqrt(252) * 100

        # Sharpe Ratio
        sharpe_strategy = (df['strategy_return_net'].mean() / df['strategy_return_net'].std()) * np.sqrt(252) \
            if df['strategy_return_net'].std() > 0 else 0
        sharpe_bh = (df['actual_return'].mean() / df['actual_return'].std()) * np.sqrt(252) \
            if df['actual_return'].std() > 0 else 0

        # Sortino Ratio (penaliza apenas volatilidade negativa)
        downside_returns_strategy = df['strategy_return_net'][df['strategy_return_net'] < 0]
        downside_std_strategy = downside_returns_strategy.std() * np.sqrt(252) if len(downside_returns_strategy) > 0 else 1
        sortino_strategy = (annual_return_strategy / 100) / (downside_std_strategy / 100) \
            if downside_std_strategy > 0 else 0

        downside_returns_bh = df['actual_return'][df['actual_return'] < 0]
        downside_std_bh = downside_returns_bh.std() * np.sqrt(252) if len(downside_returns_bh) > 0 else 1
        sortino_bh = (annual_return_bh / 100) / (downside_std_bh / 100) if downside_std_bh > 0 else 0

        # Maximum Drawdown
        def calc_max_dd(cumulative_returns):
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            return drawdown.min() * 100

        max_dd_strategy = calc_max_dd(df['cumulative_strategy'])
        max_dd_bh = calc_max_dd(df['cumulative_buy_hold'])

        # Calmar Ratio (retorno/max_drawdown)
        calmar_strategy = annual_return_strategy / abs(max_dd_strategy) if max_dd_strategy != 0 else 0
        calmar_bh = annual_return_bh / abs(max_dd_bh) if max_dd_bh != 0 else 0

        # Win rate
        winning_trades = (df['strategy_return_net'] > 0).sum()
        total_trades = (df['position_change'] > 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Profit factor
        gross_profit = df[df['strategy_return_net'] > 0]['strategy_return_net'].sum()
        gross_loss = abs(df[df['strategy_return_net'] < 0]['strategy_return_net'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_acc,
            'total_return_strategy': total_return_strategy,
            'total_return_buy_hold': total_return_bh,
            'annual_return_strategy': annual_return_strategy,
            'annual_return_buy_hold': annual_return_bh,
            'volatility_strategy': volatility_strategy,
            'volatility_buy_hold': volatility_bh,
            'sharpe_strategy': sharpe_strategy,
            'sharpe_buy_hold': sharpe_bh,
            'sortino_strategy': sortino_strategy,
            'sortino_buy_hold': sortino_bh,
            'max_drawdown_strategy': max_dd_strategy,
            'max_drawdown_buy_hold': max_dd_bh,
            'calmar_strategy': calmar_strategy,
            'calmar_buy_hold': calmar_bh,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': int(total_trades),
            'n_folds': len(self.fold_results),
            'final_portfolio_value': df['portfolio_value'].iloc[-1],
            'final_buy_hold_value': df['buy_hold_value'].iloc[-1]
        }

    def plot_results(self, title: str = "Walk-Forward Analysis Results") -> go.Figure:
        """Visualização completa dos resultados"""
        if self.results is None:
            raise RuntimeError("Execute run() primeiro")

        df = self.results

        # Cria subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.35, 0.25, 0.2, 0.2],
            subplot_titles=(
                'Valor do Portfólio (Walk-Forward)',
                'Previsões vs Realidade',
                'Retornos Diários',
                'Drawdown'
            )
        )

        # 1. Valor do portfólio
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

        # Marca re-treinamentos
        for _, fold in self.fold_results.iterrows():
            if fold['fold'] > 1:  # Pula primeiro fold
                fig.add_vline(
                    x=fold['test_start'],
                    line=dict(color='#FFA500', width=1, dash='dot'),
                    row=1, col=1
                )

        # 2. Previsões vs Realidade
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['actual_return'] * 100,
                name='Retorno Real',
                line=dict(color='#31fc94', width=1),
                opacity=0.7
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['predicted_return'] * 100,
                name='Previsão',
                line=dict(color='#3A80F6', width=1),
                opacity=0.7
            ),
            row=2, col=1
        )

        # 3. Retornos diários
        colors = ['#31fc94' if r > 0 else '#ff6f6f' for r in df['strategy_return_net']]
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['strategy_return_net'] * 100,
                name='Retornos',
                marker_color=colors,
                showlegend=False
            ),
            row=3, col=1
        )

        # 4. Drawdown
        running_max_strategy = df['cumulative_strategy'].cummax()
        dd_strategy = (df['cumulative_strategy'] - running_max_strategy) / running_max_strategy * 100

        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=dd_strategy,
                name='Drawdown',
                line=dict(color='#FF6B6B', width=2),
                fill='tozeroy',
                showlegend=False
            ),
            row=4, col=1
        )

        # Layout
        fig.update_layout(
            title=title,
            height=1000,
            hovermode='x unified',
            plot_bgcolor='#0F131A',
            paper_bgcolor='#0F131A',
            font=dict(color='#cfd8dc'),
            showlegend=True,
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
        fig.update_yaxes(title_text="Retorno (%)", row=3, col=1, gridcolor='#1f2633')
        fig.update_yaxes(title_text="Drawdown (%)", row=4, col=1, gridcolor='#1f2633')
        fig.update_xaxes(gridcolor='#1f2633')

        return fig

    def plot_fold_performance(self) -> go.Figure:
        """Plota performance de cada fold"""
        if not self.fold_results or len(self.fold_results) == 0:
            return go.Figure()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('MAE por Fold', 'RMSE por Fold')
        )

        # MAE
        fig.add_trace(
            go.Bar(
                x=self.fold_results['fold'],
                y=self.fold_results['mae'],
                name='MAE',
                marker_color='#00E1D4',
                showlegend=False
            ),
            row=1, col=1
        )

        # RMSE
        fig.add_trace(
            go.Bar(
                x=self.fold_results['fold'],
                y=self.fold_results['rmse'],
                name='RMSE',
                marker_color='#3A80F6',
                showlegend=False
            ),
            row=1, col=2
        )

        # Linha de média
        fig.add_hline(
            y=self.fold_results['mae'].mean(),
            line=dict(color='#FFA500', dash='dash'),
            row=1, col=1
        )

        fig.add_hline(
            y=self.fold_results['rmse'].mean(),
            line=dict(color='#FFA500', dash='dash'),
            row=1, col=2
        )

        fig.update_layout(
            title='Performance por Fold (Walk-Forward)',
            height=400,
            plot_bgcolor='#0F131A',
            paper_bgcolor='#0F131A',
            font=dict(color='#cfd8dc')
        )

        fig.update_yaxes(gridcolor='#1f2633')
        fig.update_xaxes(gridcolor='#1f2633', title_text='Fold')

        return fig

    def generate_report(self) -> str:
        """Gera relatório completo"""
        metrics = self.calculate_metrics()

        report = f"""
🚀 RELATÓRIO WALK-FORWARD ANALYSIS
{'='*70}

🔄 CONFIGURAÇÃO
Janela de Treino:      {self.train_window} dias
Janela de Teste:       {self.test_window} dias
Re-treino a cada:      {self.retrain_freq} dias
Número de Folds:       {metrics['n_folds']}
Capital Inicial:       ${self.initial_capital:,.2f}

📊 MÉTRICAS DE PREVISÃO
MAE (Erro Médio):                 {metrics['mae']:.6f}
RMSE (Erro Quadrático):           {metrics['rmse']:.6f}
MAPE (Erro Percentual):           {metrics['mape']:.2f}%
Acurácia Direcional:              {metrics['directional_accuracy']*100:.2f}%

💰 RETORNOS
Estratégia ML:                    {metrics['total_return_strategy']:>8.2f}%
Buy & Hold:                       {metrics['total_return_buy_hold']:>8.2f}%
Alpha (Diferença):                {metrics['total_return_strategy'] - metrics['total_return_buy_hold']:>8.2f}%

📈 RETORNO ANUALIZADO
Estratégia ML:                    {metrics['annual_return_strategy']:>8.2f}%
Buy & Hold:                       {metrics['annual_return_buy_hold']:>8.2f}%

📊 VOLATILIDADE ANUALIZADA
Estratégia ML:                    {metrics['volatility_strategy']:>8.2f}%
Buy & Hold:                       {metrics['volatility_buy_hold']:>8.2f}%

⚡ MÉTRICAS AJUSTADAS AO RISCO
Sharpe Ratio:
  Estratégia ML:                  {metrics['sharpe_strategy']:>8.2f}
  Buy & Hold:                     {metrics['sharpe_buy_hold']:>8.2f}

Sortino Ratio:
  Estratégia ML:                  {metrics['sortino_strategy']:>8.2f}
  Buy & Hold:                     {metrics['sortino_buy_hold']:>8.2f}

Calmar Ratio:
  Estratégia ML:                  {metrics['calmar_strategy']:>8.2f}
  Buy & Hold:                     {metrics['calmar_buy_hold']:>8.2f}

📉 DRAWDOWN
Max Drawdown:
  Estratégia ML:                  {metrics['max_drawdown_strategy']:>8.2f}%
  Buy & Hold:                     {metrics['max_drawdown_buy_hold']:>8.2f}%

🎯 ESTATÍSTICAS DE TRADING
Win Rate:                         {metrics['win_rate']:>8.2f}%
Profit Factor:                    {metrics['profit_factor']:>8.2f}
Total de Trades:                  {metrics['total_trades']:>8.0f}

💵 VALOR FINAL
Estratégia ML:                    ${metrics['final_portfolio_value']:>12,.2f}
Buy & Hold:                       ${metrics['final_buy_hold_value']:>12,.2f}
Diferença:                        ${metrics['final_portfolio_value'] - metrics['final_buy_hold_value']:>12,.2f}

💡 AVALIAÇÃO GERAL
{'✅ EXCELENTE' if metrics['sharpe_strategy'] > 1.5 and metrics['total_return_strategy'] > metrics['total_return_buy_hold'] else
 '✅ BOM' if metrics['sharpe_strategy'] > 1.0 and metrics['total_return_strategy'] > 0 else
 '⚠️ MODERADO' if metrics['total_return_strategy'] > 0 else
 '❌ FRACO'}

Recomendação: {'Estratégia ML supera Buy & Hold!' if metrics['total_return_strategy'] > metrics['total_return_buy_hold'] else 'Considerar melhorias no modelo'}

{'='*70}
        """

        return report
