"""
Análise de Correlação entre Eventos Geopolíticos e Preços de Criptomoedas
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict, Tuple, List


class GeopoliticalAnalyzer:
    """Analisa impacto de eventos geopolíticos nos preços"""
    
    def __init__(self):
        self.correlations = {}
        self.impact_analysis = None
    
    def analyze_event_impact(self, events_df: pd.DataFrame, 
                            prices_df: pd.DataFrame,
                            moeda_id: int = 1,
                            window_before: int = 3,
                            window_after: int = 7) -> pd.DataFrame:
        """
        Analisa impacto de eventos nos preços
        
        Args:
            events_df: DataFrame com eventos (timestamp, categoria, severidade, sentimento)
            prices_df: DataFrame com preços (timestamp, close, moeda_id)
            moeda_id: ID da moeda a analisar
            window_before: Dias antes do evento para calcular baseline
            window_after: Dias após evento para medir impacto
        
        Returns:
            DataFrame com análise de impacto por evento
        """
        # Filtra moeda
        prices = prices_df[prices_df['moeda_id'] == moeda_id].copy()
        prices['timestamp'] = pd.to_datetime(prices['timestamp'])
        prices = prices.sort_values('timestamp').set_index('timestamp')
        
        # Calcula retornos diários
        prices['return'] = prices['close'].pct_change()
        
        results = []
        
        for _, event in events_df.iterrows():
            event_time = pd.to_datetime(event['timestamp'])
            
            # Período antes e depois do evento
            start = event_time - pd.Timedelta(days=window_before)
            end = event_time + pd.Timedelta(days=window_after)
            
            # Filtra preços no período
            period_prices = prices.loc[start:end]
            
            if len(period_prices) < 2:
                continue
            
            # Calcula métricas
            baseline_price = period_prices.loc[:event_time, 'close'].mean() if len(period_prices.loc[:event_time]) > 0 else period_prices['close'].iloc[0]
            
            # Preço imediatamente após (D+1)
            after_1d = period_prices.loc[event_time:].iloc[1]['close'] if len(period_prices.loc[event_time:]) > 1 else baseline_price
            impact_1d = (after_1d / baseline_price - 1) * 100
            
            # Preço após window_after dias
            after_nd = period_prices['close'].iloc[-1]
            impact_nd = (after_nd / baseline_price - 1) * 100
            
            # Volatilidade antes vs depois
            vol_before = period_prices.loc[:event_time, 'return'].std() if len(period_prices.loc[:event_time]) > 1 else 0
            vol_after = period_prices.loc[event_time:, 'return'].std() if len(period_prices.loc[event_time:]) > 1 else 0
            vol_change = ((vol_after / vol_before - 1) * 100) if vol_before > 0 else 0
            
            results.append({
                'event_id': event.get('id', ''),
                'timestamp': event_time,
                'categoria': event.get('categoria', ''),
                'severidade': event.get('severidade', ''),
                'sentimento': event.get('sentimento', ''),
                'pais': event.get('pais_nome', ''),
                'titulo': event.get('titulo', ''),
                'baseline_price': baseline_price,
                'impact_1d_pct': impact_1d,
                'impact_7d_pct': impact_nd,
                'volatility_change_pct': vol_change,
                'price_direction': 'UP' if impact_1d > 0 else 'DOWN'
            })
        
        self.impact_analysis = pd.DataFrame(results)
        return self.impact_analysis
    
    def calculate_category_statistics(self) -> pd.DataFrame:
        """Calcula estatísticas por categoria de evento"""
        if self.impact_analysis is None or self.impact_analysis.empty:
            return pd.DataFrame()
        
        stats_by_category = self.impact_analysis.groupby('categoria').agg({
            'impact_1d_pct': ['mean', 'std', 'count'],
            'impact_7d_pct': ['mean', 'std'],
            'volatility_change_pct': 'mean'
        }).round(2)
        
        stats_by_category.columns = ['_'.join(col).strip() for col in stats_by_category.columns.values]
        stats_by_category = stats_by_category.reset_index()
        
        return stats_by_category
    
    def calculate_severity_statistics(self) -> pd.DataFrame:
        """Calcula estatísticas por nível de severidade"""
        if self.impact_analysis is None or self.impact_analysis.empty:
            return pd.DataFrame()
        
        stats_by_severity = self.impact_analysis.groupby('severidade').agg({
            'impact_1d_pct': ['mean', 'std', 'count'],
            'impact_7d_pct': ['mean', 'std'],
            'volatility_change_pct': 'mean'
        }).round(2)
        
        stats_by_severity.columns = ['_'.join(col).strip() for col in stats_by_severity.columns.values]
        stats_by_severity = stats_by_severity.reset_index()
        
        return stats_by_severity
    
    def test_significance(self, category: str = None, severity: str = None) -> Dict:
        """Testa significância estatística do impacto"""
        if self.impact_analysis is None or self.impact_analysis.empty:
            return {}
        
        df = self.impact_analysis.copy()
        
        if category:
            df = df[df['categoria'] == category]
        if severity:
            df = df[df['severidade'] == severity]
        
        if len(df) < 5:
            return {'error': 'Poucos eventos para teste estatístico'}
        
        # Teste t: impacto é significativamente diferente de zero?
        t_stat_1d, p_value_1d = stats.ttest_1samp(df['impact_1d_pct'].dropna(), 0)
        t_stat_7d, p_value_7d = stats.ttest_1samp(df['impact_7d_pct'].dropna(), 0)
        
        return {
            'n_events': len(df),
            'mean_impact_1d': df['impact_1d_pct'].mean(),
            'mean_impact_7d': df['impact_7d_pct'].mean(),
            't_statistic_1d': t_stat_1d,
            'p_value_1d': p_value_1d,
            'significant_1d': p_value_1d < 0.05,
            't_statistic_7d': t_stat_7d,
            'p_value_7d': p_value_7d,
            'significant_7d': p_value_7d < 0.05
        }
    
    def plot_event_timeline(self, prices_df: pd.DataFrame, moeda_id: int = 1) -> go.Figure:
        """Plota linha do tempo de eventos sobre preços"""
        if self.impact_analysis is None or self.impact_analysis.empty:
            return go.Figure()
        
        # Filtra preços
        prices = prices_df[prices_df['moeda_id'] == moeda_id].copy()
        prices['timestamp'] = pd.to_datetime(prices['timestamp'])
        prices = prices.sort_values('timestamp')
        
        fig = go.Figure()
        
        # Linha de preços
        fig.add_trace(go.Scatter(
            x=prices['timestamp'],
            y=prices['close'],
            name='Preço',
            line=dict(color='#00E1D4', width=2)
        ))
        
        # Marca eventos
        for _, event in self.impact_analysis.iterrows():
            color = '#31fc94' if event['price_direction'] == 'UP' else '#ff6f6f'
            
            fig.add_vline(
                x=event['timestamp'],
                line=dict(color=color, width=1, dash='dash'),
                annotation_text=event['categoria'][:10],
                annotation_position="top",
                annotation=dict(font_size=8)
            )
        
        fig.update_layout(
            title='Eventos Geopolíticos vs Preço',
            xaxis_title='Data',
            yaxis_title='Preço (USD)',
            height=500,
            plot_bgcolor='#0F131A',
            paper_bgcolor='#0F131A',
            font=dict(color='#cfd8dc'),
            hovermode='x unified'
        )
        
        fig.update_yaxes(gridcolor='#1f2633')
        fig.update_xaxes(gridcolor='#1f2633')
        
        return fig
    
    def plot_impact_distribution(self) -> go.Figure:
        """Plota distribuição de impactos por categoria"""
        if self.impact_analysis is None or self.impact_analysis.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Impacto D+1 por Categoria', 'Impacto D+7 por Categoria')
        )
        
        categories = self.impact_analysis['categoria'].unique()
        
        for i, col in enumerate(['impact_1d_pct', 'impact_7d_pct'], 1):
            for cat in categories:
                data = self.impact_analysis[self.impact_analysis['categoria'] == cat][col]
                
                fig.add_trace(
                    go.Box(
                        y=data,
                        name=cat,
                        showlegend=(i == 1)
                    ),
                    row=1, col=i
                )
        
        fig.update_layout(
            height=500,
            plot_bgcolor='#0F131A',
            paper_bgcolor='#0F131A',
            font=dict(color='#cfd8dc'),
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Impacto (%)", gridcolor='#1f2633')
        fig.update_xaxes(gridcolor='#1f2633')
        
        return fig
    
    def plot_severity_impact(self) -> go.Figure:
        """Plota impacto médio por nível de severidade"""
        stats = self.calculate_severity_statistics()
        
        if stats.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=stats['severidade'],
            y=stats['impact_1d_pct_mean'],
            name='D+1',
            marker_color='#00E1D4',
            error_y=dict(type='data', array=stats['impact_1d_pct_std'])
        ))
        
        fig.add_trace(go.Bar(
            x=stats['severidade'],
            y=stats['impact_7d_pct_mean'],
            name='D+7',
            marker_color='#3A80F6',
            error_y=dict(type='data', array=stats['impact_7d_pct_std'])
        ))
        
        fig.update_layout(
            title='Impacto Médio por Severidade',
            xaxis_title='Nível de Severidade',
            yaxis_title='Impacto Médio (%)',
            barmode='group',
            height=400,
            plot_bgcolor='#0F131A',
            paper_bgcolor='#0F131A',
            font=dict(color='#cfd8dc')
        )
        
        fig.update_yaxes(gridcolor='#1f2633')
        fig.update_xaxes(gridcolor='#1f2633')
        
        return fig
    
    def generate_insights_report(self) -> str:
        """Gera relatório com insights principais"""
        if self.impact_analysis is None or self.impact_analysis.empty:
            return "Sem dados para análise"
        
        cat_stats = self.calculate_category_statistics()
        sev_stats = self.calculate_severity_statistics()
        
        # Categoria com maior impacto
        best_cat = cat_stats.loc[cat_stats['impact_7d_pct_mean'].idxmax()]
        worst_cat = cat_stats.loc[cat_stats['impact_7d_pct_mean'].idxmin()]
        
        # Severidade com maior impacto
        best_sev = sev_stats.loc[sev_stats['impact_7d_pct_mean'].idxmax()]
        
        report = f"""
🌍 ANÁLISE DE IMPACTO GEOPOLÍTICO
{'='*60}

📊 ESTATÍSTICAS GERAIS
Total de Eventos Analisados: {len(self.impact_analysis)}
Impacto Médio D+1:            {self.impact_analysis['impact_1d_pct'].mean():.2f}%
Impacto Médio D+7:            {self.impact_analysis['impact_7d_pct'].mean():.2f}%

📈 CATEGORIA COM MAIOR IMPACTO POSITIVO
Categoria:        {best_cat['categoria']}
Impacto D+7:      {best_cat['impact_7d_pct_mean']:.2f}%
N° de Eventos:    {int(best_cat['impact_1d_pct_count'])}

📉 CATEGORIA COM MAIOR IMPACTO NEGATIVO
Categoria:        {worst_cat['categoria']}
Impacto D+7:      {worst_cat['impact_7d_pct_mean']:.2f}%
N° de Eventos:    {int(worst_cat['impact_1d_pct_count'])}

⚠️ SEVERIDADE MAIS IMPACTANTE
Nível:            {best_sev['severidade']}
Impacto D+7:      {best_sev['impact_7d_pct_mean']:.2f}%
N° de Eventos:    {int(best_sev['impact_1d_pct_count'])}

💡 INSIGHTS
- Eventos de alta severidade têm impacto {abs(sev_stats[sev_stats['severidade']=='Alto']['impact_7d_pct_mean'].values[0] if 'Alto' in sev_stats['severidade'].values else 0):.1f}% maior
- Volatilidade aumenta em média {self.impact_analysis['volatility_change_pct'].mean():.1f}% após eventos
- {(self.impact_analysis['price_direction']=='UP').sum()} eventos causaram alta ({(self.impact_analysis['price_direction']=='UP').sum()/len(self.impact_analysis)*100:.1f}%)

{'='*60}
        """
        
        return report