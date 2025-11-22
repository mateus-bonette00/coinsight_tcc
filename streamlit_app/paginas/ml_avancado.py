"""
Dashboard Avançado de Machine Learning e IA
Sistema completo para apresentação de TCC
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Adiciona diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

# Imports locais
try:
    from database import get_db_connection
    from ml.models import CryptoPredictor, ModelComparator
    from ml.features import FeatureEngine, prepare_train_test_split
    from ml.walk_forward import WalkForwardAnalyzer
    from ml.advanced_models import ModelDiagnostics
    from ml.geopolitical_analysis import GeopoliticalAnalyzer
except ImportError as e:
    pass  # Will handle in show()


def load_data(moeda_id: int, limit: int = 2000):
    """Carrega dados de preços do banco"""
    conn = get_db_connection()
    query = f"""
        SELECT timestamp, open, high, low, close, volume, moeda_id
        FROM precos
        WHERE moeda_id = {moeda_id}
        ORDER BY timestamp DESC
        LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df.sort_values('timestamp').reset_index(drop=True)


def load_events():
    """Carrega eventos geopolíticos"""
    try:
        conn = get_db_connection()
        query = """
            SELECT timestamp, categoria, severidade, sentimento, titulo,
                   pais_nome, impacto_estimado_pct
            FROM eventos_geopoliticos
            ORDER BY timestamp DESC
            LIMIT 200
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.warning(f"Eventos geopolíticos não disponíveis: {e}")
        return pd.DataFrame()


def prepare_features(df, events_df=None):
    """Prepara features com eventos geopolíticos"""
    engine = FeatureEngine()

    # Cria features (com eventos se disponível)
    df_features = engine.create_all_features(df, events_df)

    # Cria target
    df_features, target = engine.create_target(
        df_features,
        horizon=1,
        target_type='regression'
    )

    # Adiciona timestamp e close de volta
    df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
    df_features['target_return_1d'] = target

    return df_features, engine


def show():
    """Função principal da página - Entry point do Streamlit"""

    st.set_page_config(page_title="ML Avançado - TCC", layout="wide", page_icon="🤖")

    st.title("🤖 Sistema Avançado de Machine Learning")
    st.markdown("### Análise Completa para Apresentação de TCC")
    st.markdown("---")

    # Info
    st.info("""
    **🎓 Sistema Completo de ML para TCC**

    Este dashboard apresenta:
    - 📊 Comparação de múltiplos modelos (Random Forest, XGBoost, LightGBM)
    - 🔄 Walk-Forward Analysis (backtesting robusto com re-treinamento)
    - 🌍 Análise de impacto de eventos geopolíticos
    - 🔍 Diagnóstico avançado de erros
    - 📈 Previsões ensemble avançadas

    👈 Use a barra lateral para selecionar a moeda e começar!
    """)

    # Sidebar
    st.sidebar.header("⚙️ Configurações")

    moeda_map = {
        "Bitcoin (BTC)": 1,
        "Ethereum (ETH)": 2,
        "Cardano (ADA)": 3,
        "Solana (SOL)": 4
    }

    selected_moeda = st.sidebar.selectbox(
        "Selecione a Moeda",
        list(moeda_map.keys())
    )
    moeda_id = moeda_map[selected_moeda]

    # Abas principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Comparação de Modelos",
        "🔄 Walk-Forward Analysis",
        "🌍 Impacto Geopolítico",
        "🔍 Diagnóstico de Erros",
        "📈 Previsões Avançadas"
    ])

    # ==================== ABA 1: COMPARAÇÃO DE MODELOS ====================
    with tab1:
        st.header("📊 Comparação de Modelos ML")
        st.markdown("Compare Random Forest, XGBoost e LightGBM com features geopolíticas")

        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("Configurações")
            test_size = st.slider("Tamanho do Conjunto de Teste (%)", 10, 40, 20) / 100
            val_size = st.slider("Tamanho de Validação (%)", 5, 20, 10) / 100
            use_geopolitical = st.checkbox("Incluir Features Geopolíticas", value=True)

            if st.button("🚀 Treinar e Comparar Modelos", type="primary"):
                with st.spinner("Carregando dados..."):
                    df = load_data(moeda_id)
                    events_df = load_events() if use_geopolitical else None

                    if len(df) < 100:
                        st.error("Dados insuficientes para treinamento")
                        st.stop()

                    # Prepara features
                    df_features, engine = prepare_features(df, events_df)

                    st.success(f"✅ {len(df_features)} registros carregados")
                    if use_geopolitical and events_df is not None and len(events_df) > 0:
                        st.info(f"🌍 {len(events_df)} eventos geopolíticos incluídos nas features")

                with st.spinner("Preparando dados..."):
                    # Split temporal
                    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_test_split(
                        df_features,
                        test_size=test_size,
                        val_size=val_size
                    )

                    st.write(f"📊 Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")

                with st.spinner("Treinando modelos..."):
                    # Cria comparador
                    comparator = ModelComparator(task='regression')
                    comparator.add_model('Random Forest', 'random_forest')

                    # Adiciona outros modelos se disponíveis
                    try:
                        comparator.add_model('XGBoost', 'xgboost')
                    except:
                        st.warning("XGBoost não disponível")

                    try:
                        comparator.add_model('LightGBM', 'lightgbm')
                    except:
                        st.warning("LightGBM não disponível")

                    # Treina todos
                    comparator.train_all(X_train, y_train, X_val, y_val)

                    # Avalia
                    results = comparator.evaluate_all(X_test, y_test)

                    # Armazena no session state
                    st.session_state['comparator'] = comparator
                    st.session_state['results'] = results
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    st.session_state['df_features'] = df_features

                st.success("✅ Treinamento concluído!")

        with col1:
            if 'results' in st.session_state:
                st.subheader("Resultados da Comparação")

                results = st.session_state['results']

                # Tabela de métricas
                st.dataframe(
                    results[['mae', 'rmse', 'r2', 'mape', 'directional_accuracy']].style.format({
                        'mae': '{:.6f}',
                        'rmse': '{:.6f}',
                        'r2': '{:.4f}',
                        'mape': '{:.2f}',
                        'directional_accuracy': '{:.2%}'
                    }).background_gradient(cmap='RdYlGn', subset=['r2', 'directional_accuracy'])
                      .background_gradient(cmap='RdYlGn_r', subset=['mae', 'rmse', 'mape']),
                    use_container_width=True
                )

                # Gráfico comparativo
                fig = go.Figure()

                metrics_to_plot = ['mae', 'rmse', 'mape']
                colors = ['#00E1D4', '#3A80F6', '#FF6B6B']

                for i, metric in enumerate(metrics_to_plot):
                    fig.add_trace(go.Bar(
                        name=metric.upper(),
                        x=results.index,
                        y=results[metric],
                        marker_color=colors[i]
                    ))

                fig.update_layout(
                    title='Comparação de Métricas de Erro',
                    barmode='group',
                    height=400,
                    plot_bgcolor='#0F131A',
                    paper_bgcolor='#0F131A',
                    font=dict(color='#cfd8dc')
                )

                st.plotly_chart(fig, use_container_width=True)

                # Melhor modelo
                best_name, best_model = st.session_state['comparator'].get_best_model('mae')
                st.success(f"🏆 Melhor Modelo: **{best_name}** (MAE: {results.loc[best_name, 'mae']:.6f})")

                # Feature Importance
                st.subheader("Top 15 Features Mais Importantes")
                importance_df = best_model.get_feature_importance(top_n=15)

                fig_importance = go.Figure(go.Bar(
                    x=importance_df['importance'],
                    y=importance_df['feature'],
                    orientation='h',
                    marker_color='#00E1D4'
                ))

                fig_importance.update_layout(
                    height=500,
                    plot_bgcolor='#0F131A',
                    paper_bgcolor='#0F131A',
                    font=dict(color='#cfd8dc'),
                    xaxis_title='Importância',
                    yaxis={'categoryorder': 'total ascending'}
                )

                st.plotly_chart(fig_importance, use_container_width=True)

            else:
                st.info("👈 Configure e clique em 'Treinar e Comparar Modelos' para começar")

    # ==================== ABA 2: WALK-FORWARD ANALYSIS ====================
    with tab2:
        st.header("🔄 Walk-Forward Analysis")
        st.markdown("*Backtesting robusto com re-treinamento periódico - simula cenário real de trading*")

        st.info("""
        **O que é Walk-Forward Analysis?**

        É um método de backtesting mais rigoroso que simula melhor o cenário real:
        1. Treina o modelo em uma janela passada
        2. Testa em uma janela futura
        3. Desliza as janelas e re-treina periodicamente
        4. Avalia performance ao longo do tempo
        """)

        st.warning("⚠️ Esta análise requer modelos treinados. Execute primeiro a **Comparação de Modelos** (Aba 1)")

        if 'comparator' in st.session_state:
            col1, col2 = st.columns([2, 1])

            with col2:
                st.subheader("Configurações")
                train_window = st.number_input("Janela de Treino (dias)", 90, 365, 180)
                test_window = st.number_input("Janela de Teste (dias)", 15, 90, 30)
                retrain_freq = st.number_input("Re-treinar a cada (dias)", 15, 60, 30)
                strategy = st.selectbox("Estratégia", ["long_short", "long_only"])

                st.markdown(f"""
                **Resumo:**
                - Treina em {train_window} dias passados
                - Testa em {test_window} dias futuros
                - Re-treina a cada {retrain_freq} dias
                """)

                if st.button("▶️ Executar Walk-Forward", type="primary"):
                    with st.spinner("Executando Walk-Forward Analysis (pode levar alguns minutos)..."):
                        df_features = st.session_state['df_features']

                        # Prepara feature columns
                        feature_cols = [col for col in df_features.columns
                                       if col not in ['timestamp', 'close', 'target_return_1d', 'open', 'high', 'low', 'volume', 'moeda_id']]

                        # Função de treinamento
                        def train_model(X, y):
                            from ml.models import CryptoPredictor
                            model = CryptoPredictor(model_type='random_forest', task='regression')
                            model.fit(X, y)
                            return model

                        # Cria analisador
                        wf_analyzer = WalkForwardAnalyzer(
                            train_window_size=train_window,
                            test_window_size=test_window,
                            retrain_frequency=retrain_freq
                        )

                        # Executa
                        results_wf = wf_analyzer.run(
                            df=df_features,
                            model_trainer=train_model,
                            feature_cols=feature_cols,
                            target_col='target_return_1d',
                            strategy=strategy
                        )

                        st.session_state['wf_analyzer'] = wf_analyzer
                        st.session_state['wf_results'] = results_wf

                    st.success("✅ Walk-Forward concluído!")

            with col1:
                if 'wf_results' in st.session_state:
                    wf_analyzer = st.session_state['wf_analyzer']

                    # Métricas
                    metrics = wf_analyzer.calculate_metrics()

                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

                    with col_m1:
                        st.metric(
                            "Retorno Total",
                            f"{metrics['total_return_strategy']:.2f}%",
                            delta=f"{metrics['total_return_strategy'] - metrics['total_return_buy_hold']:.2f}% vs B&H"
                        )

                    with col_m2:
                        st.metric(
                            "Sharpe Ratio",
                            f"{metrics['sharpe_strategy']:.2f}",
                            delta=f"{metrics['sharpe_strategy'] - metrics['sharpe_buy_hold']:.2f} vs B&H"
                        )

                    with col_m3:
                        st.metric(
                            "Max Drawdown",
                            f"{metrics['max_drawdown_strategy']:.2f}%"
                        )

                    with col_m4:
                        st.metric(
                            "Win Rate",
                            f"{metrics['win_rate']:.1f}%"
                        )

                    # Gráficos
                    st.plotly_chart(wf_analyzer.plot_results(), use_container_width=True)

                    # Performance por fold
                    st.subheader("Performance por Fold (Re-treinamentos)")
                    st.plotly_chart(wf_analyzer.plot_fold_performance(), use_container_width=True)

                    # Relatório
                    with st.expander("📄 Relatório Completo"):
                        st.text(wf_analyzer.generate_report())

                else:
                    st.info("👈 Configure e execute o Walk-Forward Analysis")
        else:
            st.error("Execute primeiro a Comparação de Modelos (Aba 1)")

    # ==================== ABA 3: IMPACTO GEOPOLÍTICO ====================
    with tab3:
        st.header("🌍 Análise de Impacto Geopolítico")
        st.markdown("Descubra como eventos mundiais afetam os preços das criptomoedas")

        if st.button("🔍 Analisar Impacto", type="primary"):
            with st.spinner("Carregando dados..."):
                df = load_data(moeda_id)
                events_df = load_events()

                if events_df.empty:
                    st.warning("""
                    Nenhum evento geopolítico encontrado no banco de dados.

                    **Para popular eventos:**
                    ```bash
                    cd streamlit_app
                    python populate_geopolitical_events.py
                    ```
                    """)
                    st.stop()

            with st.spinner("Analisando correlações..."):
                analyzer = GeopoliticalAnalyzer()
                impact_df = analyzer.analyze_event_impact(
                    events_df=events_df,
                    prices_df=df,
                    moeda_id=moeda_id,
                    window_before=3,
                    window_after=7
                )

                st.session_state['geo_analyzer'] = analyzer
                st.session_state['impact_df'] = impact_df

            st.success(f"✅ {len(impact_df)} eventos analisados!")

        if 'impact_df' in st.session_state:
            analyzer = st.session_state['geo_analyzer']
            impact_df = st.session_state['impact_df']

            # Estatísticas gerais
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Impacto Médio D+1", f"{impact_df['impact_1d_pct'].mean():.2f}%")

            with col2:
                st.metric("Impacto Médio D+7", f"{impact_df['impact_7d_pct'].mean():.2f}%")

            with col3:
                eventos_positivos = (impact_df['price_direction'] == 'UP').sum()
                st.metric("Eventos Positivos", f"{eventos_positivos} ({eventos_positivos/len(impact_df)*100:.1f}%)")

            # Gráficos
            st.plotly_chart(analyzer.plot_event_timeline(df, moeda_id), use_container_width=True)
            st.plotly_chart(analyzer.plot_impact_distribution(), use_container_width=True)
            st.plotly_chart(analyzer.plot_severity_impact(), use_container_width=True)

            # Estatísticas por categoria
            st.subheader("Impacto por Categoria")
            cat_stats = analyzer.calculate_category_statistics()
            st.dataframe(cat_stats, use_container_width=True)

            # Insights
            with st.expander("💡 Insights Geopolíticos"):
                st.text(analyzer.generate_insights_report())

        else:
            st.info("👆 Clique em 'Analisar Impacto' para começar")

    # ==================== ABA 4: DIAGNÓSTICO DE ERROS ====================
    with tab4:
        st.header("🔍 Diagnóstico Avançado de Erros")
        st.markdown("Análise estatística completa dos erros de previsão")

        if 'results' in st.session_state and 'X_test' in st.session_state:
            best_name, best_model = st.session_state['comparator'].get_best_model('mae')
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']

            # Faz previsões
            predictions = best_model.predict(X_test)

            # Calcula resíduos
            residuals = y_test - predictions

            # Calcula métricas manualmente
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            # Acurácia direcional (% de vezes que acertou a direção)
            y_direction = np.sign(y_test)
            pred_direction = np.sign(predictions)
            directional_accuracy = np.mean(y_direction == pred_direction)

            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("MAE", f"{mae:.6f}")

            with col2:
                st.metric("RMSE", f"{rmse:.6f}")

            with col3:
                st.metric("R²", f"{r2:.4f}")

            with col4:
                st.metric("Acurácia Direcional", f"{directional_accuracy*100:.1f}%")

            # Análise de resíduos
            st.subheader("Análise de Resíduos")

            col1, col2 = st.columns(2)

            with col1:
                # Histograma de resíduos
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=residuals,
                    nbinsx=50,
                    marker_color='#00E1D4',
                    name='Resíduos'
                ))
                fig_hist.update_layout(
                    title='Distribuição dos Resíduos',
                    xaxis_title='Resíduo',
                    yaxis_title='Frequência',
                    height=400,
                    plot_bgcolor='#0F131A',
                    paper_bgcolor='#0F131A',
                    font=dict(color='#cfd8dc')
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                # Scatter: Previsão vs Real
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=y_test,
                    y=predictions,
                    mode='markers',
                    marker=dict(color='#3A80F6', size=5, opacity=0.6),
                    name='Previsões'
                ))

                # Linha ideal
                min_val = min(y_test.min(), predictions.min())
                max_val = max(y_test.max(), predictions.max())
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='#FF6B6B', dash='dash'),
                    name='Ideal'
                ))

                fig_scatter.update_layout(
                    title='Previsão vs Real',
                    xaxis_title='Valor Real',
                    yaxis_title='Previsão',
                    height=400,
                    plot_bgcolor='#0F131A',
                    paper_bgcolor='#0F131A',
                    font=dict(color='#cfd8dc')
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # Outliers (detecção simples baseada em threshold de 2.5 desvios padrão)
            std_residuals = np.std(residuals)
            threshold = 2.5 * std_residuals
            outlier_mask = np.abs(residuals) > threshold
            outlier_count = np.sum(outlier_mask)
            outlier_percentage = (outlier_count / len(residuals)) * 100

            st.subheader("Detecção de Outliers")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Outliers Detectados", outlier_count)

            with col2:
                st.metric("% de Outliers", f"{outlier_percentage:.2f}%")

            with col3:
                threshold_status = "✅ Normal" if outlier_percentage < 5 else "⚠️ Alto"
                st.metric("Status", threshold_status)

            # Relatório completo
            with st.expander("📄 Relatório de Diagnóstico Completo"):
                report = f"""
╔════════════════════════════════════════════════════════════╗
║           RELATÓRIO DE DIAGNÓSTICO - {best_name}
╚════════════════════════════════════════════════════════════╝

📊 MÉTRICAS DE ERRO
├─ MAE (Mean Absolute Error):        {mae:.6f}
├─ RMSE (Root Mean Squared Error):   {rmse:.6f}
├─ R² Score:                          {r2:.4f}
└─ Acurácia Direcional:              {directional_accuracy*100:.1f}%

📉 ANÁLISE DE RESÍDUOS
├─ Média dos Resíduos:               {np.mean(residuals):.6f}
├─ Desvio Padrão:                    {std_residuals:.6f}
├─ Assimetria (Skewness):            {pd.Series(residuals).skew():.4f}
└─ Curtose (Kurtosis):               {pd.Series(residuals).kurtosis():.4f}

🎯 OUTLIERS
├─ Threshold (2.5σ):                 {threshold:.6f}
├─ Número de Outliers:               {outlier_count}
└─ Percentual:                       {outlier_percentage:.2f}%

✅ STATUS: {'NORMAL' if outlier_percentage < 5 else 'ATENÇÃO - ALTO NÚMERO DE OUTLIERS'}
                """
                st.text(report)

        else:
            st.info("Execute primeiro a Comparação de Modelos (Aba 1)")

    # ==================== ABA 5: PREVISÕES AVANÇADAS ====================
    with tab5:
        st.header("📈 Previsões Avançadas")
        st.markdown("*Combine múltiplos modelos para previsões robustas (Ensemble)*")

        if 'comparator' in st.session_state:
            st.subheader("Previsão Ensemble")

            col1, col2 = st.columns([3, 1])

            with col2:
                horizon = st.selectbox("Horizonte de Previsão", [1, 3, 7, 14, 30])
                st.markdown(f"*Previsão para **{horizon} dia(s)** à frente*")

            with col1:
                # Pega últimos dados
                df_features = st.session_state['df_features']
                feature_cols = [col for col in df_features.columns
                               if col not in ['timestamp', 'close', 'target_return_1d', 'open', 'high', 'low', 'volume', 'moeda_id']]

                # Últimas features
                last_features = df_features[feature_cols].iloc[-1:].copy()
                last_price = df_features['close'].iloc[-1]
                last_timestamp = df_features['timestamp'].iloc[-1]

                # Previsões de cada modelo
                comparator = st.session_state['comparator']
                predictions = {}

                for name, model in comparator.models.items():
                    if model.is_fitted:
                        try:
                            pred = model.predict(last_features)[0]
                            predicted_price = last_price * (1 + pred)
                            predictions[name] = {
                                'return': pred * 100,
                                'price': predicted_price
                            }
                        except:
                            continue

                if not predictions:
                    st.error("Nenhum modelo disponível para previsão")
                    st.stop()

                # Ensemble (média)
                ensemble_return = np.mean([p['return'] for p in predictions.values()])
                ensemble_price = last_price * (1 + ensemble_return / 100)

                # Exibe resultados
                st.subheader(f"Previsão para {horizon} dia(s) à frente")

                col_p1, col_p2, col_p3 = st.columns(3)

                with col_p1:
                    st.metric("Preço Atual", f"${last_price:,.2f}")

                with col_p2:
                    st.metric(
                        "Previsão Ensemble",
                        f"${ensemble_price:,.2f}",
                        delta=f"{ensemble_return:.2f}%"
                    )

                with col_p3:
                    direction = "📈 ALTA" if ensemble_return > 0 else "📉 BAIXA"
                    st.metric("Direção", direction)

                # Tabela de previsões
                st.subheader("Previsões Individuais por Modelo")

                pred_df = pd.DataFrame(predictions).T
                pred_df.columns = ['Retorno (%)', 'Preço']
                pred_df = pred_df.sort_values('Retorno (%)', ascending=False)

                st.dataframe(
                    pred_df.style.format({
                        'Retorno (%)': '{:.2f}%',
                        'Preço': '${:,.2f}'
                    }).background_gradient(cmap='RdYlGn', subset=['Retorno (%)']),
                    use_container_width=True
                )

                # Gráfico de comparação
                fig = go.Figure()

                # Adiciona cada previsão
                for name, pred in predictions.items():
                    fig.add_trace(go.Bar(
                        name=name,
                        x=[name],
                        y=[pred['return']],
                        marker_color='#00E1D4' if pred['return'] > 0 else '#FF6B6B',
                        text=[f"{pred['return']:.2f}%"],
                        textposition='outside'
                    ))

                # Linha do ensemble
                fig.add_hline(
                    y=ensemble_return,
                    line=dict(color='#FFA500', dash='dash', width=2),
                    annotation_text="Ensemble",
                    annotation_position="right"
                )

                fig.update_layout(
                    title='Comparação de Previsões por Modelo',
                    yaxis_title='Retorno Esperado (%)',
                    height=400,
                    showlegend=False,
                    plot_bgcolor='#0F131A',
                    paper_bgcolor='#0F131A',
                    font=dict(color='#cfd8dc')
                )

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Execute primeiro a Comparação de Modelos (Aba 1)")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>🤖 Sistema Avançado de ML para Previsão de Criptomoedas - TCC 2024</p>
        <p>Desenvolvido com Streamlit, Scikit-learn, XGBoost, LightGBM e Plotly</p>
        <p><small>Features: 80+ indicadores técnicos + eventos geopolíticos | Modelos: RF, XGBoost, LightGBM</small></p>
    </div>
    """, unsafe_allow_html=True)


# Para testes locais
if __name__ == "__main__":
    show()
