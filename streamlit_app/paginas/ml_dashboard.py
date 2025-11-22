"""
Dashboard Interativo de Machine Learning para TCC
Mostra comparação de modelos, backtesting e análise geopolítica
"""
import os
import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sqlalchemy import create_engine, text

# Adiciona o diretório ml ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.features import FeatureEngine, prepare_train_test_split
from ml.models import CryptoPredictor, ModelComparator
from ml.backtest import Backtester
from ml.geopolitical_analysis import GeopoliticalAnalyzer


@st.cache_resource(show_spinner=False)
def get_engine():
    url = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/coinsight")
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_price_data(_engine, moeda_id: int, limit: int = 2000):
    """Carrega dados históricos de preços"""
    q = text("""
        SELECT timestamp, open, high, low, close, volume, moeda_id
        FROM precos
        WHERE moeda_id = :m
        ORDER BY timestamp DESC
        LIMIT :n
    """)
    df = pd.read_sql_query(q, _engine, params={"m": moeda_id, "n": limit})
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_events(_engine):
    """Carrega eventos geopolíticos"""
    try:
        q = text("""
            SELECT id, timestamp, pais_codigo, pais_nome, categoria, 
                   severidade, sentimento, titulo, descricao, impacto_pct
            FROM eventos_geopoliticos
            ORDER BY timestamp DESC
            LIMIT 500
        """)
        df = pd.read_sql_query(q, _engine)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        return df
    except:
        return pd.DataFrame()


def show():
    # Header
    st.markdown("""
        <div style="margin:4px 0 20px 0;">
          <h1 style="margin:0;
              background: linear-gradient(90deg, #00E1D4, #3A80F6, #C64BD9);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent;
              font-weight:800; font-size:48px;">🤖 Dashboard de Inteligência Artificial</h1>
          <p style="color:#9db1cb;margin:6px 0 0 0;font-size:16px;">
            Comparação de Modelos, Backtesting e Análise Geopolítica
          </p>
        </div>
    """, unsafe_allow_html=True)
    
    engine = get_engine()
    
    # Sidebar: Configurações
    st.sidebar.markdown("### ⚙️ Configurações")
    
    moedas = {
        "BTC (Bitcoin)": 1,
        "ETH (Ethereum)": 2,
        "ADA (Cardano)": 3,
        "SOL (Solana)": 4
    }
    
    moeda_nome = st.sidebar.selectbox("Criptomoeda", list(moedas.keys()))
    moeda_id = moedas[moeda_nome]
    
    task_type = st.sidebar.radio("Tipo de Previsão", ["Regressão (Retorno)", "Classificação (Direção)"])
    task = 'regression' if 'Regressão' in task_type else 'classification'
    
    test_size = st.sidebar.slider("Tamanho do Conjunto de Teste (%)", 10, 40, 20) / 100
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Comparação de Modelos", 
        "📈 Backtesting", 
        "🌍 Impacto Geopolítico",
        "🎯 Previsão ao Vivo"
    ])
    
    # ========== TAB 1: COMPARAÇÃO DE MODELOS ==========
    with tab1:
        st.markdown("### 🔍 Treinamento e Comparação de Modelos")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("💡 **Dica:** Treine múltiplos modelos para comparar performance. O sistema usa validação temporal rigorosa (sem data leakage).")
        
        with col2:
            if st.button("🚀 Treinar Todos os Modelos", use_container_width=True):
                with st.spinner("Carregando dados..."):
                    df_prices = load_price_data(engine, moeda_id, limit=2000)
                
                if len(df_prices) < 100:
                    st.error("Dados insuficientes. Execute o ETL primeiro.")
                    return
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Feature Engineering
                status_text.text("🔧 Criando features...")
                progress_bar.progress(10)
                
                fe = FeatureEngine()
                df_features = fe.create_all_features(df_prices)
                df_with_target, target = fe.create_target(df_features, horizon=1, target_type=task)
                
                # Split temporal
                status_text.text("📊 Dividindo dados temporalmente...")
                progress_bar.progress(20)
                
                X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_test_split(
                    pd.concat([df_with_target, target], axis=1),
                    test_size=test_size,
                    val_size=0.15
                )
                
                st.session_state['X_train'] = X_train
                st.session_state['X_val'] = X_val
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_val'] = y_val
                st.session_state['y_test'] = y_test
                st.session_state['df_prices'] = df_prices
                
                # Treina modelos
                status_text.text("🤖 Treinando modelos de IA...")
                progress_bar.progress(30)
                
                comparator = ModelComparator(task=task)
                comparator.add_model("Random Forest", "random_forest")
                
                try:
                    comparator.add_model("XGBoost", "xgboost")
                except:
                    st.warning("XGBoost não disponível")
                
                try:
                    comparator.add_model("LightGBM", "lightgbm")
                except:
                    st.warning("LightGBM não disponível")
                
                # Treina
                progress_bar.progress(50)
                comparator.train_all(X_train, y_train, X_val, y_val)
                
                # Avalia
                status_text.text("📊 Avaliando modelos...")
                progress_bar.progress(80)
                
                results = comparator.evaluate_all(X_test, y_test)
                st.session_state['comparator'] = comparator
                st.session_state['results'] = results
                
                progress_bar.progress(100)
                status_text.text("✅ Concluído!")
                st.success("🎉 Treinamento concluído com sucesso!")
        
        # Mostra resultados se já treinados
        if 'results' in st.session_state and not st.session_state['results'].empty:
            st.markdown("---")
            st.markdown("### 📊 Resultados da Comparação")
            
            results = st.session_state['results']
            
            # Tabela de métricas
            st.dataframe(
                results.style.highlight_max(axis=0, color='lightgreen') if task == 'classification'
                else results.style.highlight_min(axis=0, subset=['mae', 'rmse', 'mape'], color='lightgreen'),
                use_container_width=True,
                height=200
            )
            
            # Gráficos de comparação
            col1, col2 = st.columns(2)
            
            with col1:
                if task == 'regression':
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=results.index,
                        y=results['mae'],
                        name='MAE',
                        marker_color='#00E1D4'
                    ))
                    fig.update_layout(
                        title="MAE por Modelo (menor é melhor)",
                        height=350,
                        plot_bgcolor='#0F131A',
                        paper_bgcolor='#0F131A',
                        font=dict(color='#cfd8dc')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=results.index,
                        y=results['f1'] * 100,
                        name='F1-Score',
                        marker_color='#3A80F6'
                    ))
                    fig.update_layout(
                        title="F1-Score por Modelo (%)",
                        height=350,
                        plot_bgcolor='#0F131A',
                        paper_bgcolor='#0F131A',
                        font=dict(color='#cfd8dc')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if task == 'regression':
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=results.index,
                        y=results['directional_accuracy'] * 100,
                        name='Acurácia Direcional',
                        marker_color='#C64BD9'
                    ))
                    fig.update_layout(
                        title="Acurácia Direcional (%)",
                        height=350,
                        plot_bgcolor='#0F131A',
                        paper_bgcolor='#0F131A',
                        font=dict(color='#cfd8dc')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=results.index,
                        y=results['accuracy'] * 100,
                        name='Acurácia',
                        marker_color='#31fc94'
                    ))
                    fig.update_layout(
                        title="Acurácia por Modelo (%)",
                        height=350,
                        plot_bgcolor='#0F131A',
                        paper_bgcolor='#0F131A',
                        font=dict(color='#cfd8dc')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            st.markdown("### 🎯 Features Mais Importantes")
            
            comparator = st.session_state['comparator']
            best_model_name = results['mae'].idxmin() if task == 'regression' else results['f1'].idxmax()
            best_model = comparator.models[best_model_name]
            
            importance_df = best_model.get_feature_importance(top_n=15)
            
            if not importance_df.empty and 'feature' in importance_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=importance_df['importance'],
                    y=importance_df['feature'],
                    orientation='h',
                    marker_color='#00E1D4'
                ))
                fig.update_layout(
                    title=f"Top 15 Features - {best_model_name}",
                    height=500,
                    plot_bgcolor='#0F131A',
                    paper_bgcolor='#0F131A',
                    font=dict(color='#cfd8dc')
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 2: BACKTESTING ==========
    with tab2:
        st.markdown("### 📈 Backtesting: Validação Histórica")
        
        if 'comparator' not in st.session_state:
            st.warning("⚠️ Treine os modelos primeiro na aba 'Comparação de Modelos'")
        else:
            st.info("💡 **O que é Backtesting?** Testamos o modelo em dados passados para ver se as previsões teriam sido lucrativas.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                initial_capital = st.number_input("Capital Inicial (USD)", 1000, 100000, 10000, step=1000)
            
            with col2:
                transaction_cost = st.number_input("Custo de Transação (%)", 0.0, 1.0, 0.1, step=0.05) / 100
            
            with col3:
                strategy = st.selectbox("Estratégia", ["Long/Short", "Long Only"])
                strategy_code = 'long_short' if strategy == "Long/Short" else 'long_only'
            
            if st.button("🔄 Executar Backtest", use_container_width=True):
                with st.spinner("Executando backtest..."):
                    # Pega melhor modelo
                    comparator = st.session_state['comparator']
                    results = st.session_state['results']
                    best_model_name = results['mae'].idxmin() if task == 'regression' else results['f1'].idxmax()
                    best_model = comparator.models[best_model_name]
                    
                    # Faz previsões no conjunto de teste
                    X_test = st.session_state['X_test']
                    y_test = st.session_state['y_test']
                    
                    predictions = best_model.predict(X_test)
                    
                    # Pega timestamps e preços correspondentes
                    df_prices = st.session_state['df_prices']
                    test_start_idx = len(st.session_state['X_train']) + len(st.session_state['X_val'])
                    
                    # Ajusta índices
                    test_timestamps = df_prices.iloc[test_start_idx:test_start_idx+len(predictions)]['timestamp'].reset_index(drop=True)
                    test_prices = df_prices.iloc[test_start_idx:test_start_idx+len(predictions)]['close'].reset_index(drop=True)
                    
                    # Executa backtest
                    backtester = Backtester(
                        initial_capital=initial_capital,
                        transaction_cost=transaction_cost
                    )
                    
                    backtest_results = backtester.run_backtest(
                        predictions=predictions,
                        actual_prices=test_prices,
                        timestamps=test_timestamps,
                        strategy=strategy_code
                    )
                    
                    st.session_state['backtest_results'] = backtest_results
                    st.session_state['backtester'] = backtester
                    
                st.success("✅ Backtest concluído!")
            
            # Mostra resultados
            if 'backtester' in st.session_state:
                backtester = st.session_state['backtester']
                
                # Métricas
                metrics = backtester.calculate_metrics()
                
                st.markdown("### 💰 Resultados Financeiros")
                
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric(
                    "Retorno Total",
                    f"{metrics['total_return_strategy']:.2f}%",
                    delta=f"{metrics['total_return_strategy'] - metrics['total_return_buy_hold']:.2f}% vs B&H"
                )
                
                col2.metric(
                    "Sharpe Ratio",
                    f"{metrics['sharpe_strategy']:.2f}",
                    delta=f"{metrics['sharpe_strategy'] - metrics['sharpe_buy_hold']:.2f} vs B&H"
                )
                
                col3.metric(
                    "Max Drawdown",
                    f"{metrics['max_drawdown_strategy']:.2f}%",
                    delta=f"{metrics['max_drawdown_strategy'] - metrics['max_drawdown_buy_hold']:.2f}% vs B&H",
                    delta_color="inverse"
                )
                
                col4.metric(
                    "Win Rate",
                    f"{metrics['win_rate']:.1f}%",
                    delta=f"{metrics['total_trades']} trades"
                )
                
                # Gráficos
                st.markdown("### 📊 Visualizações")
                
                fig_bt = backtester.plot_results(f"Backtest - {moeda_nome}")
                st.plotly_chart(fig_bt, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_dd = backtester.plot_drawdown()
                    st.plotly_chart(fig_dd, use_container_width=True)
                
                with col2:
                    # Relatório
                    st.markdown("### 📄 Relatório Completo")
                    st.text(backtester.generate_report())
    
    # ========== TAB 3: IMPACTO GEOPOLÍTICO ==========
    with tab3:
        st.markdown("### 🌍 Correlação: Eventos Geopolíticos × Preços")
        
        df_events = load_events(engine)
        df_prices = load_price_data(engine, moeda_id, limit=2000)
        
        if df_events.empty:
            st.warning("⚠️ Sem eventos geopolíticos na base. Execute: python populate_geopolitical_events.py")
        else:
            st.info(f"📊 Analisando {len(df_events)} eventos geopolíticos")
            
            if st.button("🔍 Analisar Correlações", use_container_width=True):
                with st.spinner("Analisando impacto dos eventos..."):
                    analyzer = GeopoliticalAnalyzer()
                    
                    impact_df = analyzer.analyze_event_impact(
                        events_df=df_events,
                        prices_df=df_prices,
                        moeda_id=moeda_id,
                        window_before=3,
                        window_after=7
                    )
                    
                    st.session_state['geo_analyzer'] = analyzer
                    st.session_state['impact_df'] = impact_df
                
                st.success("✅ Análise concluída!")
            
            if 'geo_analyzer' in st.session_state:
                analyzer = st.session_state['geo_analyzer']
                
                # Insights
                st.markdown("### 💡 Insights Principais")
                st.text(analyzer.generate_insights_report())
                
                # Visualizações
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = analyzer.plot_impact_distribution()
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = analyzer.plot_severity_impact()
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Timeline
                fig3 = analyzer.plot_event_timeline(df_prices, moeda_id)
                st.plotly_chart(fig3, use_container_width=True)
                
                # Tabela detalhada
                with st.expander("📋 Ver Análise Detalhada por Evento"):
                    st.dataframe(st.session_state['impact_df'], use_container_width=True)
    
    # ========== TAB 4: PREVISÃO AO VIVO ==========
    with tab4:
        st.markdown("### 🎯 Faça uma Previsão Agora")
        
        if 'comparator' not in st.session_state:
            st.warning("⚠️ Treine os modelos primeiro")
        else:
            st.info("💡 Use os dados mais recentes para prever o próximo período")
            
            if st.button("🔮 Gerar Previsão", use_container_width=True):
                with st.spinner("Gerando previsão..."):
                    # Carrega dados recentes
                    df_prices = load_price_data(engine, moeda_id, limit=500)
                    
                    # Feature Engineering
                    fe = FeatureEngine()
                    df_features = fe.create_all_features(df_prices)
                    
                    # Pega última linha (mais recente)
                    last_features = df_features[fe.get_feature_names(df_features)].iloc[[-1]]
                    last_price = df_prices['close'].iloc[-1]
                    last_timestamp = df_prices['timestamp'].iloc[-1]
                    
                    # Faz previsão com todos os modelos
                    comparator = st.session_state['comparator']
                    
                    predictions = {}
                    for name, model in comparator.models.items():
                        pred = model.predict(last_features)[0]
                        predictions[name] = pred
                    
                    # Média ensemble
                    ensemble_pred = np.mean(list(predictions.values()))
                    
                    st.success("✅ Previsão gerada!")
                    
                    # Mostra resultados
                    st.markdown("### 📊 Previsões dos Modelos")
                    
                    cols = st.columns(len(predictions) + 1)
                    
                    for i, (name, pred) in enumerate(predictions.items()):
                        if task == 'regression':
                            pred_pct = pred * 100
                            next_price = last_price * (1 + pred)
                            cols[i].metric(
                                name,
                                f"${next_price:.2f}",
                                delta=f"{pred_pct:+.2f}%"
                            )
                        else:
                            direction = "📈 ALTA" if pred > 0.5 else "📉 BAIXA"
                            confidence = max(pred, 1-pred) * 100
                            cols[i].metric(
                                name,
                                direction,
                                delta=f"{confidence:.1f}% confiança"
                            )
                    
                    # Ensemble
                    if task == 'regression':
                        ens_pct = ensemble_pred * 100
                        ens_price = last_price * (1 + ensemble_pred)
                        cols[-1].metric(
                            "🎯 ENSEMBLE",
                            f"${ens_price:.2f}",
                            delta=f"{ens_pct:+.2f}%"
                        )
                    else:
                        ens_dir = "📈 ALTA" if ensemble_pred > 0.5 else "📉 BAIXA"
                        ens_conf = max(ensemble_pred, 1-ensemble_pred) * 100
                        cols[-1].metric(
                            "🎯 ENSEMBLE",
                            ens_dir,
                            delta=f"{ens_conf:.1f}% confiança"
                        )
                    
                    st.markdown("---")
                    st.caption(f"📅 Última atualização: {last_timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
                    st.caption(f"💵 Preço atual: ${last_price:.2f}")


if __name__ == "__main__":
    show()