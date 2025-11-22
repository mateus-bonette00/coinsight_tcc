# 🤖 Guia Completo - Sistema Avançado de ML para TCC

## 📋 Sumário
- [Visão Geral](#visão-geral)
- [O Que Foi Implementado](#o-que-foi-implementado)
- [Arquitetura do Sistema](#arquitetura-do-sistema)
- [Como Usar](#como-usar)
- [Guia para Apresentação](#guia-para-apresentação)
- [Interpretando os Resultados](#interpretando-os-resultados)

---

## 🎯 Visão Geral

Este sistema foi desenvolvido especificamente para apresentação de TCC e demonstra:

✅ **Previsão de criptomoedas** usando Machine Learning avançado
✅ **Integração de eventos geopolíticos** como features nos modelos
✅ **Backtesting robusto** com Walk-Forward Analysis
✅ **Comparação científica** de múltiplos algoritmos
✅ **Análise estatística completa** de erros e performance
✅ **Visualizações profissionais** para apresentação

---

## 🚀 O Que Foi Implementado

### 1. **Features Geopolíticas Inteligentes** (`ml/features.py`)

**Novidade:** Agora os modelos não usam apenas indicadores técnicos, mas também:

- ✅ Contagem de eventos nos últimos 7 e 30 dias
- ✅ Sentimento médio dos eventos (Positivo/Neutro/Negativo)
- ✅ Severidade dos eventos (Baixo/Médio/Alto)
- ✅ Eventos por categoria (Econômico, Político, Inovação, etc.)
- ✅ Dias desde o último evento
- ✅ Features de interação (preço × sentimento, volatilidade × eventos)

**Total:** **100+ features** (80 técnicas + 20 geopolíticas)

### 2. **Modelos Avançados** (`ml/advanced_models.py`)

Além dos modelos existentes, adicionamos:

| Modelo | Descrição | Vantagem |
|--------|-----------|----------|
| **Prophet** | Modelo do Facebook para séries temporais | Captura sazonalidade e tendências de longo prazo |
| **ARIMA** | Modelo estatístico clássico | Excelente para séries estacionárias |
| **Ensemble Voting** | Média ponderada de múltiplos modelos | Reduz variância, mais estável |
| **Ensemble Stacking** | Meta-modelo que aprende a combinar modelos | Melhor performance |

### 3. **Walk-Forward Analysis** (`ml/walk_forward.py`)

**O que é?**
Um método de backtesting que simula o cenário REAL de trading:

```
┌─────────────────────────────────────────────────┐
│ Treino: 180 dias → Teste: 30 dias → Re-treina  │
│         ↓                                        │
│  [Jan-Jun] treina → [Jul] testa                 │
│         ↓                                        │
│  [Fev-Jul] treina → [Ago] testa                 │
│         ↓                                        │
│  [Mar-Ago] treina → [Set] testa                 │
│         ... continua deslizando                  │
└─────────────────────────────────────────────────┘
```

**Por que é melhor que backtesting simples?**
- ✅ Simula re-treinamento periódico (realista)
- ✅ Detecta degradação de modelo ao longo do tempo
- ✅ Evita overfitting
- ✅ Testa múltiplos períodos

**Métricas Calculadas:**
- Retorno Total e Anualizado
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown
- Win Rate e Profit Factor
- Comparação vs Buy & Hold

### 4. **Diagnóstico Avançado de Erros** (`ml/advanced_models.py`)

Análise estatística profunda:

- 📊 Distribuição de resíduos (normalidade)
- 📊 Heteroscedasticidade (variância constante?)
- 📊 Outliers (previsões muito ruins)
- 📊 Análise temporal (piora em certos meses/dias?)
- 📊 Percentis de erro (P25, P50, P75, P90, P95)

### 5. **Dashboard Integrado** (`paginas/ml_avancado.py`)

Uma página completa com 5 abas:

1. **📊 Comparação de Modelos**: Treina RF, XGBoost, LightGBM e compara
2. **🔄 Walk-Forward Analysis**: Backtesting robusto
3. **🌍 Impacto Geopolítico**: Correlação eventos vs preços
4. **🔍 Diagnóstico de Erros**: Análise estatística
5. **📈 Previsões Avançadas**: Ensemble de modelos

---

## 🏗️ Arquitetura do Sistema

```
coinsight_tcc/
└── streamlit_app/
    ├── ml/
    │   ├── models.py                  # ✅ Modelos base (RF, XGBoost, LightGBM, LSTM)
    │   ├── features.py                # 🆕 Agora com features geopolíticas
    │   ├── advanced_models.py         # 🆕 Prophet, ARIMA, Ensemble, Diagnóstico
    │   ├── walk_forward.py            # 🆕 Walk-Forward Analysis
    │   ├── backtest.py                # ✅ Backtesting simples
    │   └── geopolitical_analysis.py   # ✅ Análise de impacto de eventos
    │
    ├── paginas/
    │   ├── ml_dashboard.py            # ✅ Dashboard ML original
    │   └── ml_avancado.py             # 🆕 Dashboard ML Avançado (TCC)
    │
    └── app.py                         # ✅ Atualizado com nova página
```

---

## 📖 Como Usar

### Passo 1: Popular Eventos Geopolíticos

```bash
cd streamlit_app
python populate_geopolitical_events.py
```

Isso cria **60+ eventos simulados** no banco de dados.

### Passo 2: Iniciar o Streamlit

```bash
streamlit run app.py
```

### Passo 3: Navegar para "🚀 ML Avançado TCC"

No menu lateral, clique em **"🚀 ML Avançado TCC"**

### Passo 4: Workflow Recomendado

#### **Aba 1: Comparação de Modelos**

1. Selecione a moeda (Bitcoin, Ethereum, etc.)
2. Configure:
   - Conjunto de Teste: 20%
   - Validação: 10%
   - ✅ **Incluir Features Geopolíticas** (importante!)
3. Clique em **"🚀 Treinar e Comparar Modelos"**

⏱️ **Tempo:** 2-3 minutos

**O que você verá:**
- Tabela comparativa de métricas (MAE, RMSE, R², MAPE, Acurácia Direcional)
- Gráfico de barras comparando erros
- **Melhor modelo** destacado
- Top 15 features mais importantes (com destaque para as geopolíticas!)

#### **Aba 2: Walk-Forward Analysis**

1. Configure:
   - Janela de Treino: 180 dias
   - Janela de Teste: 30 dias
   - Re-treinar a cada: 30 dias
   - Estratégia: `long_short`
2. Clique em **"▶️ Executar Walk-Forward"**

⏱️ **Tempo:** 5-10 minutos (depende dos dados)

**O que você verá:**
- 4 métricas principais (Retorno, Sharpe, Drawdown, Win Rate)
- Gráfico de evolução do portfólio (4 subplots)
- Performance por fold (cada re-treinamento)
- Relatório completo exportável

#### **Aba 3: Impacto Geopolítico**

1. Clique em **"🔍 Analisar Impacto"**

**O que você verá:**
- Timeline de eventos sobrepostos ao preço
- Distribuição de impactos por categoria
- Impacto médio por severidade
- Estatísticas: qual categoria afeta mais? Alta ou baixa?
- Relatório de insights

#### **Aba 4: Diagnóstico de Erros**

**Automático** após executar Aba 1

**O que você verá:**
- Histograma de resíduos (deveria ser normal)
- Scatter plot Previsão vs Real (deveria estar na linha)
- Detecção de outliers
- Relatório estatístico completo

#### **Aba 5: Previsões Avançadas**

**Automático** após executar Aba 1

**O que você verá:**
- Previsão **Ensemble** (média de todos os modelos)
- Previsões individuais de cada modelo
- Gráfico comparativo
- Preço esperado e direção (ALTA/BAIXA)

---

## 🎓 Guia para Apresentação do TCC

### **Slide 1: Introdução**

"Vou apresentar um sistema avançado de previsão de criptomoedas que integra Machine Learning com análise de eventos geopolíticos."

**Mostre:** Página inicial do ML Avançado

### **Slide 2: Diferencial - Features Geopolíticas**

"Nosso diferencial é que **não usamos apenas indicadores técnicos**. Integramos eventos geopolíticos como features nos modelos."

**Mostre:**
- Aba 1, seção "Incluir Features Geopolíticas" ✅
- Depois, mostre o gráfico de Feature Importance e aponte para features como:
  - `events_last_7d`
  - `avg_sentiment_7d`
  - `high_severity_events_7d`

**Argumente:**
"Veja que o modelo **aprendeu automaticamente** que eventos geopolíticos são importantes para a previsão."

### **Slide 3: Comparação de Modelos**

"Treinamos e comparamos **3 algoritmos** state-of-the-art: Random Forest, XGBoost e LightGBM."

**Mostre:**
- Tabela de métricas (Aba 1)
- Gráfico comparativo

**Destaque:**
- Qual modelo ganhou
- MAE, RMSE (menores são melhores)
- R² (mais próximo de 1 é melhor)
- Acurácia Direcional (% de acertos na direção)

### **Slide 4: Validação Robusta - Walk-Forward**

"Para validar de forma rigorosa, usamos **Walk-Forward Analysis**, que simula o cenário real de trading com re-treinamento periódico."

**Mostre:**
- Gráfico de evolução do portfólio (Aba 2)
- **Destaque as linhas verticais laranja** (re-treinamentos)

**Argumente:**
"Diferente de um backtest simples, aqui o modelo é **re-treinado a cada 30 dias**, simulando o que realmente aconteceria em produção."

**Mostre as Métricas:**
- "Obtivemos um retorno de **X%** vs **Y%** do Buy & Hold"
- "Sharpe Ratio de **Z** indica boa relação risco-retorno"
- "Win Rate de **W%**"

### **Slide 5: Impacto Geopolítico**

"Analisamos **como eventos mundiais afetam os preços** das criptomoedas."

**Mostre:**
- Timeline de eventos (Aba 3)
- Gráfico de impacto por categoria
- Gráfico de impacto por severidade

**Destaque:**
- "Eventos da categoria **[X]** causaram impacto médio de **Y%**"
- "Eventos de alta severidade impactam **Z vezes mais**"
- "Conseguimos detectar padrões: eventos positivos aumentam preço em **W%** em média"

### **Slide 6: Análise de Erros**

"Realizamos diagnóstico estatístico completo dos erros."

**Mostre:**
- Histograma de resíduos (Aba 4)
- Scatter plot Previsão vs Real

**Argumente:**
- "Distribuição de resíduos aproximadamente normal ✅"
- "Previsões próximas da linha ideal ✅"
- "Apenas **X%** de outliers ✅"

### **Slide 7: Previsões Ensemble**

"Para previsões futuras, combinamos múltiplos modelos em um **Ensemble**, reduzindo variância."

**Mostre:**
- Aba 5, previsões individuais vs Ensemble
- Gráfico comparativo

**Destaque:**
- Previsão atual
- Direção esperada
- Concordância entre modelos

### **Slide 8: Conclusão**

**Recapitule:**
- ✅ Sistema com 100+ features (técnicas + geopolíticas)
- ✅ 3 algoritmos comparados
- ✅ Validação robusta com Walk-Forward
- ✅ Análise de impacto geopolítico
- ✅ Diagnóstico estatístico completo
- ✅ Previsões ensemble

**Resultados:**
- MAE: **[valor]**
- R²: **[valor]**
- Retorno Walk-Forward: **[valor]%**
- Sharpe Ratio: **[valor]**

---

## 📊 Interpretando os Resultados

### **Métricas de Erro**

| Métrica | O que significa | Quanto menor, melhor? |
|---------|-----------------|----------------------|
| **MAE** | Erro médio absoluto | ✅ Sim |
| **RMSE** | Raiz do erro quadrático (penaliza erros grandes) | ✅ Sim |
| **MAPE** | Erro percentual médio | ✅ Sim |
| **R²** | Quanto da variância é explicada (0 a 1) | ❌ Não, quanto maior melhor |
| **Acurácia Direcional** | % de acertos na direção (sobe/desce) | ❌ Não, quanto maior melhor |

### **Métricas Financeiras**

| Métrica | O que significa | Bom valor |
|---------|-----------------|-----------|
| **Retorno Total** | Lucro/prejuízo total (%) | > 0% (positivo) |
| **Sharpe Ratio** | Retorno ajustado ao risco | > 1.0 (bom), > 1.5 (excelente) |
| **Sortino Ratio** | Similar ao Sharpe, mas só penaliza volatilidade negativa | > 1.0 |
| **Maximum Drawdown** | Maior perda acumulada (%) | < 20% (tolerável) |
| **Calmar Ratio** | Retorno / Max Drawdown | > 0.5 |
| **Win Rate** | % de trades vencedores | > 50% |
| **Profit Factor** | Lucro bruto / Prejuízo bruto | > 1.5 |

### **Como Saber se o Modelo é Bom?**

✅ **Modelo EXCELENTE:**
- MAE < 0.01
- R² > 0.7
- Sharpe > 1.5
- Acurácia Direcional > 60%
- Retorno > Buy & Hold

✅ **Modelo BOM:**
- MAE < 0.02
- R² > 0.5
- Sharpe > 1.0
- Acurácia Direcional > 55%
- Retorno positivo

⚠️ **Modelo MODERADO:**
- MAE < 0.05
- R² > 0.3
- Sharpe > 0.5
- Acurácia Direcional > 50%

❌ **Modelo FRACO:**
- Métricas piores que os limiares acima

### **Análise de Resíduos**

✅ **Resíduos "Bons":**
- Distribuição aproximadamente normal (sino)
- Média próxima de 0
- Sem outliers excessivos (< 5%)
- Sem padrões óbvios no scatter plot

⚠️ **Resíduos "Problemáticos":**
- Distribuição assimétrica
- Muitos outliers (> 10%)
- Padrões no scatter plot (curva, funil)

---

## 🔧 Troubleshooting

### Problema: "Eventos geopolíticos não disponíveis"

**Solução:**
```bash
cd streamlit_app
python populate_geopolitical_events.py
```

### Problema: "XGBoost não disponível"

**Solução:**
```bash
pip install xgboost
```

ou use apenas Random Forest (já funciona).

### Problema: Walk-Forward muito lento

**Solução:**
- Reduza janela de treino (180 → 120)
- Aumente frequência de re-treino (30 → 45)
- Use menos dados (limite histórico)

### Problema: Modelos com R² negativo

**Causa:** Dados insuficientes ou muito ruidosos

**Solução:**
- Use mais dados históricos
- Aumente janela de treino
- Considere suavização (médias móveis)

---

## 📝 Checklist para Apresentação

- [ ] Popular eventos geopolíticos
- [ ] Testar com Bitcoin primeiro (mais dados)
- [ ] Executar Comparação de Modelos
- [ ] Executar Walk-Forward (deixar rodar antes)
- [ ] Capturar screenshots dos resultados
- [ ] Preparar explicação de cada métrica
- [ ] Revisar interpretação de gráficos
- [ ] Testar todas as abas
- [ ] Preparar argumentação sobre diferenciais
- [ ] Ensaiar apresentação

---

## 🎉 Parabéns!

Você agora tem um sistema completo de ML com:
- ✅ Integração geopolítica
- ✅ Múltiplos modelos
- ✅ Backtesting robusto
- ✅ Visualizações profissionais
- ✅ Análise estatística completa

**Boa sorte na apresentação do TCC! 🚀**

---

## 📚 Referências para Citar no TCC

- Random Forest: Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
- XGBoost: Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system.
- Walk-Forward Analysis: Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies.
- Sentiment Analysis: Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis.
- Time Series: Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). Time series analysis: forecasting and control.

---

**Desenvolvido para TCC 2024**
**Sistema: CoinSight - Análise e Previsão de Criptomoedas com IA**
