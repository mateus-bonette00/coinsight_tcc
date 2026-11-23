# CoinSight — Cryptocurrency Return Prediction with Machine Learning

**Computer Science undergraduate thesis (TCC) · Federal University of Itajubá (UNIFEI), Brazil**

A research-oriented application for analyzing and predicting cryptocurrency returns using **market indicators, geopolitical-event features, and time-aware machine learning**.

This project explores the question: **Can price-derived and geopolitical signals help explain or predict short-term crypto returns without introducing look-ahead bias?**

> Research and educational use only. This project is not a trading recommendation or a guarantee of predictive accuracy.

## What is inside

- **Data pipeline:** market data collection and ETL for multiple crypto assets.
- **Feature engineering:** technical indicators and geopolitical-event signals, including rolling windows, momentum, volatility, event frequency and sentiment.
- **Model comparison:** Random Forest, XGBoost and LightGBM for a return-regression task.
- **Temporal evaluation:** chronological train/validation/test splits and documented walk-forward analysis, rather than randomly shuffling time series.
- **Application:** Streamlit interface and PostgreSQL-backed data workflows.
- **Research documentation:** methodology, technical implementation details and presentation material.

## High-level architecture

```text
Market data + geopolitical events
                |
         Ingestion / ETL
                |
          PostgreSQL
                |
       Feature engineering
                |
    Temporal train / validate
                |
  Random Forest / XGBoost / LightGBM
                |
       Backtesting + metrics
                |
        Streamlit interface
```

## Modeling and evaluation

The [technical summary](RESUMO_TECNICO_TCC.md) describes the research design, including 100+ candidate features across technical and geopolitical groups. The target is the **next-day percentage return**, and the documentation outlines a **70% / 10% / 20% chronological split** plus walk-forward windows.

Evaluation metrics covered in the technical documentation include MAE, RMSE, R², directional accuracy, Sharpe ratio, Sortino ratio and drawdown.

**Important:** This README does not claim any particular backtest return, live-trading result, or superior model performance. Such conclusions require reproducible experimental results and inspection of the corresponding runs.

## Repository map

| Path | Purpose |
|---|---|
| [streamlit_app/](streamlit_app/) | Streamlit application, UI components and ML modules |
| [scripts/](scripts/) | Data collection and ETL scripts |
| [models/](models/) | Model-related artifacts and resources |
| [notebooks/](notebooks/) | Exploratory analysis and experiments |
| [RESUMO_TECNICO_TCC.md](RESUMO_TECNICO_TCC.md) | Technical architecture and evaluation design |
| [DOCUMENTACAO_COMPLETA_PROJETO_ARTIGO.md](DOCUMENTACAO_COMPLETA_PROJETO_ARTIGO.md) | Extended thesis/project documentation |
| [GUIA_ML_AVANCADO_TCC.md](GUIA_ML_AVANCADO_TCC.md) | Modeling methodology and research notes |

## Running locally

**Prerequisites:** Python, PostgreSQL and the external data sources used by the pipeline.

1. Clone the repository and prepare a virtual environment:

```bash
git clone https://github.com/mateus-bonette00/coinsight_tcc.git
cd coinsight_tcc
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure a local PostgreSQL database and check the data-ingestion scripts under [scripts/](scripts/). The included [rodar_etl.sh](rodar_etl.sh) assumes a specific local virtual-environment path and a hard-coded local PostgreSQL connection; review and adapt it **before** running it.

3. Once the project dependencies and expected data are available, start the interface:

```bash
streamlit run streamlit_app/app.py
```

The setup has **not been independently re-run or validated** as part of this README improvement. External data-source availability, database initialization and environment-specific dependencies may require adjustments.

## Engineering focus

This is an applied ML research project, not a production trading engine. Its value lies in the feature-engineering approach, transparent comparisons, the time-series evaluation methodology and the combination of software engineering, data analysis and domain context.

**Author:** [Mateus Bonette](https://github.com/mateus-bonette00) — [Portfolio](https://mateusbonette.com.br) · [LinkedIn](https://www.linkedin.com/in/mateus-bonette/)
