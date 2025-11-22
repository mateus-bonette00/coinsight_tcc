"""
Modelos Avançados de Previsão
Inclui: Prophet, ARIMA, Ensemble Voting e Stacking
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Imports condicionais
try:
    from prophet import Prophet
    HAS_PROPHET = True
except:
    HAS_PROPHET = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except:
    HAS_STATSMODELS = False


class ProphetPredictor:
    """Wrapper para Facebook Prophet"""

    def __init__(self, changepoint_prior_scale=0.05, seasonality_prior_scale=10):
        if not HAS_PROPHET:
            raise ImportError("Prophet não instalado. Execute: pip install prophet")

        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model = None
        self.is_fitted = False

    def fit(self, timestamps, values):
        """
        Treina o modelo Prophet

        Args:
            timestamps: Array ou Series com timestamps
            values: Array ou Series com valores alvo
        """
        # Prepara dados no formato do Prophet
        df = pd.DataFrame({
            'ds': pd.to_datetime(timestamps),
            'y': values
        })

        # Cria e treina modelo
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )

        self.model.fit(df)
        self.is_fitted = True
        return self

    def predict(self, timestamps):
        """Faz previsões para timestamps especificados"""
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado. Execute fit() primeiro.")

        future = pd.DataFrame({'ds': pd.to_datetime(timestamps)})
        forecast = self.model.predict(future)
        return forecast['yhat'].values

    def predict_with_intervals(self, timestamps, uncertainty_samples=1000):
        """Retorna previsões com intervalos de confiança"""
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado")

        future = pd.DataFrame({'ds': pd.to_datetime(timestamps)})
        forecast = self.model.predict(future)

        return {
            'yhat': forecast['yhat'].values,
            'yhat_lower': forecast['yhat_lower'].values,
            'yhat_upper': forecast['yhat_upper'].values
        }


class ARIMAPredictor:
    """Wrapper para ARIMA/SARIMAX"""

    def __init__(self, order=(1, 1, 1), seasonal_order=None):
        if not HAS_STATSMODELS:
            raise ImportError("Statsmodels não instalado. Execute: pip install statsmodels")

        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fit = None
        self.is_fitted = False
        self.last_train_data = None

    def fit(self, values):
        """
        Treina modelo ARIMA

        Args:
            values: Array ou Series com valores temporais
        """
        try:
            if self.seasonal_order:
                self.model = SARIMAX(
                    values,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                self.model = ARIMA(values, order=self.order)

            self.model_fit = self.model.fit(disp=False)
            self.last_train_data = values
            self.is_fitted = True
        except Exception as e:
            print(f"Erro ao treinar ARIMA: {e}")
            # Fallback para ordem mais simples
            self.order = (1, 0, 0)
            self.model = ARIMA(values, order=self.order)
            self.model_fit = self.model.fit(disp=False)
            self.is_fitted = True

        return self

    def predict(self, steps=1):
        """Faz previsões para N passos à frente"""
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado")

        forecast = self.model_fit.forecast(steps=steps)
        return forecast

    def predict_with_intervals(self, steps=1, alpha=0.05):
        """Retorna previsões com intervalos de confiança"""
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado")

        forecast = self.model_fit.get_forecast(steps=steps)
        forecast_df = forecast.summary_frame(alpha=alpha)

        return {
            'yhat': forecast_df['mean'].values,
            'yhat_lower': forecast_df['mean_ci_lower'].values,
            'yhat_upper': forecast_df['mean_ci_upper'].values
        }


class EnsemblePredictor:
    """Ensemble de múltiplos modelos (Voting e Stacking)"""

    def __init__(self, base_models: dict, ensemble_type='voting', meta_model=None):
        """
        Args:
            base_models: Dict com {nome: modelo} dos modelos base
            ensemble_type: 'voting' ou 'stacking'
            meta_model: Modelo meta para stacking (default: Ridge)
        """
        self.base_models = base_models
        self.ensemble_type = ensemble_type
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.ensemble = None
        self.is_fitted = False

    def fit(self, X, y):
        """Treina o ensemble"""
        estimators = [(name, model.model) for name, model in self.base_models.items()
                     if hasattr(model, 'model') and model.model is not None]

        if not estimators:
            raise ValueError("Nenhum modelo base disponível para ensemble")

        if self.ensemble_type == 'voting':
            self.ensemble = VotingRegressor(
                estimators=estimators,
                n_jobs=-1
            )
        else:  # stacking
            self.ensemble = StackingRegressor(
                estimators=estimators,
                final_estimator=self.meta_model,
                n_jobs=-1
            )

        # Treina ensemble
        self.ensemble.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Faz previsões"""
        if not self.is_fitted:
            raise RuntimeError("Ensemble não treinado")

        return self.ensemble.predict(X)

    def evaluate(self, X, y):
        """Avalia o ensemble"""
        predictions = self.predict(X)

        return {
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / np.where(y != 0, y, 1))) * 100
        }


class ModelDiagnostics:
    """Diagnóstico avançado de erros de modelos"""

    def __init__(self):
        self.residuals = None
        self.predictions = None
        self.actuals = None

    def analyze(self, y_true, y_pred, timestamps=None):
        """
        Analisa erros do modelo

        Returns:
            Dict com métricas de diagnóstico
        """
        self.actuals = np.array(y_true)
        self.predictions = np.array(y_pred)
        self.residuals = self.actuals - self.predictions

        # Métricas básicas
        mae = mean_absolute_error(self.actuals, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.actuals, self.predictions))
        r2 = r2_score(self.actuals, self.predictions)

        # MAPE
        mape = np.mean(np.abs(self.residuals / np.where(self.actuals != 0, self.actuals, 1))) * 100

        # Análise de resíduos
        residual_mean = np.mean(self.residuals)
        residual_std = np.std(self.residuals)
        residual_skew = pd.Series(self.residuals).skew()
        residual_kurt = pd.Series(self.residuals).kurtosis()

        # Heteroscedasticidade (variância não constante)
        abs_residuals = np.abs(self.residuals)
        heteroscedasticity_test = np.corrcoef(self.predictions, abs_residuals)[0, 1]

        # Acurácia direcional
        if len(self.actuals) > 1:
            actual_direction = np.sign(np.diff(self.actuals))
            pred_direction = np.sign(np.diff(self.predictions))
            directional_accuracy = np.mean(actual_direction == pred_direction)
        else:
            directional_accuracy = 0.0

        # Distribuição de erros por quantis
        error_percentiles = np.percentile(np.abs(self.residuals), [25, 50, 75, 90, 95])

        # Análise temporal (se timestamps disponíveis)
        temporal_analysis = {}
        if timestamps is not None and len(timestamps) == len(self.residuals):
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps),
                'residual': self.residuals,
                'abs_residual': np.abs(self.residuals)
            })
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek

            temporal_analysis = {
                'worst_month': int(df.groupby('month')['abs_residual'].mean().idxmax()),
                'best_month': int(df.groupby('month')['abs_residual'].mean().idxmin()),
                'worst_day': int(df.groupby('day_of_week')['abs_residual'].mean().idxmax()),
                'avg_error_by_month': df.groupby('month')['abs_residual'].mean().to_dict()
            }

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'residual_skewness': residual_skew,
            'residual_kurtosis': residual_kurt,
            'heteroscedasticity_corr': heteroscedasticity_test,
            'directional_accuracy': directional_accuracy,
            'error_p25': error_percentiles[0],
            'error_p50': error_percentiles[1],
            'error_p75': error_percentiles[2],
            'error_p90': error_percentiles[3],
            'error_p95': error_percentiles[4],
            'temporal_analysis': temporal_analysis
        }

    def detect_outlier_predictions(self, threshold_std=2.5):
        """Detecta previsões outliers (erros muito grandes)"""
        if self.residuals is None:
            raise RuntimeError("Execute analyze() primeiro")

        threshold = threshold_std * self.residual_std
        outliers_idx = np.where(np.abs(self.residuals) > threshold)[0]

        return {
            'outlier_indices': outliers_idx.tolist(),
            'outlier_count': len(outliers_idx),
            'outlier_percentage': len(outliers_idx) / len(self.residuals) * 100,
            'outlier_residuals': self.residuals[outliers_idx].tolist()
        }

    def compare_models(self, models_results: dict):
        """
        Compara múltiplos modelos

        Args:
            models_results: Dict com {nome_modelo: {'y_true': ..., 'y_pred': ...}}

        Returns:
            DataFrame comparativo
        """
        comparison = []

        for name, results in models_results.items():
            y_true = results['y_true']
            y_pred = results['y_pred']

            metrics = self.analyze(y_true, y_pred)
            metrics['model'] = name
            comparison.append(metrics)

        df = pd.DataFrame(comparison)
        df = df.set_index('model')

        # Adiciona ranking
        df['rank_mae'] = df['mae'].rank()
        df['rank_rmse'] = df['rmse'].rank()
        df['rank_r2'] = df['r2'].rank(ascending=False)
        df['rank_avg'] = (df['rank_mae'] + df['rank_rmse'] + df['rank_r2']) / 3

        return df.sort_values('rank_avg')

    def generate_diagnostics_report(self, model_name='Model'):
        """Gera relatório textual de diagnóstico"""
        if self.residuals is None:
            return "Execute analyze() primeiro"

        metrics = self.analyze(self.actuals, self.predictions)
        outliers = self.detect_outlier_predictions()

        report = f"""
🔍 RELATÓRIO DE DIAGNÓSTICO - {model_name}
{'='*70}

📊 MÉTRICAS DE ERRO
MAE (Erro Médio Absoluto):        {metrics['mae']:.4f}
RMSE (Raiz do Erro Quadrático):   {metrics['rmse']:.4f}
R² (Coeficiente de Determinação): {metrics['r2']:.4f}
MAPE (Erro Percentual Médio):     {metrics['mape']:.2f}%

🎯 ACURÁCIA DIRECIONAL
Acerto de Direção:                {metrics['directional_accuracy']*100:.2f}%

📈 ANÁLISE DE RESÍDUOS
Média dos Resíduos:               {metrics['residual_mean']:.4f}
Desvio Padrão:                    {metrics['residual_std']:.4f}
Assimetria (Skewness):            {metrics['residual_skewness']:.4f}
Curtose (Kurtosis):               {metrics['residual_kurtosis']:.4f}
Correlação Heteroscedasticidade:  {metrics['heteroscedasticity_corr']:.4f}

📊 DISTRIBUIÇÃO DE ERROS (Percentis)
P25: {metrics['error_p25']:.4f}
P50: {metrics['error_p50']:.4f}
P75: {metrics['error_p75']:.4f}
P90: {metrics['error_p90']:.4f}
P95: {metrics['error_p95']:.4f}

⚠️ OUTLIERS
Quantidade:                       {outliers['outlier_count']}
Porcentagem:                      {outliers['outlier_percentage']:.2f}%

💡 INTERPRETAÇÃO
- Resíduos próximos de 0: {'✅ BOM' if abs(metrics['residual_mean']) < 0.01 else '⚠️ ATENÇÃO'}
- Heteroscedasticidade: {'✅ Baixa' if abs(metrics['heteroscedasticity_corr']) < 0.3 else '⚠️ Moderada/Alta'}
- Distribuição Normal: {'✅ Sim' if abs(metrics['residual_skewness']) < 0.5 and abs(metrics['residual_kurtosis']) < 1 else '⚠️ Não'}

{'='*70}
        """

        return report
