"""
Modelos de Machine Learning para Previsão de Criptomoedas
Inclui: Random Forest, XGBoost, LightGBM e LSTM
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Imports condicionais
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_KERAS = True
except:
    HAS_KERAS = False


class CryptoPredictor:
    """Classe unificada para diferentes modelos de previsão"""
    
    def __init__(self, model_type: str = 'random_forest', task: str = 'regression'):
        """
        Args:
            model_type: 'random_forest', 'xgboost', 'lightgbm', 'lstm'
            task: 'regression' ou 'classification'
        """
        self.model_type = model_type
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.metrics = {}
        
        self._init_model()
    
    def _init_model(self):
        """Inicializa o modelo baseado no tipo escolhido"""
        if self.model_type == 'random_forest':
            if self.task == 'regression':
                self.model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    max_features='sqrt',
                    n_jobs=-1,
                    random_state=42
                )
            else:
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    max_features='sqrt',
                    n_jobs=-1,
                    random_state=42
                )
        
        elif self.model_type == 'xgboost':
            if not HAS_XGB:
                raise ImportError("XGBoost não instalado. Execute: pip install xgboost")
            
            if self.task == 'regression':
                self.model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                self.model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
        
        elif self.model_type == 'lightgbm':
            if not HAS_LGB:
                raise ImportError("LightGBM não instalado. Execute: pip install lightgbm")
            
            if self.task == 'regression':
                self.model = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            else:
                self.model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
        
        elif self.model_type == 'lstm':
            if not HAS_KERAS:
                raise ImportError("TensorFlow não instalado. Execute: pip install tensorflow")
            # LSTM será criado no fit com base na forma dos dados
            pass
        
        else:
            raise ValueError(f"Modelo '{self.model_type}' não reconhecido")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Treina o modelo"""
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        
        # Escala os dados
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.model_type == 'lstm':
            self._fit_lstm(X_train_scaled, y_train, X_val, y_val)
        else:
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                
                if self.model_type in ['xgboost', 'lightgbm']:
                    self.model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        verbose=False
                    )
                else:
                    self.model.fit(X_train_scaled, y_train)
            else:
                self.model.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        return self
    
    def _fit_lstm(self, X_train, y_train, X_val, y_val):
        """Treina modelo LSTM"""
        # Reshape para LSTM: (samples, timesteps, features)
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        # Cria arquitetura
        self.model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(1, X_train.shape[1])),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1) if self.task == 'regression' else Dense(1, activation='sigmoid')
        ])
        
        # Compila
        if self.task == 'regression':
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        else:
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Treina
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
            validation_data = (X_val_lstm, y_val)
        
        self.model.fit(
            X_train_lstm, y_train,
            epochs=100,
            batch_size=32,
            validation_data=validation_data,
            callbacks=[early_stop],
            verbose=0
        )
    
    def predict(self, X):
        """Faz previsões"""
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado. Execute fit() primeiro.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'lstm':
            X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            predictions = self.model.predict(X_lstm, verbose=0).flatten()
        else:
            predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X):
        """Retorna probabilidades (apenas para classificação)"""
        if self.task != 'classification':
            raise ValueError("predict_proba só funciona para classificação")
        
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'lstm':
            X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            proba = self.model.predict(X_lstm, verbose=0).flatten()
            return np.column_stack([1 - proba, proba])
        else:
            return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X, y) -> Dict[str, float]:
        """Avalia o modelo e retorna métricas"""
        predictions = self.predict(X)
        
        if self.task == 'regression':
            metrics = {
                'mae': mean_absolute_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions)),
                'r2': r2_score(y, predictions),
                'mape': np.mean(np.abs((y - predictions) / np.where(y != 0, y, 1))) * 100
            }
            
            # Acerto direcional (sign accuracy)
            y_direction = np.sign(y)
            pred_direction = np.sign(predictions)
            metrics['directional_accuracy'] = np.mean(y_direction == pred_direction)
            
        else:  # classification
            metrics = {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, zero_division=0),
                'recall': recall_score(y, predictions, zero_division=0),
                'f1': f1_score(y, predictions, zero_division=0)
            }
            
            try:
                proba = self.predict_proba(X)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y, proba)
            except:
                metrics['roc_auc'] = 0.0
        
        self.metrics = metrics
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Retorna importância das features (para modelos tree-based)"""
        if self.model_type == 'lstm':
            return pd.DataFrame({'feature': ['LSTM não tem feature importance'], 'importance': [0]})
        
        if not hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({'feature': ['Modelo não suporta feature importance'], 'importance': [0]})
        
        importance = self.model.feature_importances_
        feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(importance))]
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return df_importance
    
    def save(self, filepath: str):
        """Salva o modelo"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'task': self.task,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str):
        """Carrega um modelo salvo"""
        data = joblib.load(filepath)
        
        predictor = cls(model_type=data['model_type'], task=data['task'])
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_names = data['feature_names']
        predictor.metrics = data['metrics']
        predictor.is_fitted = True
        
        return predictor


class ModelComparator:
    """Compara múltiplos modelos"""
    
    def __init__(self, task: str = 'regression'):
        self.task = task
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model_type: str):
        """Adiciona um modelo para comparação"""
        self.models[name] = CryptoPredictor(model_type=model_type, task=self.task)
    
    def train_all(self, X_train, y_train, X_val=None, y_val=None):
        """Treina todos os modelos"""
        print("🚀 Iniciando treinamento de todos os modelos...\n")
        
        for name, model in self.models.items():
            print(f"⏳ Treinando {name}...")
            try:
                model.fit(X_train, y_train, X_val, y_val)
                print(f"✅ {name} treinado com sucesso!")
            except Exception as e:
                print(f"❌ Erro ao treinar {name}: {str(e)}")
        
        print("\n✨ Treinamento concluído!")
    
    def evaluate_all(self, X_test, y_test) -> pd.DataFrame:
        """Avalia todos os modelos no conjunto de teste"""
        results = []
        
        for name, model in self.models.items():
            if not model.is_fitted:
                continue
            
            metrics = model.evaluate(X_test, y_test)
            metrics['model'] = name
            results.append(metrics)
        
        self.results = pd.DataFrame(results).set_index('model')
        return self.results
    
    def get_best_model(self, metric: str = 'mae') -> Tuple[str, CryptoPredictor]:
        """Retorna o melhor modelo baseado em uma métrica"""
        if self.results.empty:
            raise RuntimeError("Execute evaluate_all() primeiro")
        
        if self.task == 'regression':
            # Para regressão, menor é melhor (exceto R²)
            if metric == 'r2':
                best_name = self.results[metric].idxmax()
            else:
                best_name = self.results[metric].idxmin()
        else:
            # Para classificação, maior é melhor
            best_name = self.results[metric].idxmax()
        
        return best_name, self.models[best_name]