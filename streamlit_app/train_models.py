"""
Script para treinar modelos offline e salvar para uso posterior
Execute: python train_models.py --moeda BTC --task regression
"""
import os
import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Adiciona ml ao path
sys.path.append(os.path.dirname(__file__))

from ml.features import FeatureEngine, prepare_train_test_split
from ml.models import CryptoPredictor, ModelComparator

load_dotenv()


def load_data(moeda_id: int, limit: int = 2000):
    """Carrega dados do banco"""
    url = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/coinsight")
    engine = create_engine(url)
    
    q = text("""
        SELECT timestamp, open, high, low, close, volume, moeda_id
        FROM precos
        WHERE moeda_id = :m
        ORDER BY timestamp DESC
        LIMIT :n
    """)
    
    df = pd.read_sql_query(q, engine, params={"m": moeda_id, "n": limit})
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    print(f"✅ Carregados {len(df)} registros para moeda_id={moeda_id}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Treina modelos de ML para previsão de criptomoedas')
    parser.add_argument('--moeda', type=str, default='BTC', choices=['BTC', 'ETH', 'ADA', 'SOL'],
                       help='Moeda para treinar')
    parser.add_argument('--task', type=str, default='regression', choices=['regression', 'classification'],
                       help='Tipo de tarefa')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporção de teste (0-1)')
    parser.add_argument('--horizon', type=int, default=1, help='Horizonte de previsão (dias)')
    
    args = parser.parse_args()
    
    # Mapeia moeda para ID
    moeda_map = {'BTC': 1, 'ETH': 2, 'ADA': 3, 'SOL': 4}
    moeda_id = moeda_map[args.moeda]
    
    print(f"\n{'='*60}")
    print(f"🚀 TREINAMENTO DE MODELOS - {args.moeda}")
    print(f"{'='*60}\n")
    
    # 1. Carrega dados
    print("📊 Carregando dados...")
    df_prices = load_data(moeda_id, limit=2000)
    
    if len(df_prices) < 100:
        print("❌ Dados insuficientes. Execute o ETL primeiro.")
        return
    
    # 2. Feature Engineering
    print("🔧 Criando features...")
    fe = FeatureEngine()
    df_features = fe.create_all_features(df_prices)
    df_with_target, target = fe.create_target(df_features, horizon=args.horizon, target_type=args.task)
    
    print(f"   - Features criadas: {len(fe.get_feature_names(df_features))}")
    print(f"   - Amostras: {len(df_with_target)}")
    
    # 3. Split temporal
    print("📊 Dividindo dados temporalmente...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_test_split(
        pd.concat([df_with_target, target], axis=1),
        test_size=args.test_size,
        val_size=0.15
    )
    
    print(f"   - Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")
    
    # 4. Treina modelos
    print("\n🤖 Treinando modelos...\n")
    
    comparator = ModelComparator(task=args.task)
    comparator.add_model("Random Forest", "random_forest")
    
    try:
        comparator.add_model("XGBoost", "xgboost")
    except ImportError:
        print("⚠️  XGBoost não disponível. Instale com: pip install xgboost")
    
    try:
        comparator.add_model("LightGBM", "lightgbm")
    except ImportError:
        print("⚠️  LightGBM não disponível. Instale com: pip install lightgbm")
    
    # Treina
    comparator.train_all(X_train, y_train, X_val, y_val)
    
    # 5. Avalia
    print("\n📊 Avaliando modelos...\n")
    results = comparator.evaluate_all(X_test, y_test)
    
    print(results)
    print()
    
    # 6. Melhor modelo
    if args.task == 'regression':
        best_metric = 'mae'
        best_name = results[best_metric].idxmin()
    else:
        best_metric = 'f1'
        best_name = results[best_metric].idxmax()
    
    print(f"🏆 Melhor modelo: {best_name} ({best_metric}={results.loc[best_name, best_metric]:.4f})")
    
    # 7. Salva modelos
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\n💾 Salvando modelos em {models_dir}/...")
    
    for name, model in comparator.models.items():
        if model.is_fitted:
            filename = f"{args.moeda}_{args.task}_{name.lower().replace(' ', '_')}.joblib"
            filepath = os.path.join(models_dir, filename)
            model.save(filepath)
            print(f"   ✅ {filename}")
    
    # 8. Salva melhor modelo como "best"
    best_model = comparator.models[best_name]
    best_filepath = os.path.join(models_dir, f"{args.moeda}_{args.task}_BEST.joblib")
    best_model.save(best_filepath)
    print(f"   🏆 {args.moeda}_{args.task}_BEST.joblib")
    
    print(f"\n{'='*60}")
    print("✨ TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"{'='*60}\n")
    
    # 9. Feature importance do melhor modelo
    print("🎯 Top 10 Features Mais Importantes:\n")
    importance = best_model.get_feature_importance(top_n=10)
    for idx, row in importance.iterrows():
        print(f"   {idx+1}. {row['feature']:<30} {row['importance']:.4f}")
    
    print()


if __name__ == "__main__":
    main()