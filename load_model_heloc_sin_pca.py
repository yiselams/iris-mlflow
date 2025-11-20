# load_model_heloc_sin_pca.py  ──────────────────────────────────────────────────
import mlflow
import pandas as pd

# Configurar URI de MLflow
mlflow.set_tracking_uri("./mlruns")

# Run ID del mejor modelo (sin_pca_v2 - con 72.94% accuracy)
best_run_id = "9b5024f634754273bf12f6b3c8293e21"

try:
    # Cargar el modelo
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    print("=" * 60)
    print("✅ Modelo cargado correctamente")
    print(f"Run ID: {best_run_id}")
    print("=" * 60)
    
    # Cargar datos HELOC SIN PCA
    df = pd.read_csv("/workspaces/iris-mlflow/heloc.csv")
    X = df.drop("RiskPerformance", axis=1)
    y_actual = df["RiskPerformance"]
    
    # Predicciones
    predicciones_todas = model.predict(X)
    correctas = sum(predicciones_todas == y_actual)
    total = len(y_actual)
    accuracy = correctas / total * 100
    
    print(f"\n📈 Resultados del modelo:")
    print(f"   Predicciones correctas: {correctas}/{total}")
    print(f"   Accuracy: {accuracy:.2f}%")
    
    # Predicción con datos nuevos
    print(f"\n🔮 Ejemplo de predicción:")
    nuevo_registro = pd.DataFrame({
        'ExternalRiskEstimate': [65.0],
        'MSinceOldestTradeOpen': [200],
        'MSinceMostRecentTradeOpen': [12],
        'AverageMInFile': [80],
        'NumSatisfactoryTrades': [10],
        'NumTrades60Ever2DerogPubRec': [0],
        'NumTrades90Ever2DerogPubRec': [0],
        'PercentTradesNeverDelq': [95],
        'MSinceMostRecentDelq': [48],
        'MaxDelq2PublicRecLast12M': [1],
        'MaxDelqEver': [1],
        'NumTotalTrades': [15],
        'NumTradesOpeninLast12M': [2],
        'PercentInstallTrades': [30],
        'MSinceMostRecentInqexcl7days': [6],
        'NumInqLast6M': [1],
        'NumInqLast6Mexcl7days': [1],
        'NetFractionRevolvingBurden': [40.0],
        'NetFractionInstallBurden': [10.0],
        'NumRevolvingTradesWBalance': [5],
        'NumInstallTradesWBalance': [2],
        'NumBank2NatlTradesWHighUtilization': [0],
        'PercentTradesWBalance': [80]
    })
    prediccion_nueva = model.predict(nuevo_registro)
    print(f"   Predicción: {prediccion_nueva[0]}")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Verifica el Run ID")
