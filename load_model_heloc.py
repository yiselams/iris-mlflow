# load_model_heloc.py  ───────────────────────────────────────────────────────────
import mlflow
import pandas as pd

# Configurar URI de MLflow
mlflow.set_tracking_uri("./mlruns")

# Run ID del mejor modelo (optimized_v1 - con 71.82% accuracy)
best_run_id = "3f57eaf39e9b4757a958501e6614a910"

try:
    # Cargar el modelo
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    print("=" * 60)
    print("✅ Modelo cargado correctamente")
    print(f"Run ID: {best_run_id}")
    print("=" * 60)
    
    # Cargar datos HELOC
    df = pd.read_csv("/workspaces/iris-mlflow/heloc_procesado.csv")
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
        'PC1': [-1.5],
        'PC2': [0.5],
        'PC3': [1.2],
        'ExternalRiskEstimate': [65.0],
        'NetFractionRevolvingBurden': [40.0]
    })
    prediccion_nueva = model.predict(nuevo_registro)
    print(f"   Predicción: {prediccion_nueva[0]}")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Verifica el Run ID")
