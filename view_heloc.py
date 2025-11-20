import mlflow
import pandas as pd

mlflow.set_tracking_uri("./mlruns")

# Obtener solo experimento HELOC
experiment = mlflow.get_experiment("856783270997459814")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

print("\n" + "="*80)
print(f"EXPERIMENTO: {experiment.name}")
print("="*80)

for idx, (run_id, row) in enumerate(runs.iterrows(), 1):
    print(f"\n✨ Run #{idx}")
    print(f"   ID: {row.get('run_id', 'N/A')}")
    
    # Métricas principales
    acc = row.get('metric_accuracy', None)
    f1 = row.get('metric_f1', None)
    if acc:
        print(f"   Accuracy: {acc:.3f}")
    if f1:
        print(f"   F1: {f1:.3f}")

print("\n" + "="*80 + "\n")
