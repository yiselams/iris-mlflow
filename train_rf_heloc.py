# train_rf_heloc.py  ───────────────────────────────────────────────────────────
import argparse
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
import pandas as pd

# -------------------- CLI ------------------------------------------------------
parser = argparse.ArgumentParser(description="Random Forest + MLflow (HELOC)")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=None)
parser.add_argument("--experiment_name", type=str, default="HELOC_RF")
parser.add_argument("--run_name", type=str, default=None)
args = parser.parse_args()

# -------------------- Datos ----------------------------------------------------
df = pd.read_csv("heloc_procesado_regresion.csv")

# Separa features (X) y target (y)
X = df.drop("RiskPerformance", axis=1)  # Features: PC1, PC2, PC3, ExternalRiskEstimate, NetFractionRevolvingBurden
y = df["RiskPerformance"]  # Target: Bad/Good

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------- Experimento y run ----------------------------------------
mlflow.set_experiment(args.experiment_name)
with mlflow.start_run(run_name=args.run_name) as run:
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    # ---- loggeo ----------------------------------------------------------------
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_binary", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    signature = infer_signature(X_train, clf.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=clf,
        name="model",  
        signature=signature,
        input_example=X_train[:5],
    )

    print("=" * 60)
    print("Run ID:", run.info.run_id)
    print("Experiment ID:", run.info.experiment_id)
    print("Tracking URI:", mlflow.get_tracking_uri())
    print(f"accuracy={acc:.3f}  f1={f1:.3f}  precision={precision:.3f}  recall={recall:.3f}")
    print("=" * 60)
