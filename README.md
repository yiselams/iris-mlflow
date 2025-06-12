# Random Forest + MLflow (Iris)

Repositorio de ejemplo para **documentar experimentos de Machine Learning con MLflow y/o Databricks**.  
Entrena un modelo *Random Forest* con el dataset Iris, registra hiperparámetros, métricas y artefactos, y muestra cómo comparar runs.

---

## 1 · Requisitos

| Herramienta | Versión mínima |
|-------------|----------------|
| Python      | 3.9 |
| Git         | 2.25 |
| (Opcional) Databricks Workspace | --- |

> Las dependencias Python están en `requirements.txt`.

---

## 2 · Instalación

```bash
# 1. Clona el repo
git clone https://github.com/delany-ramirez/iris-mlflow.git
cd iris-mlflow

# 2. Crea y activa un entorno virtual
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.\.venv\Scriptsctivate

# 3. Instala dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3 · Tracking local vs Databricks

| Opción | Tracking URI | Cómo lanzar |
|--------|--------------|-------------|
| **Local (por defecto)** | `file://./mlruns` | `python train_rf_mlflow.py --run_name rf_local` |
| **Ruta local personalizada** | `file:///C:/_mlflow_runs` | 1) `export MLFLOW_TRACKING_URI=file:///C:/_mlflow_runs`<br>2) Ejecuta el script |
| **Databricks** | `https://<workspace>.databricks.com` | 

```bash
export MLFLOW_TRACKING_URI=https://<workspace>.databricks.com
export DATABRICKS_HOST=$MLFLOW_TRACKING_URI
export DATABRICKS_TOKEN=<PAT>
python train_rf_mlflow.py --n_estimators 150 --max_depth 6 --run_name rf_dbx
```

---

## 4 · Ejecutar experimentos

```bash
# Ejemplo rápido (tracking local)
python train_rf_mlflow.py --n_estimators 200 --max_depth 4 --run_name rf_200_4
python train_rf_mlflow.py --n_estimators 80  --max_depth 3 --run_name rf_80_3
```

*Verás en consola el `Run ID`, `Experiment ID` y las métricas registradas.*

---

## 5 · Visualizar la UI de MLflow

### 5.1 Local

```bash
mlflow ui --backend-store-uri ./mlruns
# Luego abre http://127.0.0.1:5000
```

### 5.2 Databricks

La interfaz está integrada en *Workspace » Experiments*.

---

## 6 · Estructura del proyecto

```
.
├─ train_rf_mlflow.py          # Script principal de entrenamiento + logging
├─ requirements_mlflow.txt     # Dependencias
├─ delete_desktop_ini.py       # Limpieza opcional de desktop.ini
├─ mlruns/                     # (Auto‑generado) runs locales
└─ README.md
```

---

## 7 · Prevención de `desktop.ini`

En Windows, el Explorador puede crear archivos `desktop.ini` dentro de
`mlruns/**/metrics/`, lo que rompe la UI.  
Formas de evitarlo:

1. No abrir `mlruns/` con el Explorador (usa terminal/VS Code).
2. Mover el tracking URI a una ruta que no navegues.
3. Actualizar a **MLflow ≥ 3.3**, que ignora archivos no‑.csv/.jsonl.
4. El script `delete_desktop.py` borra esos archivos antes de lanzar la UI.

---

## 8 · Licencia

[MIT](LICENSE)

---

## 9 · Autor

Délany Ramírez — *Universidad Tecnológica de Pereira*  
¿Sugerencias o errores? Abre un *issue* o un *pull request*.
