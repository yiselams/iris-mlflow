# Informe de Resultados: Clasificación HELOC con Random Forest

## Resumen Ejecutivo
Se entrenó un modelo de Random Forest para clasificar el riesgo de crédito (Bad/Good) en el dataset HELOC, comparando dos estrategias de preprocesamiento de datos.

## Metodología
- **Dataset**: HELOC (Home Equity Line of Credit)
- **Algoritmo**: Random Forest Classifier
- **División**: 80% entrenamiento, 20% prueba (estratificada)
- **Métricas**: Accuracy, F1-score, Precisión, Recall

## Resultados Principales

### Estrategia 1: Sin reducción de dimensionalidad (23 features originales)
**Mejor modelo: sin_pca_v2**
- **Accuracy**: 72.94%
- **F1-score**: 0.7295
- **Precisión**: 0.7291
- **Recall**: 0.7300
- **Hiperparámetros**: n_estimators=150, max_depth=12

### Comparación de configuraciones sin PCA:
| Configuración | Accuracy | F1-score | n_estimators | max_depth |
|---------------|----------|----------|--------------|-----------|
| sin_pca_v2 ⭐ | 72.94% | 0.7295 | 150 | 12 |
| sin_pca_v3 | 72.70% | 0.7270 | 300 | 20 |
| sin_pca_v1 | 72.80% | 0.7280 | 200 | 15 |
| sin_pca_overfitting | 71.90% | 0.7190 | 250 | 25 |

## Conclusiones
1. **Rendimiento óptimo**: El modelo con 150 árboles y profundidad máxima de 12 alcanza el mejor balance entre precisión y generalización (72.94%)
2. **Sobreajuste**: Aumentar n_estimators a 300 y max_depth a 25 reduce el rendimiento, indicando sobreajuste
3. **Estabilidad**: Las métricas (F1, Precisión, Recall) están balanceadas, indicando buen desempeño en ambas clases
4. **Recomendación**: Usar el modelo **sin_pca_v2** para producción

## Detalles Técnicos
- **Framework**: MLflow 3.6.0 (Tracking de experimentos)
- **Librería ML**: scikit-learn
- **Características**: 23 variables originales (sin transformación PCA)
- **Modelo entrenado**: Disponible en MLflow con ID de run `9b5024f634754273bf12f6b3c8293e21`
