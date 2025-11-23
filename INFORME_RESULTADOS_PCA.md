# Informe de Resultados: Clasificación HELOC con Random Forest + PCA

## Resumen Ejecutivo
Se entrenó un modelo de Random Forest para clasificar el riesgo de crédito (Bad/Good) en el dataset HELOC, utilizando Análisis de Componentes Principales (PCA) para reducir la dimensionalidad de 23 a 5 características.

## Metodología
- **Dataset**: HELOC (Home Equity Line of Credit)
- **Preprocesamiento**: PCA (5 componentes principales)
- **Algoritmo**: Random Forest Classifier
- **División**: 80% entrenamiento, 20% prueba (estratificada)
- **Métricas**: Accuracy, F1-score, Precisión, Recall

## Resultados Principales

### Estrategia: Con reducción PCA (5 componentes principales)
**Mejor modelo: optimized_v2**
- **Accuracy**: 71.82%
- **F1-score**: 0.7182
- **Precisión**: 0.7055
- **Recall**: 0.7182
- **Hiperparámetros**: n_estimators=100, max_depth=12

### Comparación de configuraciones con PCA:
| Configuración | Accuracy | F1-score | n_estimators | max_depth |
|---------------|----------|----------|--------------|-----------|
| optimized_v2 ⭐ | 71.82% | 0.7182 | 100 | 12 |
| optimized_v4 | 70.99% | 0.7099 | 300 | 20 |
| optimized_v3 | 70.64% | 0.7064 | 200 | 20 |
| optimized_v1 | 71.18% | 0.7118 | 250 | 25 |

## Conclusiones
1. **Rendimiento con PCA**: El mejor modelo con reducción dimensional alcanza 71.82% de accuracy
2. **Impacto de la reducción**: PCA reduce de 23 a 5 features pero mantiene rendimiento comparable (71.82% vs 72.94% sin PCA)
3. **Balance mejora/simplificidad**: Se logra una reducción del 78% de features con solo 1.12% menos accuracy
4. **Estabilidad**: Las métricas están balanceadas, indicando buen desempeño en ambas clases
5. **Recomendación**: Usar PCA si se requiere **modelo más simple y rápido** (5 features vs 23). Usar sin PCA si se prioritiza **máxima accuracy** (72.94%)

## Comparación PCA vs Sin PCA
| Aspecto | Con PCA | Sin PCA |
|--------|---------|---------|
| Features | 5 | 23 |
| Accuracy | 71.82% | 72.94% |
| Complejidad | Baja | Alta |
| Tiempo entrenamiento | Rápido | Moderado |
| Interpretabilidad | Baja* | Alta |

*PCA transforma features a componentes principales (no interpretables directamente)

## Detalles Técnicos
- **Framework**: MLflow 3.6.0 (Tracking de experimentos)
- **Librería ML**: scikit-learn
- **Características**: 5 componentes principales (PCA)
- **Modelo entrenado**: Disponible en MLflow con ID de run `dc95a6bc1c5741d5aeee7b92a9c4644e`
