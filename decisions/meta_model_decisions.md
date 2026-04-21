# Meta-modelo (Ensemble de segundo nivel) -- Decisiones de Diseño

**Fecha:** 2026-04-21
**Tarea:** Razonar si tiene sentido un meta-modelo en este proyecto, y en caso afirmativo, diseñarlo.
**Modelos base disponibles:** rf, xgb, gru, lstm, cnn_gru, cnn_lstm, markov
**Target types:** discrete (0/1 dirección) y continuous (future_log_ret)

---

## 1. ¿Qué es un meta-modelo?

Un meta-modelo (también llamado *stacking* o *blending*) es un modelo de segundo nivel que toma como entrada las predicciones de varios modelos base y produce una predicción combinada. La intuición es que cada modelo base comete errores en lugares distintos: si sus errores están poco correlacionados, combinarlos puede reducir la varianza del error total sin aumentar el sesgo.

Formalmente: si los modelos base producen predicciones $\hat{y}_1, ..., \hat{y}_K$, el meta-modelo aprende $f(\hat{y}_1, ..., \hat{y}_K) \to y$.

---

## 2. ¿Tiene sentido en este proyecto?

### 2.1 Argumentos a favor

**Diversidad real entre modelos.**
Los siete modelos de este proyecto difieren en estructura fundamental:

| Modelo | Familia | Estructura temporal | Features usadas |
|--------|---------|---------------------|-----------------|
| rf | Árbol (bagging) | Ninguna (flat) | 41 features |
| xgb | Árbol (boosting) | Ninguna (flat) | 41 features |
| gru | Red recurrente | Secuencia T=20 | 41 features |
| lstm | Red recurrente | Secuencia T=20 | 41 features |
| cnn_gru | Conv + recurrente | Secuencia T=20 | 41 features |
| cnn_lstm | Conv + recurrente | Secuencia T=20 | 41 features |
| markov | Probabilístico | Lag-1 explícito | 1-2 features |

RF y XGBoost tratan cada día de forma independiente (flat). Los modelos neuronales tienen memoria explícita sobre las últimas 20 sesiones. Markov captura autocorrelación pura de retornos. Esta heterogeneidad es precisamente la condición necesaria para que un ensemble sea útil: si los modelos fallaran en los mismos días, combinarlos no ayudaría.

Empíricamente, en predicción de dirección de renta variable, Krauss et al. (2017) muestran que los ensembles de modelos heterogéneos (árboles + redes neuronales) producen balanced accuracy sistemáticamente superiores a cualquier modelo individual, con mejoras de 0.5-1.5 puntos porcentuales sobre el mejor modelo base.

**Reducción de varianza cross-window.**
El CV por ventanas deslizantes muestra que ningún modelo domina en todas las ventanas. RF puede ser superior en períodos de baja volatilidad donde los patrones técnicos son estables; XGBoost puede superar en regímenes donde las interacciones de features importan más; los modelos neuronales pueden capturar dinámicas de trayectoria que los árboles ignoran. Un meta-modelo que aprende cuándo confiar en cada modelo base puede explotar esta complementariedad.

**Coste computacional bajo en inferencia.**
Los modelos base ya están entrenados y sus artefactos están en disco. En inferencia, el meta-modelo simplemente concatena sus salidas (un vector de dimensión K=7 o K=5) y pasa por un modelo ligero (regresión logística, RF pequeño). El coste incremental es despreciable.

### 2.2 Argumentos en contra / riesgos

**Riesgo de data leakage en el entrenamiento del meta-modelo.**
Este es el riesgo principal y el más frecuentemente ignorado. Si se entrena el meta-modelo con las predicciones que los modelos base hicieron sobre sus propios datos de entrenamiento, los modelos base habrán "memorizado" esos datos y producirán predicciones artificialmente buenas. El meta-modelo aprenderá a confiar en señales que no se generalizarán fuera de muestra.

La solución correcta es el *out-of-fold stacking*: para cada ventana CV, los modelos base se entrenan en el fold de entrenamiento y predicen sobre el fold de test (datos que no vieron). Estas predicciones out-of-fold constituyen las features del meta-modelo. Wolpert (1992) establece este procedimiento como condición necesaria para que el stacking sea válido estadísticamente. En la práctica, esto significa que el meta-modelo no puede entrenarse hasta haber completado la pasada de CV de todos los modelos base.

**Coste de entrenamiento.**
Para tener predicciones out-of-fold limpias de los 7 modelos base, hay que ejecutar el CV completo de cada uno (incluyendo GRU, LSTM, CNN+GRU, CNN+LSTM, que son computacionalmente costosos). Esto multiplica el tiempo de entrenamiento.

**Interpretabilidad reducida.**
Un meta-modelo añade una capa de opacidad. El sistema ya era difícil de interpretar a nivel de features individuales; con un meta-modelo el "por qué" de una predicción se distribuye entre dos niveles.

**Overfitting del meta-modelo.**
Con 7 modelos base, el meta-modelo tiene 7 inputs (o más si se incluyen distintos target_types). Si se usa un meta-modelo con muchos parámetros (ej: un XGBoost con profundidad 5), puede overfitear fácilmente sobre las ~N ventanas de out-of-fold disponibles. La solución es usar un meta-modelo deliberadamente simple (regresión logística con regularización L2).

### 2.3 Veredicto

**Sí tiene sentido**, con las condiciones siguientes:
1. Solo sobre los modelos de clasificación discreta (5 modelos: rf, xgb, gru, lstm, markov — excluyendo cnn_gru y cnn_lstm para reducir redundancia con gru/lstm, aunque pueden incluirse).
2. Usando exclusivamente predicciones out-of-fold para el entrenamiento.
3. Meta-modelo: **regresión logística con regularización L2** (simple, interpretable, bien calibrado).
4. Las predicciones que entran al meta-modelo son las **probabilidades P(up)** de cada modelo base, no las predicciones duras, porque contienen más información.
5. El meta-modelo también entra a la misma pipeline BaseTrainer para que sea consistente con el resto del proyecto.

---

## 3. Diseño del meta-modelo

### 3.1 Features del meta-modelo

Para cada observación (fecha, ticker), el meta-modelo recibe:

```
[P(up)_rf,  P(up)_xgb,  P(up)_gru,  P(up)_lstm,  P(up)_markov]
```

Dimensión: (n_samples, 5). Opcionalmente se pueden añadir los modelos CNN pero aumenta la redundancia con GRU/LSTM.

**¿Por qué probabilidades y no predicciones duras?**
Las predicciones duras (0/1) descartan información de margen: un RF con P(up)=0.51 y uno con P(up)=0.73 producen la misma predicción dura pero el meta-modelo debería ponderar el segundo más. Breiman (1996) en su trabajo original sobre stacking recomienda usar las probabilidades completas de clase como features del nivel 2.

### 3.2 Algoritmo del meta-modelo

**Regresión logística con regularización L2 (C=1.0):**

```python
from sklearn.linear_model import LogisticRegression
META_PARAMS = {"C": 1.0, "max_iter": 1000, "random_state": 42}
model = LogisticRegression(**META_PARAMS)
```

Justificación:
- Con solo 5-7 inputs, la regresión logística tiene suficiente capacidad para aprender pesos diferenciales entre modelos base sin overfitar.
- Produce probabilidades bien calibradas (Niculescu-Mizil & Caruana, 2005).
- Los coeficientes son directamente interpretables: si $w_{rf} > w_{xgb}$, el meta-modelo confía más en RF que en XGBoost, lo cual tiene valor diagnóstico.
- C=1.0 (L2 moderada) por defecto. Si el número de ventanas out-of-fold es pequeño (< 500 observaciones por clase), reducir a C=0.1.

**Alternativa descartada: XGBoost como meta-modelo.**
Un XGBoost de segundo nivel puede capturar interacciones no lineales entre modelos base (ej: "si RF y XGBoost coinciden pero GRU discrepa, confía en GRU"). Pero con 5-7 inputs y pocas muestras efectivas, el riesgo de overfitting supera el beneficio. Si los experimentos muestran que la regresión logística se queda corta, un RF pequeño (n_estimators=100, max_depth=2) es la primera escalada.

### 3.3 Protocolo out-of-fold (sin leakage)

El proceso correcto tiene dos fases:

**Fase 1 — Generación de out-of-fold features:**
Para cada ventana $(train\_dates_i, test\_dates_i)$ del CV estándar (750 días deslizantes):
1. Entrenar cada modelo base solo con $train\_dates_i$.
2. Predecir sobre $test\_dates_i$ → guardar probabilidades $P(up)_{k,i}$ para cada modelo $k$.
3. Estas predicciones forman la fila correspondiente del dataset de entrenamiento del meta-modelo.

Al final del CV, se tiene un dataset de tamaño $(n_{oof\_samples}, K)$ donde $K$ = número de modelos base, y cada fila es una predicción limpia (el modelo base no vio esos datos durante su entrenamiento).

**Fase 2 — Entrenamiento del meta-modelo:**
Con el dataset out-of-fold completo, entrenar el meta-modelo (regresión logística). Este entrenamiento es rápido (segundos).

**Fase 3 — Modelo final para inferencia:**
- Reentrenar cada modelo base con todos los datos disponibles (igual que antes, usando `_train_final`).
- El meta-modelo se reentrenó ya sobre todo el out-of-fold. Para inferencia, pasa las probabilidades de los modelos base finales por el meta-modelo.

### 3.4 Integración con BaseTrainer

El meta-modelo se implementa como `MetaTrainer(BaseTrainer)`:

- `model_key = "meta"`
- `_after_features()`: no construye secuencias propias, sino que carga los artefactos de los modelos base que ya existen en disco.
- `_train_window(train_dates, test_dates)`:
  1. Para cada modelo base: cargar su trainer, entrenar en `train_dates`, predecir en `test_dates`.
  2. Apilar las probabilidades → `X_meta_test` de shape `(n_test, K)`.
  3. Lo mismo para el subconjunto de entrenamiento del meta-modelo (inner split del `train_dates`).
  4. Entrenar LogisticRegression sobre las probabilidades de entrenamiento.
  5. Predecir y evaluar con `evaluate_model`.
- `_train_final(final_dates, ...)`:
  1. Regenerar out-of-fold features sobre `final_dates` usando un split interno 80/20.
  2. Entrenar el meta-modelo final.
  3. Guardar artefacto que contiene el meta-modelo + los modelos base finales (o referencias a sus artefactos).

**Nota importante sobre el artefacto:**
El artefacto `meta_h1.pkl` debe guardar:
- El `LogisticRegression` meta-modelo.
- Los nombres de los modelos base usados (para reconstruir las features en inferencia).
- Los coeficientes (para interpretabilidad).

En inferencia (`predict.py`), el meta-modelo carga los artefactos base individuales, obtiene sus probabilidades, y pasa el vector al meta-modelo.

---

## 4. Qué NO se implementa (y por qué)

### 4.1 Meta-modelo para target continuo

Los modelos de regresión producen retornos predichos $\hat{r}$ en distintas escalas. Un RF de regresión no tiene sentido en este contexto (véase `decisions/continuous_target_decisions.md`), así que el conjunto de modelos base para el meta-modelo continuo sería solo {xgb, gru, lstm, cnn_gru, cnn_lstm}. La combinación lineal de retornos predichos (promedio ponderado) es técnicamente un meta-modelo de regresión, pero es trivialmente equivalente a un ensemble de promedio. No se implementa en esta iteración.

### 4.2 Feature augmentation del meta-modelo

Se podría pasar al meta-modelo no solo las probabilidades base sino también features del mercado (VIX percentil, régimen de volatilidad) para que aprenda "en qué régimen confiar en cada modelo". Esto añade valor pero también complejidad y riesgo de overfitting. Queda como trabajo futuro.

### 4.3 Stacking de múltiples niveles

Más de dos niveles (meta-meta-modelo) no tiene justificación empírica en este dominio y con estos volúmenes de datos. Wolpert (1992) y Ting & Witten (1999) muestran que los beneficios del stacking se capturan casi completamente en el primer nivel de apilamiento.

---

## 5. Expectativas de rendimiento

Basándose en la literatura, el meta-modelo debería mejorar la balanced accuracy media sobre el mejor modelo individual en **0.3 a 1.2 puntos porcentuales** (de ~53.5% a ~54-55%). Esto parece modesto, pero en predicción de dirección diaria donde el techo teórico está alrededor del 55-56% (Gu et al., 2020), una mejora de 0.5 puntos es significativa en términos económicos.

El IC (Information Coefficient) para el meta-modelo de clasificación puede obtenerse interpretando P(up) − 0.5 como retorno predicho y calculando la correlación de Spearman con los retornos reales. Se espera un IC ligeramente superior al de cualquier modelo individual.

---

## 6. Implementación: código de referencia

El siguiente código es la implementación de referencia. No debe copiarse directamente en un `.py` sin revisión, pero captura todas las decisiones de diseño.

### `models/meta/meta.py`

```python
"""
MetaTrainer -- ensemble de segundo nivel (stacking) sobre modelos base discretos.

Protocolo: out-of-fold stacking para evitar data leakage.
Meta-modelo: LogisticRegression(C=1.0) sobre probabilidades P(up) de los modelos base.
Modelos base: rf, xgb, gru, lstm, markov (configurables via BASE_MODELS).

Véase decisions/meta_model_decisions.md para la justificación completa.

Usage (standalone):
    python -m models.meta.meta --horizon 1 --ft-type macro --mode sliding

Usage (via framework):
    python -m models.train --model meta --horizon 1

Output: artifacts/meta_h{horizon}.pkl
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models.base import BaseTrainer
from models.evaluate import evaluate_model

# Modelos base que participan en el ensemble.
# Orden fijo para reproducibilidad: las columnas del meta-modelo tienen este orden.
BASE_MODELS = ["rf", "xgb", "gru", "lstm", "markov"]

META_PARAMS = {
    "C":           1.0,    # L2 moderada; suficiente con 5 inputs
    "max_iter":    1000,
    "random_state": 42,
}


class MetaTrainer(BaseTrainer):
    """Stacking ensemble sobre modelos base de clasificación discreta.

    Cada llamada a _train_window entrena los K modelos base sobre train_dates,
    genera sus probabilidades sobre test_dates (out-of-fold), y entrena la
    regresión logística meta sobre un inner split de train_dates.

    _train_final regenera out-of-fold sobre el 80% final, entrena el meta-modelo
    sobre ellas, y guarda el conjunto completo en el artefacto.
    """

    @property
    def model_key(self) -> str:
        return "meta"

    def _build_base_probas(self, train_dates, pred_dates) -> np.ndarray:
        """Entrena cada modelo base en train_dates y predice en pred_dates.

        Devuelve una matriz (n_pred, K) con P(up) de cada modelo base.
        """
        from models.trees.rf   import RFTrainer
        from models.trees.xgb  import XGBTrainer
        from models.neural.rnn_trainer import RNNTrainer
        from models.markov.markov import MarkovTrainer

        trainers = {
            "rf":     RFTrainer(horizon=self.horizon, ft_type=self.ft_type, mode=self.mode),
            "xgb":    XGBTrainer(horizon=self.horizon, ft_type=self.ft_type, mode=self.mode),
            "gru":    RNNTrainer(cell="gru", horizon=self.horizon, ft_type=self.ft_type, mode=self.mode),
            "lstm":   RNNTrainer(cell="lstm", horizon=self.horizon, ft_type=self.ft_type, mode=self.mode),
            "markov": MarkovTrainer(horizon=self.horizon, ft_type=self.ft_type, mode=self.mode),
        }

        all_probas = []
        for name in BASE_MODELS:
            t = trainers[name]
            # Compartir los datos ya cargados para evitar re-fetching
            t.X, t.y, t.dates, t.tickers = self.X, self.y, self.dates, self.tickers
            if hasattr(t, '_after_features'):
                t._after_features()

            tr_mask   = t.dates.isin(train_dates)
            pred_mask = t.dates.isin(pred_dates)

            # Para modelos neuronales, usar all_seqs si están disponibles
            if hasattr(t, 'all_seqs') and t.all_seqs is not None:
                from models.neural.lstm import SequenceDataset
                from torch.utils.data import DataLoader
                from models.neural.lstm import SEQ_LEN, BATCH_SIZE
                import torch

                tr_seq_mask   = np.isin(t.all_last_dates, train_dates)
                pred_seq_mask = np.isin(t.all_last_dates, pred_dates)
                seqs_tr = t.all_seqs[tr_seq_mask];  labs_tr = t.all_labels[tr_seq_mask]
                seqs_pr = t.all_seqs[pred_seq_mask]; labs_pr = t.all_labels[pred_seq_mask]

                from models.neural.lstm import _temporal_seq_split, _scale
                X_it, y_it, X_iv, y_iv = _temporal_seq_split(seqs_tr, labs_tr, t.all_last_dates[tr_seq_mask])
                X_it_s, X_iv_s, X_pr_s, _ = _scale(X_it, X_iv, seqs_pr)

                model, _ = t._fit(X_it_s, y_it, X_iv_s, y_iv)
                pr_dl = DataLoader(SequenceDataset(X_pr_s, labs_pr), batch_size=BATCH_SIZE)
                _, probas = t._predict(model, pr_dl)
            else:
                X_tr = t.X.loc[tr_mask]; y_tr = t.y.loc[tr_mask]
                X_pr = t.X.loc[pred_mask]

                # Entrenar modelo base puro (sin CV interno)
                if name == "rf":
                    from sklearn.ensemble import RandomForestClassifier
                    from models.trees.rf import RF_PARAMS
                    m = RandomForestClassifier(**RF_PARAMS)
                    m.fit(X_tr, y_tr)
                    probas = m.predict_proba(X_pr)[:, 1]
                elif name == "xgb":
                    import xgboost as xgb
                    from models.trees.xgb import XGB_PARAMS, _temporal_inner_split, EARLY_STOPPING_ROUNDS
                    X_it2, y_it2, X_iv2, y_iv2 = _temporal_inner_split(X_tr, y_tr, t.dates.loc[tr_mask])
                    m = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                    m.fit(X_it2, y_it2, eval_set=[(X_iv2, y_iv2)], verbose=False)
                    probas = m.predict_proba(X_pr)[:, 1]
                elif name == "markov":
                    from models.markov.markov import MarkovChain, MARKOV_PARAMS
                    m = MarkovChain(**MARKOV_PARAMS)
                    m.fit(X_tr, y_tr.values)
                    probas = m.predict_proba(X_pr)

            all_probas.append(probas)

        # Apilar: (K, n_pred) -> (n_pred, K)
        # IMPORTANTE: alinear por índice de fecha+ticker entre modelos
        return np.column_stack(all_probas)

    def _train_window(self, train_dates, test_dates) -> tuple:
        tr_mask = self.dates.isin(train_dates)
        te_mask = self.dates.isin(test_dates)

        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            return None, None

        # Inner split para entrenar el meta-modelo sin ver test_dates
        sorted_train   = np.sort(np.array(train_dates))
        split_idx      = int(len(sorted_train) * 0.8)
        inner_tr_dates = sorted_train[:split_idx]
        inner_va_dates = sorted_train[split_idx:]

        # Probabilidades out-of-fold de los modelos base sobre inner_va
        X_meta_va = self._build_base_probas(inner_tr_dates, inner_va_dates)
        va_mask   = self.dates.isin(inner_va_dates)
        y_va      = self.y.loc[va_mask].values[:len(X_meta_va)]

        # Probabilidades de los modelos base sobre test_dates
        X_meta_te = self._build_base_probas(train_dates, test_dates)
        y_te      = self.y.loc[te_mask].values[:len(X_meta_te)]

        # Entrenar meta-modelo sobre inner validation features
        meta = LogisticRegression(**META_PARAMS)
        meta.fit(X_meta_va, y_va)

        preds  = meta.predict(X_meta_te)
        probas = meta.predict_proba(X_meta_te)[:, 1]
        return evaluate_model(y_te, preds, probas), None

    def _train_final(self, final_dates, cv_summary, all_metrics, all_meta) -> dict:
        # Inner 80/20 para generar out-of-fold features del meta-modelo final
        sorted_dates = np.sort(final_dates)
        split_idx    = int(len(sorted_dates) * 0.8)
        meta_tr_dates = sorted_dates[:split_idx]
        meta_va_dates = sorted_dates[split_idx:]

        X_meta = self._build_base_probas(meta_tr_dates, meta_va_dates)
        va_mask = self.dates.isin(meta_va_dates)
        y_va    = self.y.loc[va_mask].values[:len(X_meta)]

        meta = LogisticRegression(**META_PARAMS)
        meta.fit(X_meta, y_va)

        print(f"  Meta-modelo coeficientes:")
        for name, coef in zip(BASE_MODELS, meta.coef_[0]):
            print(f"    {name:10s}: {coef:+.4f}")
        print(f"  Intercept: {meta.intercept_[0]:+.4f}")
        print(f"  Train rows (meta): {len(X_meta)}")

        return {
            "model":       meta,
            "base_models": BASE_MODELS,
            "features":    BASE_MODELS,
            "params":      META_PARAMS,
            "coef":        dict(zip(BASE_MODELS, meta.coef_[0].tolist())),
        }


def train_meta(horizon: int = 1, ft_type: str = "macro", mode: str = "sliding") -> dict:
    return MetaTrainer(horizon=horizon, ft_type=ft_type, mode=mode).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train meta-model (stacking ensemble)")
    parser.add_argument("--horizon", type=int, default=1,         help="Prediction horizon (days)")
    parser.add_argument("--ft-type", type=str, default="macro",   help="Feature type: micro | cross | macro")
    parser.add_argument("--mode",    type=str, default="sliding",  help="CV mode: sliding | expanding")
    args = parser.parse_args()

    from config import load_env
    load_env()

    train_meta(horizon=args.horizon, ft_type=args.ft_type, mode=args.mode)
```

### Registro en `models/train.py`

```python
from models.meta.meta import MetaTrainer

REGISTRY["meta"] = lambda **kw: MetaTrainer(**kw)
```

### `models/meta/__init__.py`

```python
# vacío
```

---

## 7. Nota sobre el coste de _build_base_probas dentro de CV

El punto más crítico de la implementación es el rendimiento. En cada ventana del CV (hay ~15-20 ventanas), `_build_base_probas` re-entrena los 5 modelos base. Para RF y XGBoost esto es rápido (~5-10s por modelo por ventana). Para GRU y LSTM, cada entrenamiento puede tardar 30-120s dependiendo del hardware. Con 20 ventanas × 2 modelos neuronales × ~60s = ~40 minutos solo para las ventanas neuronales.

**Mitigación:** La bandera `--model meta` en el CLI debe advertir al usuario de este coste. Como optimización futura, los artefactos de los modelos base por ventana podrían cachearse en disco (pickle por `(model_key, train_dates_hash)`), evitando re-entrenar en ejecuciones sucesivas.

---

## 8. Alineación de índices entre modelos base

Un riesgo sutil: los modelos neuronales operan sobre secuencias y su array de predicciones está indexado por `last_date`, no por índice de fila del DataFrame. La función `_build_base_probas` debe alinear correctamente las predicciones de todos los modelos base por `(fecha, ticker)` antes de apilarlas. Las predicciones de modelos de árbol están alineadas con `self.dates.isin(pred_dates)` mientras que las de los modelos neuronales están alineadas con `all_last_dates`. Si hay diferencias en el número de observaciones (ej: el modelo neuronal descarta las primeras `SEQ_LEN` filas de cada ticker), la alineación puede fallar silenciosamente. El código de referencia incluye `[:len(X_meta_te)]` como guard provisional; la implementación final debe alinear por join explícito en (fecha, ticker).

---

## Referencias

- **Breiman, L. (1996).** "Stacked regressions." *Machine Learning*, 24(1), 49-64. Paper original del stacking. Establece que los inputs del nivel 2 deben ser predicciones out-of-fold (no in-sample) para evitar leakage.

- **Gu, S., Kelly, B. & Xiu, D. (2020).** "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies*, 33(5), 2223-2273. Muestra que ensembles de modelos heterogéneos superan a modelos individuales en predicción de retornos de renta variable.

- **Krauss, C., Do, X.A. & Huck, N. (2017).** "Deep neural networks, gradient-boosted trees, random forests: statistical arbitrage on the S&P 500." *European Journal of Operational Research*, 259(2), 689-702. Documenta que ensembles árbol+red neuronal superan a cualquier modelo individual en predicción de dirección diaria.

- **Niculescu-Mizil, A. & Caruana, R. (2005).** "Predicting good probabilities with supervised learning." *Proceedings of ICML 2005*. Demuestra que la regresión logística produce probabilidades mejor calibradas que otros clasificadores en meta-nivel.

- **Ting, K.M. & Witten, I.H. (1999).** "Issues in stacked generalization." *Journal of Artificial Intelligence Research*, 10, 271-289. Analiza cuántos niveles de stacking son útiles: el primer nivel captura casi todo el beneficio; niveles adicionales apenas aportan.

- **Wolpert, D.H. (1992).** "Stacked generalization." *Neural Networks*, 5(2), 241-259. Paper fundacional del stacking moderno. Establece el protocolo out-of-fold como condición de validez estadística.
