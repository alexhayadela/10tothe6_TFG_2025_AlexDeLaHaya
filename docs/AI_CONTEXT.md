# AI_CONTEXT.md — Contexto completo del proyecto para asistentes IA

> Documento de referencia rápida. Permite a cualquier asistente IA comenzar a trabajar en este proyecto en < 2 minutos sin necesidad de explorar el codebase.
> Última actualización: 2026-04-21

---

## 1. Resumen ejecutivo

**Nombre del proyecto:** 10\*\*6 — Herramienta de análisis de inversión multifuente con ML para el IBEX35

**Autor:** Alex De La Haya Gutiérrez — TFG (Trabajo de Fin de Grado), Ingeniería Informática

**Repositorio:** `https://github.com/alexhayadela/10tothe6_TFG_2025_AlexDeLaHaya`

**Qué es:** Sistema integral de predicción bursátil para los 35 valores del IBEX35. Integra ingesta de datos de mercado, análisis de noticias con LLM, modelos de ML para predicción de dirección de precio, backtesting y un bot de trading. Incluye un newsletter automatizado y un website con predicciones actualizadas diariamente.

**Estado actual (rama `development-cl`):**
- Entregables 1-4 completamente implementados y en producción.
- Rama `development-cl`: modelos ML avanzados (XGBoost, GRU, LSTM, CNN+GRU, CNN+LSTM, Markov).
- Rama `dev-cl-continous`: extiende `development-cl` con soporte de target continuo (regresión) para XGB y redes neuronales.
- El modelo en producción (actualmente RF) genera predicciones diarias en Supabase y `data/predictions.json`.
- El trading bot (Entregable 5) está implementado pero requiere IB Gateway local; no se ejecuta en GitHub Actions.

---

## 2. Arquitectura técnica completa

### Estructura de directorios

```
/
├── .github/workflows/          # GitHub Actions (4 workflows)
│   ├── newsletter.yml          # Envío newsletter 09:00 UTC diario
│   ├── news.yml                # Ingesta de noticias 08:00 UTC diario
│   ├── ohlcv.yml               # Ingesta OHLCV 08:00 UTC (Tue-Sat, = lunes-viernes mercado)
│   └── preds.yml               # Cálculo y publicación predicciones 08:30 UTC (Mon-Fri)
│
├── artifacts/                  # Modelos entrenados serializados (.pkl)
│   └── rf_h1_full.pkl          # Random Forest h=1 (único artefacto en producción actual)
│
├── config.py                   # Rutas centralizadas + carga de variables de entorno
│
├── data/
│   └── predictions.json        # Predicciones del día actual (actualizadas por preds.yml)
│
├── db/
│   ├── __init__.py
│   ├── base.py                 # sqlite_connection() context manager
│   ├── migrations.py           # Inicializar/actualizar esquema SQLite local
│   ├── tables.txt              # Definición de tablas (referencia)
│   ├── utils_ohlcv.py          # get_ibex_tickers(), get_macro_tickers()
│   ├── sqlite/                 # Módulos de consulta SQLite
│   │   └── queries_ohlcv.py    # fetch_ohlcv(tickers, start, end)
│   └── supabase/               # Módulos de Supabase
│       ├── ingest_ohlcv.py     # Ingesta OHLCV diaria → Supabase
│       ├── ingest_news.py      # Ingesta de noticias → Supabase
│       └── upload_preds.py     # Calcula predicciones y las sube a Supabase + predictions.json
│
├── decisions/                  # Documentación de decisiones de diseño (leer antes de tocar modelos)
│   ├── rf_decisions.md         # RF hiperparámetros + métricas compartidas de todos los modelos
│   ├── xgboost_decisions.md    # XGBoost + early stopping strategy
│   ├── rnn_decisions.md        # GRU/LSTM, secuencias T=20, features, macro
│   ├── cnn_rnn_decisions.md    # CNN+GRU/CNN+LSTM, Conv1d, kernel=3, filtros=32
│   ├── markov_decisions.md     # Cadena de Markov, bins cuantílicos, Laplace smoothing
│   ├── features_decisions.md   # Feature engineering completo, 41 features, qué incluir/excluir
│   ├── backtesting_decisions.md# Protocolo walk-forward, costes transacción, execution assumptions
│   └── continuous_target_decisions.md # Target continuo (regresión) vs discreto, qué modelos soportan qué
│
├── docs/                       # GitHub Pages website + documentación TFG
│   ├── index.html              # Página principal del website estático
│   ├── script.js               # JS que consume predictions.json / Supabase API
│   └── style.css               # Estilos del website
│
├── llm/
│   ├── gpt_service.py          # Wrapper compatible OpenAI → Groq API (openai/gpt-oss-120b)
│   └── rate_limit.py           # Control de tasa para la API de Groq
│
├── models/
│   ├── base.py                 # BaseTrainer: template method, CV walk-forward, serialización
│   ├── evaluate.py             # evaluate_model() (clasificación) + evaluate_regression() (regresión)
│   ├── train.py                # Entry point CLI: python -m models.train --model rf --horizon 1
│   ├── predict.py              # Genera predicciones del modelo en producción
│   ├── arima/                  # Notebooks ARIMA (price.ipynb, log_returns.ipynb)
│   ├── garch/                  # Notebooks GARCH (log_returns.ipynb)
│   ├── markov/
│   │   └── markov.py           # MarkovTrainer (hereda BaseTrainer)
│   ├── neural/
│   │   ├── gru.py              # GRUTrainer
│   │   ├── lstm.py             # LSTMTrainer
│   │   ├── cnn_rnn.py          # CNNGRUTrainer + CNNLSTMTrainer
│   │   └── rnn_trainer.py      # RNNBaseTrainer (base compartida para GRU/LSTM/CNN)
│   └── trees/
│       ├── features.py         # ml_ready(): genera la matriz de 41 features desde OHLCV crudo
│       ├── rf.py               # RFTrainer
│       └── xgb.py              # XGBTrainer
│
├── news/
│   ├── news_rss.py             # Fetch y filtrado de feeds RSS de Expansión
│   └── classification.py      # Clasificación LLM de noticias (categoría, sentimiento, entidades)
│
├── newsletter/
│   ├── send.py                 # Compone y envía newsletter HTML por email (SMTP Gmail)
│   ├── eg.html                 # Plantilla HTML de ejemplo del newsletter
│   └── eg_gmail.html           # Variante Gmail-compatible
│
├── notebooks/                  # Notebooks Jupyter de exploración y análisis
│
├── requirements/               # Ficheros de dependencias por componente
│   ├── all.txt                 # Todas las dependencias (desarrollo local)
│   ├── newsletter_auto.txt     # Solo para el workflow de newsletter
│   ├── news_auto.txt           # Solo para el workflow de noticias
│   ├── ohlcv_auto.txt          # Solo para el workflow de ingesta OHLCV
│   └── preds_auto.txt          # Solo para el workflow de predicciones
│
├── trading/
│   ├── execute.py              # Bot de trading: abre/cierra posiciones con ib_async
│   ├── params.py               # Parámetros del bot (stop_loss, max_pos_pct, etc.)
│   ├── test.py                 # Tests del bot
│   └── bot.ps1                 # Script PowerShell para Windows Task Scheduler
│
└── venv/                       # Entorno virtual Python (no versionado)
```

---

## 3. Estado de las ramas git

| Rama | Estado | Contenido |
|------|--------|-----------|
| `main` | Estable, producción | Entregables 1-5 completos, RF en producción, website activo |
| `development-cl` | Activo desarrollo | Añade XGBoost, GRU, LSTM, CNN+GRU, CNN+LSTM, Markov. Framework BaseTrainer con soporte multi-modelo. **Rama actual de trabajo.** |
| `dev-cl-continous` | Ramificado de `development-cl` | Extiende con `target_type="continuous"` (regresión). XGB usa `objective="reg:squarederror"`. Redes neuronales usan HuberLoss. Nuevas métricas: IC, MAE, RMSE, R². |

**Rama activa por defecto en este entorno:** `development-cl`

**Para cambiar a la rama de target continuo:**
```bash
git checkout dev-cl-continous
```

---

## 4. Modelos ML: tabla completa

| model_key | Archivo implementación | target_type | Artifact (.pkl) | Hiperparámetros clave |
|-----------|----------------------|-------------|-----------------|----------------------|
| `rf` | `models/trees/rf.py` | discrete only | `rf_h{h}.pkl` | n_est=500, depth=5, max_feat=0.3, min_leaf=50 |
| `xgb` | `models/trees/xgb.py` | discrete + continuous | `xgb_h{h}.pkl` / `xgb_h{h}_cont.pkl` | n_est=300, lr=0.05, depth=3, mcw=100, sub=0.7, ES=30 |
| `gru` | `models/neural/gru.py` | discrete + continuous | `gru_h{h}.pkl` / `gru_h{h}_cont.pkl` | hidden=64, T=20, F=41, dropout=0.3, HuberLoss(δ=0.01) |
| `lstm` | `models/neural/lstm.py` | discrete + continuous | `lstm_h{h}.pkl` / `lstm_h{h}_cont.pkl` | hidden=64, T=20, F=41, dropout=0.3, HuberLoss(δ=0.01) |
| `cnn_gru` | `models/neural/cnn_rnn.py` | discrete + continuous | `cnn_gru_h{h}.pkl` / `cnn_gru_h{h}_cont.pkl` | Conv1d(32, k=3) + GRU(64), T=20 |
| `cnn_lstm` | `models/neural/cnn_rnn.py` | discrete + continuous | `cnn_lstm_h{h}.pkl` / `cnn_lstm_h{h}_cont.pkl` | Conv1d(32, k=3) + LSTM(64), T=20 |
| `markov` | `models/markov/markov.py` | discrete only | `markov_h{h}.pkl` | n_states=3, quantile bins, Laplace smoothing |

**Convención de artifact filename:**
- `{model_key}_h{horizon}.pkl` → clasificación discreta (target_type="discrete")
- `{model_key}_h{horizon}_cont.pkl` → regresión continua (target_type="continuous")
- Todos en directorio `artifacts/` (path configurado en `config.py` como `ARTIFACTS_PATH`)

**Formato del artefacto (joblib pkl dict):**
```python
{
    "model_key": str,          # e.g. "rf"
    "horizon": int,            # e.g. 1
    "ft_type": str,            # "macro" | "cross" | "micro"
    "mode": str,               # "sliding" | "expanding"
    "target_type": str,        # "discrete" | "continuous"
    "window_days": int,        # días de entrenamiento del modelo final
    "train_start": str,        # fecha inicio entrenamiento final (YYYY-MM-DD)
    "train_end": str,          # fecha fin entrenamiento final
    "cv_metrics": list[dict],  # una entrada por ventana CV
    "cv_summary": dict,        # media/std de cada métrica en CV
    "model": object,           # el modelo entrenado (sklearn/xgb/torch state dict)
    "features": list[str],     # lista de feature names en orden
    "params": dict,            # hiperparámetros usados
    # + campos modelo-específicos (feature_importances_ para RF, best_iteration para XGB, etc.)
}
```

---

## 5. Pipeline de datos

### Flujo completo desde ingesta hasta predicción

```
[yfinance API]
    │
    ▼
[db/supabase/ingest_ohlcv.py] ──→ [Supabase tabla 'ohlcv'] ← website / bot
    │
    ▼
[db/sqlite/] (SQLite local)
    │
    ▼
[models/trees/features.py :: ml_ready()]
    │  inputs: df_micro_raw (IBEX35 OHLCV), df_macro_raw (^IBEX, ^GSPC, ^VIX)
    │  outputs: df_features, X (n×41 DataFrame), y_discrete (Series 0/1), mask (bool), y_cont (Series float)
    │  features: 28 micro + 2 cross (breadth) + 9 macro + 2 dow_cyclic = 41 total
    ▼
[models/base.py :: BaseTrainer.run()]
    │  1. load_raw() → fetch OHLCV desde SQLite
    │  2. build_features() → ml_ready(), selecciona y según target_type
    │  3. _after_features() → neural trainers construyen secuencias (n, T=20, F=41)
    │  4. CV loop → sliding_windows(750d, step=63d, embargo=1d)
    │  5. _train_window() → entrena 1 fold, devuelve métricas
    │  6. aggregate CV metrics
    │  7. _train_final() → entrena con últimos 750d (sliding) o todos los datos (expanding)
    │  8. serialize artifact → artifacts/{model_key}_h{horizon}[_cont].pkl
    ▼
[models/predict.py] ← consume el artifact pkl
    │
    ▼
[db/supabase/upload_preds.py]
    │  ── sube a Supabase tabla 'predictions'
    └── actualiza data/predictions.json
         │
         ▼
     [GitHub Pages / website] ← Lee predictions.json o Supabase API
```

### Función ml_ready() — detalles

```python
# models/trees/features.py
df, X, y_discrete, mask, y_cont = ml_ready(
    horizon=1,           # h=1 o h=5
    df_micro=df_micro,   # OHLCV de los 35 valores IBEX35
    df_macro=df_macro,   # OHLCV de ^IBEX, ^GSPC, ^VIX (None para ft_type != "macro")
    ft_type="macro"      # "micro" | "cross" | "macro"
)
# X: DataFrame (n_rows × 41), ya limpio (sin NaN, sin columnas de target)
# y_discrete: Series de 0/1 (future_log_ret > 0)
# y_cont: Series de float (future_log_ret crudo)
# mask: índice booleano de filas válidas en df
```

---

## 6. Decisiones de diseño clave

### decisions/rf_decisions.md
- RF con 500 árboles, depth=5, max_features=0.3 (~9/41 features por split), min_samples_leaf=50.
- max_features=0.3 en lugar de "sqrt" porque las features financieras están altamente correlacionadas; 0.3 compensa la baja dimensionalidad efectiva.
- min_samples_leaf=50 para probabilidades estables (SE≈0.07); con 25 (valor anterior) era demasiado ruidoso.
- **Métricas compartidas (aplican a TODOS los modelos):** balanced_accuracy (primaria), ROC-AUC, log-loss, accuracy, MCC. Función unificada: `models/evaluate.py::evaluate_model()`.
- Benchmark de la literatura: 52-55% balanced accuracy para predicción diaria de renta variable (Krauss et al., 2017).

### decisions/xgboost_decisions.md
- Early stopping (paciencia=30) es el mecanismo de regularización más importante. Sin él, XGB sobreajusta en rondas tardías.
- División temporal 80/20 dentro de la ventana de entrenamiento para el early stopping: los 80% más antiguos entrenan, el 20% más reciente valida.
- eval_metric="logloss" (detecta sobreajuste antes que accuracy).
- depth=3 (más conservador que RF=5; boosting compensa por iteración, no por profundidad).
- Two-step final training para el modelo de producción: (1) determinar best_iteration con early stopping en subconjunto, (2) reentrenar con todos los datos usando ese número de rondas.

### decisions/rnn_decisions.md
- Usar features pre-calculadas (las mismas 41 que RF/XGB), NO inputs raw de precio.
- Secuencias de T=20 timesteps (~1 mes de trading). Cada timestep es un vector de 41 features.
- GRU vs LSTM: diferencia de rendimiento marginal; ambos se entrenan para comparar.
- `dow` se codifica cíclicamente: `dow_sin = sin(2π*dow/5)`, `dow_cos = cos(2π*dow/5)`. Resultado: 41 features (40 del árbol + 1 extra por codificación cíclica).
- Para regresión: HuberLoss(delta=0.01) en lugar de MSE por las colas pesadas de retornos.

### decisions/cnn_rnn_decisions.md
- 1 sola capa Conv1D (no 2+). Con T=20, secuencias cortas. Features ya son de alto nivel.
- Conv1d: in_channels=41, out_channels=32, kernel_size=3, padding=1.
- La Conv1d detecta patrones multi-feature en ventanas de 3 días consecutivos. La RNN posterior captura cómo esos patrones evolucionan a lo largo de la secuencia.
- Ganancia esperada sobre RNN pura: 1-3 puntos porcentuales de balanced accuracy.

### decisions/markov_decisions.md
- State variable: `log_ret_1` discretizado en 3 bins cuantílicos (no umbrales fijos).
- Bins cuantílicos garantizan frecuencias iguales en cada estado → estimaciones más estables.
- Suavizado de Laplace para evitar probabilidades cero en transiciones no observadas.
- Rol: baseline interpretable + sanity check del pipeline (debe reproducir reversión a corto plazo documentada por Lo & MacKinlay, 1988).

### decisions/features_decisions.md
- 41 features total para h=1: 28 micro + 2 cross + 9 macro + 2 dow_cíclico.
- Features eliminadas por redundancia: `log_ret_3` (≈0.85 corr con `log_ret_5`), `ema_ratio_5_20` (≈0.95 corr con `sma_ratio_5_20`), `stoch_k` (redundante con RSI), `slope_20` (redundante con `sma_ratio_10_50`).
- **IMPORTANTE:** `ibx_breadth` se calcula como leave-one-out (excluyendo el propio ticker) para evitar look-ahead bias.
- Todas las features usan solo datos disponibles al cierre del día t (ninguna centra ventanas).

### decisions/backtesting_decisions.md
- Walk-forward con ventana deslizante: WINDOW_DAYS=750 (~3 años), STEP_DAYS=63 (~trimestral), embargo=1 día.
- Ejecutar en T+1 apertura: señal generada al cierre de t, ejecución en apertura de t+1.
- Costes de transacción por tiers: 10 bps (top-10), 15 bps (siguiente 10), 20 bps (resto) — ida y vuelta.
- Escaler (StandardScaler) siempre fit en datos de entrenamiento, transform en test. Nunca fit_transform en test.

### decisions/continuous_target_decisions.md
- **RF**: NO soporta regresión (incapaz de extrapolar; las hojas son promedios de entrenamiento).
- **XGB**: Sí. Cambio mínimo: `objective="reg:squarederror"`. Early stopping y two-step idénticos.
- **GRU/LSTM/CNN+GRU/CNN+LSTM**: Sí. Cambio: reemplazar BCEWithLogitsLoss por HuberLoss(delta=0.01); eliminar sigmoid de la salida.
- **Markov**: NO. No hay extensión natural a continuo sin rediseñar el modelo (HMM).
- Evaluación regresión: `evaluate_regression()` reporta MAE, RMSE, R², directional_accuracy, IC (Spearman).

---

## 7. Cómo ejecutar cada componente

### Entrenar un modelo

```bash
# Activar entorno virtual
venv\Scripts\activate

# Entrenamiento via CLI (entry point único)
python -m models.train --model rf     --horizon 1 --mode sliding
python -m models.train --model xgb    --horizon 1 --mode sliding
python -m models.train --model gru    --horizon 1 --mode sliding
python -m models.train --model lstm   --horizon 1 --mode sliding
python -m models.train --model cnn_gru  --horizon 1 --mode sliding
python -m models.train --model cnn_lstm --horizon 1 --mode sliding
python -m models.train --model markov --horizon 1 --mode sliding

# Target continuo (solo en rama dev-cl-continous)
python -m models.train --model xgb --horizon 1 --target-type continuous
python -m models.train --model gru --horizon 1 --target-type continuous

# También se pueden llamar directamente:
python -m models.trees.rf
python -m models.trees.xgb
python -m models.neural.gru
python -m models.neural.lstm
```

### Generar predicciones y subir a Supabase

```bash
# Requiere variables de entorno: SUPABASE_API_KEY, SUPABASE_URL
python -m db.supabase.upload_preds
```

### Ingestar datos OHLCV

```bash
python -m db.supabase.ingest_ohlcv    # sube a Supabase
python -m db.migrations               # sincroniza SQLite local
```

### Ingestar noticias

```bash
# Requiere: GROQ_API_KEY, SUPABASE_API_KEY, SUPABASE_URL
python -m db.supabase.ingest_news
```

### Enviar newsletter

```bash
# Requiere: EMAIL_USER, EMAIL_PASSWORD, SUPABASE_API_KEY, SUPABASE_URL
python -m newsletter.send
```

### Trading bot

```bash
# Requiere IB Gateway corriendo en localhost:4002 (paper trading)
# Ejecutar como administrador en Windows:
.\trading\bot.ps1

# Para detener tareas programadas:
Unregister-ScheduledTask -TaskName "Open Positions" -Confirm:$false
Unregister-ScheduledTask -TaskName "Close Positions" -Confirm:$false
```

### Website local

```bash
python -m http.server 8000
# Abrir: http://localhost:8000/docs/
```

---

## 8. Base de datos

### Supabase (PostgreSQL cloud)

**URL:** Configurada en `SUPABASE_URL` (env var).
**Autenticación:** `SUPABASE_API_KEY` (service role key).

| Tabla | Columnas | Constraints | Uso |
|-------|----------|-------------|-----|
| `ohlcv` | ticker TEXT, date DATE, open FLOAT8, high FLOAT8, low FLOAT8, close FLOAT8, volume FLOAT8 | PK (ticker, date) | Precios históricos IBEX35 + macroeconómicos |
| `news` | id INT8, date DATE, title TEXT, section TEXT, body TEXT, url TEXT, category TEXT, relevance FLOAT8, sentiment TEXT | PK: id; UNIQUE: url | Noticias clasificadas por LLM |
| `news_entities` | news_id INT8, ticker TEXT | PK+FK: news_id → news.id | Relación noticia-empresa |
| `newsletter` | id INT8, created_at TIMESTAMP, email TEXT | PK: id | Lista de suscriptores |
| `predictions` | id INT8, ticker TEXT, date DATE, pred BOOL, proba FLOAT8, model TEXT | PK: id | Predicciones ML del día |

Todos los campos excepto PK son nullable.

### SQLite (local)

**Path:** Configurado en `config.py` como `SQLITE_PATH` (por defecto `db/sqlite/market.db`).
**Acceso:** `db/base.py::sqlite_connection()` (context manager).

Contiene únicamente la tabla `ohlcv` con el mismo esquema que Supabase, optimizada para lecturas masivas durante el entrenamiento de modelos. No contiene noticias, suscriptores ni predicciones.

### Patrones de acceso

```python
# Leer OHLCV desde SQLite (durante entrenamiento)
from db.base import sqlite_connection
from db.sqlite.queries_ohlcv import fetch_ohlcv
from db.utils_ohlcv import get_ibex_tickers, get_macro_tickers

with sqlite_connection() as conn:
    df = fetch_ohlcv(get_ibex_tickers())   # todos los IBEX35

# Escribir predicciones a Supabase (ver upload_preds.py para ejemplo completo)
```

---

## 9. Automatizaciones GitHub Actions

| Workflow | Archivo | Schedule (UTC) | Días | Comando |
|----------|---------|---------------|------|---------|
| Newsletter | `.github/workflows/newsletter.yml` | `0 9 * * *` (09:00) | Todos | `python -m newsletter.send` |
| Noticias | `.github/workflows/news.yml` | `0 8 * * *` (08:00) | Todos | `python -m db.supabase.ingest_news` |
| OHLCV | `.github/workflows/ohlcv.yml` | `0 8 * * TUE-SAT` (08:00) | Mar-Sáb (= Lun-Vie mercado) | `python -m db.supabase.ingest_ohlcv` |
| Predicciones | `.github/workflows/preds.yml` | `30 8 * * MON-FRI` (08:30) | Lun-Vie | `python -m db.supabase.upload_preds` |

**Notas importantes:**
- `ohlcv.yml` corre Martes-Sábado porque ingesta datos del día laborable anterior (el mercado cierra L-V, los datos están disponibles al día siguiente).
- `preds.yml` tiene permiso `contents: write` para poder hacer git commit de `data/predictions.json` y actualizar el website.
- Todos los workflows usan `workflow_dispatch` para poder lanzarse manualmente.
- Los workflows corren en `ubuntu-latest` con Python 3.10.
- Credenciales configuradas como GitHub Secrets: `EMAIL_USER`, `EMAIL_PASSWORD`, `GROQ_API_KEY`, `SUPABASE_API_KEY`, `SUPABASE_URL`.

---

## 10. Variables de entorno necesarias

| Variable | Dónde se usa | Cómo obtenerla |
|----------|-------------|----------------|
| `EMAIL_USER` | newsletter/send.py | Tu dirección @gmail.com |
| `EMAIL_PASSWORD` | newsletter/send.py | Google App Password (no la contraseña normal) |
| `GROQ_API_KEY` | llm/gpt_service.py | console.groq.com/keys |
| `SUPABASE_API_KEY` | db/supabase/*.py | Supabase Dashboard → Settings → API → service_role key |
| `SUPABASE_URL` | db/supabase/*.py | Supabase Dashboard → Settings → API → Project URL |

**Fichero .env (desarrollo local):**
```
EMAIL_USER=tu@gmail.com
EMAIL_PASSWORD=xxxx xxxx xxxx xxxx
GROQ_API_KEY=gsk_...
SUPABASE_API_KEY=sb_secret_...
SUPABASE_URL=https://<project-id>.supabase.co
```

El fichero `.env` se carga mediante `config.py::load_env()`. Esta función debe llamarse explícitamente en los entry points (ver `if __name__ == "__main__":` en rf.py como ejemplo).

---

## 11. Lo que falta / trabajo pendiente conocido

### Funcionalidad pendiente (documentada en ramas de desarrollo)

1. **Experimentos de evaluación completos:** Las tablas de resultados en el TFG tienen marcadores `[PENDIENTE — ejecutar experimentos]`. Hay que entrenar todos los modelos, ejecutar el CV walk-forward completo y rellenar las tablas.

2. **Integración de features de noticias en modelos ML:** Las noticias se clasifican y almacenan pero no se usan como features en los modelos ML. La integración requiere: sincronización temporal (noticias de t disponibles como feature para predecir t+1), agregación por ticker (máxima relevancia del día, sentimiento promedio), y gestión de días sin noticias (imputación por cero o valor por defecto).

3. **Calibración de probabilidades:** XGBoost produce probabilidades menos calibradas que RF. Implementar Platt scaling (`sklearn.calibration.CalibratedClassifierCV`) post-hoc mejoraría el position sizing basado en probabilidades.

4. **Modelos h=5 (horizonte semanal):** El framework soporta `--horizon 5` pero los modelos h=5 no están entrenados ni evaluados. Las features para h=5 son diferentes (ver decisions/features_decisions.md: excluir features muy lentas para h=1 pero incluirlas para h=5).

5. **SHAP values / interpretabilidad:** No hay análisis de importancia de features con SHAP implementado. Solo MDI (Mean Decrease Impurity) del Random Forest.

6. **Test suite:** No hay tests unitarios automatizados más allá de `trading/test.py`. Los modelos no tienen tests de regresión que detecten cambios de comportamiento.

### Limitaciones conocidas del sistema actual

- **Sesgo de supervivencia:** El universo usa solo los actuales 35 componentes del IBEX35; empresas excluidas/quebradas no están incluidas.
- **Sin impacto de mercado:** Los costes de transacción son fijos por tier; no modelizan el impacto de grandes órdenes.
- **Modelo global único:** No hay modelos por sector. Todos los tickers comparten los mismos pesos del modelo ML.
- **Newsletter todos los días (incluidos festivos):** El cron de newsletter y noticias corre todos los días, incluyendo festivos bursátiles. No hay lógica de calendario que lo evite.

---

## 12. Convenciones de código

### Patrones que seguir obligatoriamente

**1. Nuevos modelos ML → heredar BaseTrainer**
```python
# models/trees/nuevo_modelo.py
from models.base import BaseTrainer

class NuevoModeloTrainer(BaseTrainer):
    @property
    def model_key(self) -> str:
        return "nuevo"  # determina el nombre del artifact
    
    def _train_window(self, train_dates, test_dates) -> tuple:
        # ... train
        return evaluate_model(y_te, preds, probas), meta  # o evaluate_regression()
    
    def _train_final(self, final_dates, cv_summary, all_metrics, all_meta) -> dict:
        # ... entrena modelo final
        return {"model": model, "features": list(X.columns), "params": PARAMS}
```

**2. Métricas → siempre usar evaluate_model() de models/evaluate.py**
```python
from models.evaluate import evaluate_model, evaluate_regression
# NUNCA calcular métricas manualmente en el trainer; usar las funciones compartidas
metrics = evaluate_model(y_true, y_pred, y_proba)          # clasificación
metrics = evaluate_regression(y_true, y_pred_cont)          # regresión
```

**3. Rutas → siempre usar config.py**
```python
from config import ARTIFACTS_PATH, SQLITE_PATH, load_env
# NUNCA usar rutas hardcodeadas como "artifacts/" o "./db/market.db"
```

**4. Acceso a base de datos → context managers**
```python
from db.base import sqlite_connection
with sqlite_connection() as conn:
    df = pd.read_sql("SELECT ...", conn)
# NUNCA dejar conexiones abiertas o usar rutas SQLite directamente
```

**5. Features → solo via ml_ready()**
```python
from models.trees.features import ml_ready
df, X, y_discrete, mask, y_cont = ml_ready(horizon, df_micro, df_macro, ft_type)
# NUNCA calcular features ad hoc fuera de esta función
```

**6. Supabase → credenciales solo desde env vars**
```python
import os
supabase_url = os.environ["SUPABASE_URL"]
supabase_key = os.environ["SUPABASE_API_KEY"]
# NUNCA hardcodear credenciales
```

### Convenciones de naming

- Modelos: `snake_case` para model_key (`cnn_gru`, `cnn_lstm`, `rf`)
- Artifacts: `{model_key}_h{horizon}[_cont].pkl`
- Features: `snake_case` descriptivo (`log_ret_1`, `vol_ratio_5_20`, `ibx_breadth`)
- Trainers: `{NombreModelo}Trainer` (e.g. `RFTrainer`, `XGBTrainer`, `GRUTrainer`)
- Workflows: `{componente}.yml` en minúsculas (`newsletter.yml`, `preds.yml`)

### Convenciones de documentación

- Cada trainer debe tener docstring que indique qué decisions/*.md referencia.
- Los hiperparámetros deben estar en un dict con nombre (`RF_PARAMS`, `XGB_PARAMS`) con un comentario inline por cada uno.
- Las funciones de producción (train, predict, ingest) deben tener al menos un bloque de docstring con parámetros y output.

### Lo que NO hacer

- NO usar k-fold CV aleatorio para datos de series temporales. Siempre walk-forward.
- NO centrar ventanas de features (incluye datos futuros). Todas las ventanas deben ser trailing.
- NO hacer `scaler.fit_transform(X_test)`. Siempre fit en train, transform en test.
- NO omitir el embargo en walk-forward (1 día de separación entre fin de train y inicio de test).
- NO añadir features con alta correlación entre sí sin justificación (ver decisions/features_decisions.md).
- NO entrenar el modelo de producción con datos del período de test (leakage).

---

## 13. Referencia rápida de imports más usados

```python
# Config y rutas
from config import ARTIFACTS_PATH, SQLITE_PATH, load_env

# Base de datos
from db.base import sqlite_connection
from db.sqlite.queries_ohlcv import fetch_ohlcv
from db.utils_ohlcv import get_ibex_tickers, get_macro_tickers

# Features
from models.trees.features import ml_ready

# BaseTrainer + ventanas CV
from models.base import BaseTrainer, sliding_windows, expanding_windows
from models.base import WINDOW_DAYS, STEP_DAYS  # 750, 63

# Evaluación (compartida por TODOS los modelos)
from models.evaluate import evaluate_model       # clasificación discreta
from models.evaluate import evaluate_regression  # regresión continua

# Modelos individuales
from models.trees.rf import RFTrainer, RF_PARAMS, train_rf
from models.trees.xgb import XGBTrainer, XGB_PARAMS
from models.neural.gru import GRUTrainer
from models.neural.lstm import LSTMTrainer
from models.neural.cnn_rnn import CNNGRUTrainer, CNNLSTMTrainer
from models.markov.markov import MarkovTrainer

# LLM
from llm.gpt_service import GPTService
```
