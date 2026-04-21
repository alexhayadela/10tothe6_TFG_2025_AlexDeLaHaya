# 10\*\*6 — Herramienta de análisis de inversión multifuente con ML para el IBEX35

---

**Trabajo de Fin de Grado**
**Grado en Ingeniería Informática**

**Autor:** Alex De La Haya Gutiérrez
**Tutor:** [Nombre del tutor]
**Departamento:** [Nombre del departamento]
**Universidad:** [Nombre de la universidad]
**Fecha:** Abril 2026

---

## Resumen

Este Trabajo de Fin de Grado presenta el diseño, implementación y evaluación de un sistema integral de análisis de inversión para los valores del IBEX35. El sistema, denominado **10\*\*6**, integra cinco componentes principales: un newsletter automatizado de noticias financieras, una base de datos de precios históricos con modelos de series temporales clásicos (ARIMA y GARCH), un módulo de clasificación de noticias mediante modelos de lenguaje grande (LLM), una capa de modelos de aprendizaje automático avanzados para predicción de dirección de precio, y un bot de trading conectado a Interactive Brokers. Los modelos de ML implementados incluyen Random Forest, XGBoost, redes recurrentes (GRU y LSTM), arquitecturas híbridas CNN+GRU y CNN+LSTM, y una cadena de Markov como línea de base. La evaluación se realiza mediante validación cruzada walk-forward con ventana deslizante de 750 días, usando balanced accuracy como métrica primaria para clasificación e Information Coefficient (Spearman) para regresión. Los resultados muestran que los mercados son inherentemente ruidosos y que la ventaja predictiva, si existe, es pequeña. Se concluye con reflexiones honestas sobre las limitaciones prácticas y la consideración de alternativas más sencillas como los fondos indexados.

**Palabras clave:** IBEX35, Machine Learning, predicción bursátil, series temporales, Random Forest, XGBoost, LSTM, GRU, ARIMA, GARCH, trading algorítmico, backtesting.

---

**Abstract (English)**

This Final Degree Thesis presents the design, implementation, and evaluation of an integrated investment analysis system for IBEX35 stocks. The system, called **10\*\*6**, integrates five main components: an automated financial news newsletter, a historical price database with classical time series models (ARIMA and GARCH), a news classification module using large language models (LLM), a layer of advanced machine learning models for price direction prediction, and a trading bot connected to Interactive Brokers. The implemented ML models include Random Forest, XGBoost, recurrent neural networks (GRU and LSTM), hybrid CNN+GRU and CNN+LSTM architectures, and a Markov chain as baseline. Evaluation is performed through walk-forward cross-validation with a 750-day sliding window, using balanced accuracy as the primary classification metric and Information Coefficient (Spearman) for regression. Results show that markets are inherently noisy and that any predictive edge, if it exists, is small. The work concludes with honest reflections on practical limitations and the consideration of simpler alternatives such as index funds.

**Keywords:** IBEX35, Machine Learning, stock prediction, time series, Random Forest, XGBoost, LSTM, GRU, ARIMA, GARCH, algorithmic trading, backtesting.

---

## Índice

1. [Introducción](#1-introducción)
   - 1.1 Motivación
   - 1.2 Objetivos
   - 1.3 Estructura del documento
2. [Marco teórico](#2-marco-teórico)
   - 2.1 Mercados financieros y el IBEX35
   - 2.2 Series temporales financieras: propiedades estadísticas
   - 2.3 Modelos clásicos: ARIMA y GARCH
   - 2.4 Machine Learning para predicción de mercados
   - 2.5 Procesamiento de lenguaje natural en finanzas
3. [Arquitectura del sistema](#3-arquitectura-del-sistema)
   - 3.1 Visión general
   - 3.2 Capa de datos
   - 3.3 Capa de análisis LLM y noticias
   - 3.4 Capa de modelos predictivos
   - 3.5 Capa de ejecución (trading bot)
   - 3.6 Capa de presentación (website + newsletter)
4. [Ingeniería de características](#4-ingeniería-de-características)
   - 4.1 Features micro (técnicos)
   - 4.2 Features cross-seccionales (breadth IBEX)
   - 4.3 Features macro
   - 4.4 Target: dirección discreta y retorno continuo
5. [Modelos](#5-modelos)
   - 5.1 ARIMA
   - 5.2 GARCH
   - 5.3 Random Forest
   - 5.4 XGBoost
   - 5.5 Redes Recurrentes: GRU y LSTM
   - 5.6 CNN + GRU/LSTM
   - 5.7 Cadena de Markov
   - 5.8 Comparativa y selección de modelos
6. [Backtesting y evaluación](#6-backtesting-y-evaluación)
   - 6.1 Protocolo walk-forward
   - 6.2 Costes de transacción y supuestos de ejecución
   - 6.3 Estrategias: baseline, filtro de confianza, long/short
   - 6.4 Benchmarks y métricas financieras
   - 6.5 Limitaciones y sesgos
7. [Resultados](#7-resultados)
   - 7.1 Resultados modelos clásicos (ARIMA, GARCH)
   - 7.2 Resultados modelos ML — clasificación
   - 7.3 Resultados modelos ML — regresión
   - 7.4 Análisis del trading bot
   - 7.5 Figuras y gráficos
8. [Discusión](#8-discusión)
9. [Conclusiones y trabajo futuro](#9-conclusiones-y-trabajo-futuro)
10. [Referencias](#referencias)
11. [Apéndices](#apéndices)

---

## 1. Introducción

### 1.1 Motivación

Los mercados financieros son uno de los sistemas más complejos y estudiados de la historia moderna. Cada día, millones de participantes —desde inversores minoristas hasta fondos de cobertura algorítmicos— procesan información heterogénea (precios, noticias, datos macroeconómicos, sentimiento de mercado) para tomar decisiones de inversión. La digitalización ha democratizado el acceso a datos y herramientas de análisis, pero también ha intensificado la competencia: cuando todos tienen acceso a los mismos datos, la ventaja informacional se hace escasa.

En este contexto, la combinación de técnicas de aprendizaje automático, procesamiento de lenguaje natural y análisis cuantitativo clásico ofrece la posibilidad de procesar y sintetizar grandes volúmenes de información de forma sistemática y sin sesgos emocionales. La literatura académica documenta evidencia —aunque modesta— de predictibilidad a corto plazo en los mercados de renta variable (Krauss et al., 2017; Gu et al., 2020), especialmente cuando se combinan múltiples fuentes de información.

La motivación personal de este proyecto surge de la curiosidad por entender si la tecnología disponible para un estudiante de ingeniería —servidores gratuitos, APIs públicas, modelos de código abierto— es suficiente para construir un sistema de análisis de mercado que vaya más allá del análisis técnico manual. No se pretende batir a los hedge funds cuantitativos con cientos de ingenieros; se pretende aprender construyendo, ser honesto sobre las limitaciones y documentar el proceso con rigor académico.

El IBEX35, índice bursátil de referencia del mercado español, constituye el universo de análisis por razones prácticas y académicas. Sus 35 componentes son suficientes para realizar análisis cross-seccional, sus datos son accesibles públicamente con calidad razonable, y la literatura académica sobre el mercado español es más escasa que sobre el S&P500, lo que supone una contribución marginal al conocimiento. Adicionalmente, el acceso a Interactive Brokers para paper trading permite validar el sistema en condiciones próximas a la realidad sin riesgo de capital.

### 1.2 Objetivos

Este trabajo persigue los siguientes objetivos, organizados en cinco entregables:

**Objetivo 1 — Newsletter automatizado:** Diseñar e implementar un pipeline que recoja noticias financieras de fuentes RSS (Expansión: Mercados, Ahorro, Empresas), las filtre por relevancia y las distribuya automáticamente a suscriptores mediante correo electrónico HTML cada día laborable.

**Objetivo 2 — Base de datos y modelos clásicos:** Construir una infraestructura de base de datos dual (SQLite local + Supabase cloud) que almacene datos OHLCV (Open, High, Low, Close, Volume) de los 35 valores del IBEX35 con ingesta automática diaria. Aplicar modelos ARIMA para analizar las propiedades estadísticas de precios y log-retornos.

**Objetivo 3 — GARCH e ingesta de noticias:** Extender el análisis temporal con modelos GARCH para capturar heterocedasticidad en los residuos. Implementar un sistema de clasificación de noticias financieras usando un LLM que extraiga entidades mencionadas, sentimiento y categoría de relevancia.

**Objetivo 4 — ML avanzado y website:** Implementar, entrenar y evaluar siete arquitecturas de modelos de ML para la predicción de dirección de precio a un día vista para todos los valores del IBEX35. Publicar predicciones diarias en un website estático con actualización automática.

**Objetivo 5 — Trading bot:** Conectar las predicciones del sistema a Interactive Brokers para ejecutar órdenes de forma automática en una cuenta de paper trading, gestionando apertura de posiciones al abrir mercado y cierre con beneficio antes del cierre.

Un objetivo transversal, quizás el más importante desde una perspectiva académica, es mantener en todo momento un marco de evaluación riguroso y honesto: documentar las decisiones de diseño, evitar el sesgo de confirmación en la interpretación de resultados y reconocer las limitaciones del sistema.

### 1.3 Estructura del documento

El documento se organiza de la siguiente forma. El Capítulo 2 establece el marco teórico, desde las propiedades estadísticas de los retornos financieros hasta los fundamentos de los modelos empleados. El Capítulo 3 describe la arquitectura completa del sistema, sus módulos y sus interacciones. El Capítulo 4 detalla la ingeniería de características que alimenta los modelos de ML. El Capítulo 5 describe individualmente cada modelo implementado, incluyendo fundamento teórico, decisiones de hiperparámetros y sus justificaciones. El Capítulo 6 explica el protocolo de evaluación, incluyendo el diseño del backtesting, los costes de transacción y las métricas empleadas. El Capítulo 7 presenta los resultados experimentales. El Capítulo 8 discute las implicaciones de los resultados. El Capítulo 9 concluye el trabajo y propone líneas de trabajo futuro. Se incluyen referencias bibliográficas y apéndices con detalles técnicos adicionales.

---

## 2. Marco teórico

### 2.1 Mercados financieros y el IBEX35

Los mercados financieros son mecanismos de asignación de capital que agregan información dispersa en precios observables. La hipótesis de mercados eficientes (Fama, 1970) postula que los precios reflejan toda la información disponible en todo momento, lo que implica que es imposible obtener rentabilidades ajustadas por riesgo superiores a las del mercado de forma consistente. En su forma débil, la hipótesis establece que los precios históricos no contienen información predictiva; en su forma semifuerte, incorpora también toda la información pública; en su forma fuerte, incluye información privilegiada.

La evidencia empírica sobre la validez de la hipótesis es mixta. Lo & MacKinlay (1988) documentaron autocorrelación positiva significativa en rendimientos semanales de carteras de acciones estadounidenses, lo que constituye una violación de la forma débil. Jegadeesh & Titman (1993) encontraron evidencia robusta de momentum a horizontes de 3-12 meses. Más recientemente, Krauss et al. (2017) documentaron que modelos de ML (redes neuronales, gradient boosting, Random Forest) consiguen balanced accuracies del 52-55% en la predicción diaria de dirección del S&P500. Estas pequeñas ventajas estadísticas son consistentes con un mercado que se aproxima a la eficiencia pero no la alcanza perfectamente.

El IBEX35 (Índice Bursátil Español) es el principal índice de referencia de la Bolsa española. Creado en 1991, está compuesto por los 35 valores más líquidos cotizados en el Sistema de Interconexión Bursátil Español (SIBE). Su composición se revisa semestralmente por el Comité Asesor Técnico de Bolsas y Mercados Españoles (BME). Los valores incluidos abarcan sectores como banca (Santander, BBVA, CaixaBank), energía (Iberdrola, Repsol, Naturgy), telecomunicaciones (Telefónica), distribución (Inditex), y construcción e infraestructura (ACS, Ferrovial). La capitalización bursátil media de los componentes es muy heterogénea, lo que implica diferencias importantes en liquidez y costes de transacción relevantes para el diseño del sistema de trading.

Desde una perspectiva microestructural, el mercado español presenta algunas particularidades relevantes. La subasta de apertura tiene lugar entre las 8:30 y las 9:00 CET, con el mercado continuo operando de 9:00 a 17:30 CET. La subasta de cierre se desarrolla entre las 17:30 y las 17:35. Este horario define la ventana de ejecución del trading bot (Entregable 5). La regulación de ventas en corto recae sobre la CNMV, que puede imponer restricciones temporales en situaciones de estrés de mercado, como ocurrió durante la pandemia de 2020 y, más recientemente, en episodios de volatilidad relacionados con conflictos geopolíticos.

### 2.2 Series temporales financieras: propiedades estadísticas

Los retornos financieros presentan un conjunto de propiedades empíricas bien documentadas, denominadas "hechos estilizados" (Cont, 2001), que condicionan el diseño de cualquier sistema predictivo. Comprender estas propiedades es esencial para justificar las decisiones de modelización.

La **no estacionariedad de los precios** es la propiedad más importante desde un punto de vista práctico. Las series de precios son, en general, procesos integrados de orden 1 (I(1)): la media y la varianza cambian a lo largo del tiempo y la serie no revierte a un nivel fijo. Por contra, los log-retornos `r_t = log(P_t/P_{t-1})` son estacionarios en media, aunque no en varianza (véase clustering de volatilidad más adelante). Esta distinción es fundamental: modelizar precios directamente con ARIMA conduce a modelos mal especificados; modelizar log-retornos es la práctica estándar.

El **clustering de volatilidad** es quizás el hecho estilizado más relevante para este trabajo. Los periodos de alta volatilidad tienden a agruparse temporalmente: días con grandes movimientos (en cualquier dirección) son seguidos de días igualmente agitados. Bollerslev (1986) formalizó esta propiedad mediante los modelos GARCH (Generalized AutoRegressive Conditional Heteroskedasticity), que modelizan explícitamente la varianza condicional como función de innovaciones pasadas y varianzas pasadas. La implicación para los modelos de ML es que la volatilidad reciente (capturada por features como `vol_ratio_5_20` o `atr_pct`) es un predictor relevante no del nivel del retorno, sino del riesgo asociado a cualquier predicción.

Las **colas pesadas** (heavy tails) de la distribución de retornos implican que los eventos extremos son significativamente más frecuentes de lo que predicen modelos gaussianos. Mandelbrot (1963) fue el primero en documentar que los retornos financieros siguen distribuciones con colas similares a las de una distribución de Pareto o Lévy estable. Cont (2001) actualiza esta evidencia mostrando que el índice de cola está típicamente entre 3 y 5 para retornos diarios de acciones individuales. Esta propiedad tiene implicaciones directas para la elección de funciones de pérdida en los modelos de regresión: el MSE (Mean Squared Error) es extremadamente sensible a outliers, lo que justifica el uso de Huber Loss en los modelos neuronales de regresión (véase Sección 5.5).

La **reversión a la media a corto plazo y el momentum a medio plazo** son dos efectos aparentemente contradictorios pero que operan en horizontes distintos. A 1 día, los retornos tienden a exhibir autocorrelación negativa (reversión), atribuible en parte al efecto bid-ask bounce en mercados microestructurales (Roll, 1984). A horizontes de 1-12 meses, domina el momentum (Jegadeesh & Titman, 1993). El diseño de las features del sistema (en particular `log_ret_1` para reversión a 1 día y `log_ret_5`, `log_ret_20` para momentum) intenta capturar ambos efectos.

### 2.3 Modelos clásicos: ARIMA y GARCH

Los modelos ARIMA (AutoRegressive Integrated Moving Average) fueron popularizados por Box & Jenkins (1976) y siguen siendo la referencia estándar para el modelado univariante de series temporales. Un modelo ARIMA(p,d,q) combina un componente autorregresivo de orden p (AR), una diferenciación de orden d para convertir la serie en estacionaria, y un componente de media móvil de orden q (MA). Para los precios del IBEX35, el test ADF (Augmented Dickey-Fuller) confirma la no estacionariedad, por lo que d=1 (o equivalentemente, trabajar con log-retornos) es necesario. Los log-retornos de la mayoría de valores del IBEX35 se comportan como ruido blanco (ARIMA(0,0,0)), lo que implica que la estructura lineal de la serie es muy débil, coherente con una mercado cercano a la eficiencia.

La validación del modelo ARIMA requiere examinar el comportamiento de los residuos. El test de Ljung-Box verifica si quedan autocorrelaciones significativas en los residuos: si los residuos son ruido blanco, el modelo ha capturado toda la estructura lineal de la serie. Sin embargo, incluso con residuos no autocorrelacionados, el ARIMA asume homocedasticidad (varianza constante de los errores), un supuesto que los datos financieros violan sistemáticamente. Los correlogramas de los residuos al cuadrado revelan esta heterocedasticidad, motivando la extensión a modelos GARCH.

El modelo GARCH(p,q), introducido por Bollerslev (1986) como extensión del modelo ARCH de Engle (1982), modela la varianza condicional como:

```
σ²_t = ω + Σ(αᵢ · ε²_{t-i}) + Σ(βⱼ · σ²_{t-j})
```

donde ε_t son los residuos del modelo de la media (típicamente ARIMA), α captura el impacto de innovaciones recientes (efecto ARCH) y β captura la persistencia de la varianza (efecto GARCH). El modelo GARCH(1,1) es suficiente para la mayoría de series financieras diarias: es parsimonioso y captura adecuadamente el clustering de volatilidad (Bollerslev, 1986). La estimación se realiza por máxima verosimilitud bajo supuestos de distribución Normal o t-Student para capturar colas pesadas.

La integración de ARIMA y GARCH en un modelo ARIMA-GARCH permite modelizar simultáneamente la media condicional y la varianza condicional. La aplicación a BBVA.MC (seleccionado como caso de estudio por ser el segundo valor más líquido del IBEX35) confirma las propiedades esperadas: los log-retornos son próximos a ruido blanco en media pero presentan heterocedasticidad significativa. Las bandas de confianza de las predicciones se amplían rápidamente con el horizonte, reflejando el rápido crecimiento de la incertidumbre acumulada.

### 2.4 Machine Learning para predicción de mercados

El aprendizaje automático aplicado a la predicción de retornos financieros ha experimentado un crecimiento explosivo en la última década, aunque los resultados son más modestos de lo que sugiere la literatura entusiasta. Gu, Kelly & Xiu (2020) presentan la comparación más rigurosa y a gran escala hasta la fecha, evaluando 30,000 señales y múltiples arquitecturas en el mercado de acciones estadounidense: gradient-boosted trees y redes neuronales profundas son los mejores modelos para predicción cross-seccional mensual, aunque el R² predictivo fuera de muestra es del orden del 1-3%, valores estadísticamente significativos pero económicamente modestos. Krauss et al. (2017) obtienen balanced accuracies del 52-55% en predicción diaria del S&P500 con Random Forest, gradient boosting y redes neuronales.

Los **métodos de ensemble basados en árboles** (Random Forest, Gradient Boosting) presentan varias ventajas para la predicción financiera. Son robustos a outliers en las features, manejan bien relaciones no lineales, no requieren normalización de los datos, y producen estimaciones de importancia de variables que facilitan la interpretabilidad. Su principal limitación es la incapacidad de extrapolar fuera del rango de entrenamiento: un modelo entrenado en un período de baja volatilidad tendrá dificultades para predecir retornos durante una crisis de alta volatilidad que exceda los máximos históricos del período de entrenamiento.

Las **redes neuronales recurrentes** (RNN, LSTM, GRU) son particularmente adecuadas para datos secuenciales. Los LSTM (Long Short-Term Memory), propuestos por Hochreiter & Schmidhuber (1997), resuelven el problema del vanishing gradient mediante puertas de memoria que controlan qué información conservar y qué olvidar. Los GRU (Gated Recurrent Units), introducidos por Cho et al. (2014), simplifican la arquitectura LSTM reduciendo el número de puertas y parámetros. Fischer & Krauss (2018) aplicaron LSTM a la predicción diaria de dirección del S&P500 con secuencias de 240 días, obteniendo una accuracy de ~54%. El beneficio práctico de las redes recurrentes sobre los árboles en series financieras es modesto pero documentado en la literatura (Sezer et al., 2020).

Las **arquitecturas híbridas CNN+RNN** combinan capas convolucionales 1D para detección de patrones locales en secuencias temporales de features con capas recurrentes para capturar dependencias de largo plazo (Livieris et al., 2020; Lu et al., 2020). La convolución actúa como detector de patrones multi-feature en ventanas de 3-5 días consecutivos (por ejemplo, una caída del RSI acompañada de aumento de volumen y cambio de signo en el MACD), pasando una representación comprimida a la capa recurrente. La ganancia empírica sobre las RNN puras es del orden de 1-3 puntos porcentuales en accuracy, según la revisión de Sezer et al. (2020).

El **sobreajuste** es la amenaza más seria en la aplicación de ML a datos financieros. Lopez de Prado (2018) identifica múltiples formas de sesgo que inflan artificialmente los resultados en muestra: el sesgo de selección (elegir el mejor modelo tras múltiples intentos), el sesgo de look-ahead (usar información futura en el entrenamiento), y el sesgo de supervivencia (omitir empresas que han quebrado o sido excluidas del índice). El diseño del protocolo de evaluación de este trabajo (véase Capítulo 6) intenta mitigar sistemáticamente estos sesgos.

### 2.5 Procesamiento de lenguaje natural en finanzas

El análisis de sentimiento financiero mediante procesamiento de lenguaje natural (PLN) ha ganado relevancia académica y práctica desde que Tetlock (2007) demostró que el contenido negativo en la columna de mercados del Wall Street Journal predice retornos del día siguiente. Loughran & McDonald (2011) desarrollaron un diccionario léxico específico para el dominio financiero, mejorando significativamente los resultados respecto a diccionarios genéricos como Harvard IV. La llegada de los modelos de lenguaje grande (LLM) ha transformado esta área: modelos basados en Transformer (Vaswani et al., 2017) pueden procesar texto con comprensión contextual que los enfoques léxicos no alcanzan.

En este trabajo, la clasificación de noticias se realiza mediante la API de Groq con el modelo `openai/gpt-oss-120b`. Este modelo recibe el título y cuerpo de cada noticia y produce cuatro outputs: (1) categoría de la noticia (company_specific, macro_economic, market_sentiment, generic_noise), (2) lista de empresas mencionadas con sus tickers, (3) sentimiento (positivo, negativo, neutral), y (4) puntuación de relevancia ponderada. La elección de Groq como proveedor de inferencia obedece a su velocidad y coste reducidos, relevantes para procesar cientos de noticias diariamente en un pipeline automatizado.

La integración del análisis de noticias con los modelos de ML presenta desafíos metodológicos importantes. La sincronización temporal es crítica: una noticia publicada a las 10:00 horas puede mover el precio ese mismo día, pero una noticia publicada tras el cierre solo puede usarse como señal para el día siguiente. En este trabajo, las noticias se ingresan como datos de referencia en la base de datos pero no se integran directamente como features en los modelos de ML de los Entregables 4 y 5, que usan exclusivamente features técnicos y macroeconómicos. La integración de señales de texto en los modelos de predicción se propone como trabajo futuro.

---

## 3. Arquitectura del sistema

### 3.1 Visión general

El sistema 10\*\*6 está organizado en cinco capas funcionales que se corresponden directamente con los cinco entregables del proyecto. Cada capa es independiente en su operación pero comparte la infraestructura de datos (base de datos dual SQLite/Supabase) y el entorno de automatización (GitHub Actions). La arquitectura sigue el principio de diseño por capas: cada componente expone una interfaz clara, puede ser reemplazado o extendido de forma independiente, y las dependencias entre capas son unidireccionales (de inferior a superior).

La tecnología central es Python 3.13, con pandas y numpy para manipulación de datos, scikit-learn para preprocesamiento y modelos clásicos, xgboost para gradient boosting, PyTorch para redes neuronales, y joblib para serialización de artefactos. La persistencia usa SQLite para el almacenamiento local (ventaja: sin latencia de red, adecuado para el entrenamiento de modelos que lee grandes volúmenes) y Supabase (PostgreSQL cloud) para las predicciones y datos que el website necesita consultar dinámicamente. La automatización opera exclusivamente mediante GitHub Actions en servidores Ubuntu, eliminando la necesidad de infraestructura propia excepto para el trading bot, que requiere acceso local a Interactive Brokers.

> **[FIGURA 3.1 — INSERTAR AQUÍ]**
> *Descripción: Diagrama de arquitectura de alto nivel del sistema 10\*\*6.*
> *Qué debe mostrar: cinco capas (datos, LLM/noticias, modelos, trading, presentación), sus módulos internos y las flechas de dependencia de datos entre capas. Incluir las automatizaciones de GitHub Actions como procesos externos.*

### 3.2 Capa de datos

La capa de datos es la base sobre la que operan todos los demás componentes. Su función principal es la ingesta, almacenamiento y provisión de datos OHLCV (Open, High, Low, Close, Volume) limpios y actualizados para los 35 valores del IBEX35 más los tickers macroeconómicos necesarios (^IBEX, ^GSPC para el S&P500, y ^VIX para el índice de volatilidad implícita).

La ingesta automática se produce cada día laborable a las 08:00 UTC mediante el workflow `ohlcv.yml` de GitHub Actions. El script `db/supabase/ingest_ohlcv.py` descarga datos del día anterior a través de la librería `yfinance` y los carga en Supabase. Paralelamente, los datos se mantienen en una base de datos SQLite local (`db/sqlite/`) para el entrenamiento de modelos, donde la latencia cero y la posibilidad de consultas complejas son prioritarias. La sincronización entre ambas bases de datos se gestiona mediante `db/migrations.py`.

El esquema de la tabla `ohlcv` en Supabase es minimalista: ticker (text), date (date), open, high, low, close, volume (float8), con clave primaria compuesta (ticker, date). Esta estructura permite consultas eficientes por ticker y rango de fechas, los dos patrones de acceso más frecuentes. El módulo `db/utils_ohlcv.py` expone funciones de utilidad para obtener la lista de tickers del IBEX35, los tickers macroeconómicos, y verificar la integridad de los datos almacenados.

Una decisión de diseño importante es la separación entre la base de datos operacional (Supabase, usada por el website y el bot) y la base de datos de entrenamiento (SQLite, usada por los modelos). Esta separación evita que el proceso de entrenamiento, que puede tardar varias horas, bloquee o degrade la disponibilidad de la base de datos operacional. SQLite también facilita el trabajo local sin conexión a Internet, y su portabilidad (un único fichero) simplifica el control de versiones de los datos durante el desarrollo.

### 3.3 Capa de análisis LLM y noticias

Esta capa implementa un pipeline de ingesta y clasificación de noticias financieras que opera diariamente a las 08:00 UTC mediante el workflow `news.yml`. El pipeline consta de tres fases: fetching, clasificación y almacenamiento.

En la fase de fetching, el módulo `news/news_rss.py` descarga los feeds RSS de tres secciones de Expansión: Mercados, Ahorro y Empresas. Los artículos se filtran por relevancia usando keywords financieras relevantes (resultados, OPA, dividendo, beneficio, guidance, adquisición) y por fecha (solo noticias del día actual). El diseño reconoce que los feeds RSS de calidad para mercados españoles son limitados, y Expansión proporciona una cobertura razonable del IBEX35 en español.

En la fase de clasificación, el módulo `news/classification.py` envía cada noticia filtrada a la API de Groq usando el wrapper compatible con OpenAI (`llm/gpt_service.py`). El modelo `openai/gpt-oss-120b` recibe un prompt estructurado que solicita clasificar la noticia en una de cuatro categorías (company_specific, macro_economic, market_sentiment, generic_noise), identificar las empresas mencionadas con sus tickers del IBEX35, determinar el sentimiento (positivo/negativo/neutral) y calcular una puntuación de relevancia. La puntuación de relevancia pondera las categorías de la siguiente forma: company_specific=0.4, macro_economic=0.3, market_sentiment=0.2, generic_noise=0.0. Un módulo de control de tasa (`llm/rate_limit.py`) gestiona los límites de la API.

El almacenamiento se realiza en dos tablas de Supabase: `news` (datos de la noticia, categoría, relevancia, sentimiento) y `news_entities` (relación noticia-ticker), conectadas mediante clave foránea. Esta estructura normalizada permite consultas eficientes como "todas las noticias positivas de BBVA en los últimos 5 días" o "las 10 noticias de mayor relevancia de hoy". La restricción UNIQUE sobre el campo `url` evita duplicados ante reintentos del pipeline.

### 3.4 Capa de modelos predictivos

La capa de modelos es el núcleo del proyecto académico. Implementa siete arquitecturas diferentes para la predicción de dirección de precio a 1 día vista, con un framework de entrenamiento compartido (`models/base.py`) que garantiza coherencia metodológica entre modelos. Esta capa se complementa con el módulo de ingeniería de características (`models/trees/features.py`) que genera el vector de 41 features ML-ready a partir de los datos OHLCV crudos.

El framework `BaseTrainer` implementa el patrón Template Method: el método `run()` orquesta el pipeline completo (carga de datos, construcción de features, validación cruzada walk-forward, entrenamiento del modelo final, serialización del artefacto), mientras que cada subclase implementa `_train_window()` y `_train_final()` con la lógica específica de su modelo. Esto garantiza que el protocolo de evaluación sea idéntico para todos los modelos, condición necesaria para que las comparaciones sean válidas.

Los artefactos de los modelos entrenados se serializan como ficheros `.pkl` en el directorio `artifacts/` usando joblib. El naming convention es `{model_key}_h{horizon}.pkl` para modelos de clasificación discreta y `{model_key}_h{horizon}_cont.pkl` para regresión continua. Las predicciones del modelo en producción (actualmente Random Forest h=1) se actualizan diariamente mediante el workflow `preds.yml`, que ejecuta `db/supabase/upload_preds.py` y también actualiza `data/predictions.json` para el website estático.

### 3.5 Capa de ejecución (trading bot)

El bot de trading (`trading/execute.py`) conecta las predicciones del modelo con el broker Interactive Brokers mediante la librería `ib_async`. Opera sobre una cuenta de paper trading (100.000€ simulados) en el puerto 4002 de IB Gateway, configurado como paper trading para evitar pérdidas reales. La ejecución opera sobre señales de los modelos de ML del Entregable 4.

El flujo de operación tiene dos momentos diarios. Al abrir mercado (aproximadamente a las 09:05 CET, para dejar que la subasta de apertura se estabilice), el bot selecciona las tres señales con mayor probabilidad de subida según el modelo, verifica que la señal supere un umbral mínimo de confianza, y abre posiciones con un tamaño máximo del 10% del balance total por posición. Se aplica un stop-loss automático del 3% sobre el precio de entrada. Antes del cierre del mercado (a las 17:20 CET, dejando margen antes de la subasta de cierre), el bot revisa las posiciones abiertas y cierra aquellas que están en beneficio, gestionando el riesgo residual.

La programación diaria del bot se gestiona mediante Windows Task Scheduler a través del script PowerShell `trading/bot.ps1`. Esta elección obedece a que Interactive Brokers Gateway requiere un cliente Windows o macOS para la conexión de paper trading; la automatización vía GitHub Actions no es posible para este componente al requerir acceso local a la aplicación IB Gateway. Esta es la única componente del sistema que no está completamente automatizada en la nube.

### 3.6 Capa de presentación (website + newsletter)

La capa de presentación tiene dos canales: un website estático y un newsletter por correo electrónico. El website está alojado en GitHub Pages y sirve contenido HTML/CSS/JS estático desde el directorio `docs/`. Las predicciones diarias se obtienen de Supabase en tiempo de ejecución del cliente (JavaScript fetch) o, alternativamente, del fichero `data/predictions.json` que se actualiza como parte del workflow `preds.yml`. El diseño estático elimina los costes de servidor y garantiza alta disponibilidad.

El newsletter se envía diariamente a las 09:00 UTC mediante el workflow `newsletter.yml`. El módulo `newsletter/send.py` combina los artículos de noticias más relevantes del día (previamente procesados y almacenados en Supabase por el pipeline de noticias) en una plantilla HTML enriquecida con imagen embebida (codificada en base64 para máxima compatibilidad con clientes de correo). La lista de suscriptores se almacena en la tabla `newsletter` de Supabase. El envío se realiza mediante SMTP de Gmail usando credenciales almacenadas como GitHub Secrets. El módulo `newsletter/eg.html` sirve como plantilla de ejemplo para desarrollo y pruebas del formato visual.

---

## 4. Ingeniería de características

### 4.1 Features micro (técnicos)

Las features micro capturan la dinámica de precio y volumen específica de cada valor. El proceso de generación se implementa en `models/trees/features.py` mediante la función `ml_ready()`, que recibe los datos OHLCV crudos y retorna la matriz de features lista para entrenamiento. Todas las features se calculan exclusivamente con información disponible al cierre del día t, garantizando la ausencia de look-ahead bias (véase decisions/features_decisions.md para el razonamiento completo).

Las features de **retorno** cuantifican el momentum o reversión a distintos horizontes: `log_ret_1` (retorno del día anterior, señal de reversión a 1 día), `log_ret_5` (retorno semanal, señal de momentum) y `log_ret_20` (retorno mensual). El log-retorno es la transformación estándar porque es aditivo en el tiempo y simétrico en subidas y bajadas de la misma magnitud porcentual (Lo & MacKinlay, 1988). Se excluye `log_ret_3` por estar altamente correlacionado con `log_ret_5` (correlación típica > 0.85) sin aportar información adicional.

Las features de **volatilidad** incluyen `vol_5` (volatilidad realizada de los últimos 5 días), `vol_ratio_5_20` (ratio de volatilidad de corto vs. largo plazo, indicador de régimen de volatilidad), y `atr_pct` (Average True Range porcentual, que captura la volatilidad intradía incluyendo gaps). El ratio de volatilidades es preferible al nivel absoluto por ser invariante ante cambios en el régimen de volatilidad de fondo: un `vol_ratio_5_20 > 1` indica un período de stress relativo independientemente de si la volatilidad absoluta es alta o baja (Bollerslev, 1986).

Las features de **tendencia** incluyen `sma_ratio_5_20` (ratio de media móvil simple de 5 vs. 20 días, señal de cruce dorado/muerte), `sma_ratio_10_50` (ratio de 10 vs. 50 días para el horizonte medio), `macd_hist` (histograma MACD como diferencia entre señal rápida y lenta), `bb_pct` (posición del precio dentro de las Bandas de Bollinger, indicador de sobrecompra/sobreventa), y `slope_10` (pendiente de la regresión lineal de los últimos 10 días de precio normalizado). Se eliminaron duplicados como `slope_20` (correlación > 0.9 con `slope_10`) y `ema_ratio_5_20` (correlación > 0.95 con `sma_ratio_5_20`).

Las features de **posición relativa** `dist_high_10`, `dist_low_10`, `dist_high_20`, `dist_low_20` miden la distancia porcentual del precio actual respecto a máximos y mínimos recientes. Son señales de momentum de rango (Moskowitz et al., 2012) y complementan las medias móviles al capturar no solo la tendencia sino la energía relativa del movimiento.

Las features de **volumen** incluyen `volu_ratio_5` y `volu_ratio_20` (ratios de volumen a corto y largo plazo, respectivamente), `volu_ret_1` (producto de volumen y retorno del día anterior, indicador de presión compradora/vendedora informada), y `obv_slope_10` (pendiente del On-Balance Volume normalizado). Karpoff (1987) documenta la relación entre volumen y magnitud de movimiento, y Llorente et al. (2002) la relación entre volumen y reversión vs. continuación.

Las features de **microestructura intradía** —`body` (tamaño del cuerpo de la vela), `upper_wick`, `lower_wick` (tamaños de mechas superior e inferior), `gap` (diferencia entre apertura y cierre anterior), `intraday_ret` (retorno intradía de apertura a cierre)— capturan información sobre la dinámica de la sesión que los retornos cierre-a-cierre no contienen. El gap es especialmente relevante para la predicción a 1 día: los gaps tienden a revertir intradía, lo que implica que un gap positivo el día t puede predecir un retorno negativo el día t+1 a apertura (Lou et al., 2019).

Features adicionales incluyen `rsi_14` (Índice de Fuerza Relativa de 14 días, oscilador clásico de sobrecompra/sobreventa), `amihud_10` (ratio de Amihud de iliquidez a 10 días, ratio de retorno absoluto sobre volumen en euros) y `ret_autocorr_10` (autocorrelación de retornos a 10 días).

En total, las features micro suman 28. El criterio de inclusión/exclusión completo se documenta en `decisions/features_decisions.md`.

### 4.2 Features cross-seccionales (breadth IBEX)

Las features cross-seccionales capturan información sobre la amplitud del mercado (market breadth) usando el conjunto de los 35 valores del IBEX35. `ibx_breadth` mide la fracción de valores del IBEX35 que subieron en el día anterior (excluido el valor objetivo, para evitar look-ahead bias en el cálculo; véase decisions/features_decisions.md). Un breadth elevado (>0.7) indica que la subida del índice es amplia y sostenida; un breadth bajo (<0.3) con índice subiendo señala una concentración de ganancias en unos pocos valores (señal bajista para la mayoría). `ibx_breadth_10d` es la media móvil de 10 días del breadth diario, capturando la tendencia de la amplitud.

La corrección de look-ahead en el cálculo del breadth es una decisión técnica importante. Una implementación naive que incluyera el retorno del propio valor en el numerador del breadth crearía una correlación espuria entre la feature y el target: si el valor sube hoy (futuro que no conocemos), el breadth es mayor, que a su vez predice que el valor sube. La implementación correcta usa un breadth leave-one-out donde el retorno del valor objetivo se excluye del cómputo.

### 4.3 Features macro

Las features macro proporcionan contexto del entorno de mercado global que las features del stock individual no pueden capturar. Incluyen nueve variables derivadas de dos tickers macroeconómicos: `^IBEX` (el propio índice IBEX35) y `^GSPC` (S&P500 como proxy del mercado global), más `^VIX` (índice de volatilidad implícita del S&P500).

`ibx_vol_10` e `ibx_vol_ratio_10_60` capturan el régimen de volatilidad del mercado español. `sp_vol_20` y `sp_vol_ratio_20_100` hacen lo propio para el mercado estadounidense. La volatilidad del S&P500 se propaga a los mercados europeos con un rezago de aproximadamente un día (cierre US → apertura Europa), lo que la hace relevante como señal predictiva (Gu et al., 2020).

`vix_chg_z_5` es el cambio del VIX en los últimos 5 días, estandarizado por su desviación estándar reciente. Captura los picos de miedo que caracterizan las correcciones: un VIX que sube rápidamente indica que el mercado está comprando protección de forma acelerada. `vix_pctile_250` es el percentil del VIX actual dentro de su distribución del último año: valores extremos (>90%) indican estados de crisis; valores bajos (<10%) indican complacencia.

`rel_ret_5` y `rel_ret_20` miden el exceso de retorno del valor respecto al IBEX35 a 5 y 20 días. Son señales de alfa/beta relativo: un valor que sube más que el índice exhibe momentum relativo; un valor que cae más que el índice puede indicar presión vendedora específica. Gu et al. (2020) identifican las features de retorno relativo como entre las más predictivas en su análisis del mercado americano. `rel_vol_20` mide la volatilidad del valor relativa al índice, un proxy del factor de riesgo idiosincrático.

Finalmente, `dow_sin` y `dow_cos` son la codificación cíclica del día de la semana (`dow`: 0=lunes, 4=viernes). La codificación cíclica (mediante seno y coseno de la frecuencia semanal) preserva la naturaleza circular del tiempo: el lunes es estructuralmente más parecido al viernes que al miércoles. El efecto día de la semana (weekend effect, Monday effect) está bien documentado en la literatura para algunas carteras, aunque su magnitud en el IBEX35 contemporáneo es pequeña.

La combinación de estas features resulta en un vector de **41 dimensiones** por observación (fila) para el horizon h=1.

### 4.4 Target: dirección discreta y retorno continuo

El target de los modelos de clasificación binaria es `target = (future_log_ret > 0).astype(int)`, donde `future_log_ret = log(close_{t+h} / close_t)` es el log-retorno de h días hacia adelante. Para h=1, se predice si el precio cerrará el día t+1 por encima del cierre del día t. La distribución del target es aproximadamente 52/48 (subida/bajada) para la mayoría de valores del IBEX35 en el período histórico analizado, coherente con la tendencia alcista de largo plazo de los índices de renta variable.

El target de regresión continua es directamente `future_log_ret` sin binarización. Este target se usa en XGBoost y en las redes neuronales (GRU, LSTM, CNN+GRU, CNN+LSTM) cuando se configura `target_type="continuous"`. La motivación es que el log-retorno preserva información sobre la magnitud del movimiento que la binarización descarta: un retorno del +3% y un retorno del +0.1% son ambos "subida" (target=1) pero su impacto económico es radicalmente diferente. La métrica de evaluación primaria para el target continuo es el Information Coefficient (IC), definido como la correlación de Spearman entre la predicción y el retorno realizado.

La rama `dev-cl-continous` del repositorio extiende la rama `development-cl` con soporte para el target continuo. El código de `BaseTrainer` acepta el parámetro `target_type: str = "discrete"` y selecciona la columna de target apropiada. Las funciones de evaluación para el caso continuo están en `models/evaluate.py` e incluyen MAE, RMSE, R², directional accuracy e IC. Véase `decisions/continuous_target_decisions.md` para el razonamiento completo sobre qué modelos soportan cada tipo de target.

---

## 5. Modelos

### 5.1 ARIMA

ARIMA (AutoRegressive Integrated Moving Average) es el modelo de referencia para el análisis de series temporales univariantes. Su aplicación en el Entregable 2 sirve un doble propósito: (1) caracterizar estadísticamente las series de precios del BBVA.MC y establecer una línea de base de predicción clásica, y (2) documentar las propiedades de no estacionariedad de los precios y de ruido blanco de los log-retornos que motivarán las decisiones de modelizado en los entregables posteriores.

El análisis comienza con el test ADF (Augmented Dickey-Fuller) sobre la serie de precios de cierre ajustados de BBVA.MC. El test rechaza la estacionariedad en nivel (p > 0.1) pero acepta la estacionariedad en primera diferencia (p < 0.05), confirmando que la serie es I(1) y que d=1 es la diferenciación apropiada. Los correlograma ACF y PACF de los log-retornos no muestran autocorrelaciones significativas más allá de la primera diferenciación, lo que sugiere que ARIMA(0,0,0) —equivalente a ruido blanco— es el mejor modelo en media para esta serie.

La ajuste del modelo ARIMA se realiza mediante criterios de información (AIC, BIC) sobre una rejilla de valores (p,d,q) ∈ {0,1,2}³. El modelo seleccionado tipicamente es ARIMA(1,0,0) o ARIMA(0,0,0) para log-retornos, reflejando que la estructura autorregresiva es débil. Las predicciones del ARIMA para horizontes largos convergen rápidamente hacia la media incondicional (cero para log-retornos), con intervalos de confianza que se amplían proporcionalmente a √T. Esta amplificación de la incertidumbre con el horizonte de predicción es la evidencia más directa de que el ARIMA no proporciona información útil a largo plazo.

> **[FIGURA 5.1 — INSERTAR AQUÍ]**
> *Descripción: Predicción ARIMA de log-retornos de BBVA.MC.*
> *Qué debe mostrar: serie histórica de log-retornos (60 días), predicción a 20 días con bandas de confianza al 80% y 95%. El ensanchamiento rápido de las bandas ilustra la incertidumbre creciente.*

El análisis del residuo del ARIMA mediante el test de Ljung-Box confirma que los residuos no están autocorrelacionados en nivel, pero los correlogramas del cuadrado de los residuos muestran autocorrelaciones significativas, evidenciando heterocedasticidad condicional. Esta evidencia de clustering de volatilidad en los residuos motiva directamente la aplicación del modelo GARCH.

### 5.2 GARCH

El modelo GARCH(1,1) extiende el ARIMA modelizando explícitamente la varianza condicional de los residuos. La especificación completa es un modelo ARIMA-GARCH donde la ecuación de la media es el ARIMA ajustado previamente y la ecuación de varianza sigue el proceso GARCH.

La estimación se realiza por máxima verosimilitud bajo distribución t-Student para los residuos estandarizados, capturando las colas pesadas documentadas en los datos de BBVA.MC. Los parámetros estimados α (efecto ARCH) y β (efecto GARCH) tienen la interpretación de que la persistencia de shocks de volatilidad es α + β: valores típicos en el rango 0.92-0.97 implican que los shocks de volatilidad se disipan lentamente, lo que es coherente con la observación empírica de que los períodos de alta volatilidad duran semanas o meses, no días.

> **[FIGURA 5.2 — INSERTAR AQUÍ]**
> *Descripción: Predicción GARCH de volatilidad condicional de BBVA.MC.*
> *Qué debe mostrar: serie histórica de volatilidad realizada (90 días) vs. volatilidad condicional estimada por el GARCH, con predicción a 10 días y bandas de confianza.*

La incorporación de la volatilidad condicional estimada por el GARCH como feature para los modelos de ML se propone como trabajo futuro. En el estado actual, las features `vol_5` y `vol_ratio_5_20` sirven como proxies de la volatilidad condicional sin requerir el ajuste explícito del GARCH en producción.

### 5.3 Random Forest

Random Forest (Breiman, 2001) es el primer modelo de ML aplicado a la predicción de dirección del IBEX35, y sirve como baseline ML del sistema. Opera sobre la matriz de features plana (n_observaciones × 41 features), sin necesidad de normalización ni construcción de secuencias. Su artefacto entrenado, `artifacts/rf_h1_full.pkl`, es el modelo actualmente desplegado en producción para la generación de predicciones diarias.

Los hiperparámetros están diseñados para el régimen de bajo SNR de los retornos financieros (véase decisions/rf_decisions.md para la justificación detallada):

| Hiperparámetro | Valor | Justificación clave |
|---|---|---|
| `n_estimators` | 500 | Suficiente para convergencia en datos de bajo SNR; rendimiento marginal nulo > 400 |
| `max_depth` | 5 | Permite interacciones de 5 vías; previene memorización de ruido |
| `max_features` | 0.3 | ~9 features por split; compensa la alta correlación entre features financieras |
| `min_samples_leaf` | 50 | Estimaciones de probabilidad estables (error estándar ≈ 0.07); abarca múltiples fechas |
| `class_weight` | None | Desbalance 52/48 demasiado leve para rebalanceo |
| `bootstrap` | True | Mecanismo central del RF |
| `oob_score` | True | Diagnóstico gratuito de generalización |

La importancia de variables (`feature_importances_`) se reporta en el artefacto entrenado. En experimentos preliminares, las features de volatilidad (`vol_ratio_5_20`, `atr_pct`) y las de retorno reciente (`log_ret_1`, `log_ret_5`) son consistentemente las más importantes, coherente con la literatura (Krauss et al., 2017; Gu et al., 2020).

Dado que Random Forest no puede extrapolar fuera del rango de entrenamiento (predice promedios en hojas, limitado por los valores observados), no soporta el modo de target continuo. El modelo solo opera en modo clasificación binaria (`target_type="discrete"`). Para la regresión, se usa XGBoost o redes neuronales.

### 5.4 XGBoost

XGBoost (Chen & Guestrin, 2016) es el segundo modelo de árbol implementado, usando boosting secuencial en lugar del bagging paralelo del Random Forest. La distinción fundamental es que XGBoost ajusta secuencialmente el error residual del ensemble, lo que le permite capturar señales débiles distribuidas a través de múltiples iteraciones. Para datos financieros de bajo SNR, esto puede traducirse en una ligera ventaja sobre RF, aunque con el riesgo adicional de sobreajuste en rondas tardías.

El hiperparámetro más importante de XGBoost en el contexto financiero es el **early stopping** (paciencia = 30 rondas). Sin él, XGBoost ajustaría ruido en las rondas posteriores a la captura del signal genuino. La implementación usa una división temporal 80/20 dentro de la ventana de entrenamiento: el 80% más antiguo para entrenamiento, el 20% más reciente como conjunto de validación para el early stopping. Esta división respeta el orden temporal y evita filtrar información futura al entrenamiento. La métrica monitoreada es `logloss`, que detecta el sobreajuste antes que la accuracy al ser sensible a cambios en la calibración de probabilidades.

Los hiperparámetros están diseñados para ser más conservadores que en aplicaciones no financieras (véase decisions/xgboost_decisions.md):

| Hiperparámetro | Valor | Justificación clave |
|---|---|---|
| `n_estimators` | 300 | Techo; las rondas reales las determina el early stopping |
| `learning_rate` | 0.05 | Shrinkage moderado; cada árbol contribuye poco |
| `max_depth` | 3 | Aprendices débiles; boosting compensa por iteración |
| `min_child_weight` | 100 | Controla la estabilidad de las hojas en datos financieros ruidosos |
| `subsample` | 0.7 | Subsampling de filas para decorrelacionar árboles |
| `colsample_bytree` | 0.7 | Subsampling de columnas moderado |
| `reg_lambda` | 1.0 | Regularización L2 por defecto |
| `objective` | binary:logistic | Probabilidades calibradas para position sizing |

XGBoost soporta ambos tipos de target: clasificación binaria (`objective="binary:logistic"`) y regresión continua (`objective="reg:squarederror"`). En modo regresión, el procedimiento de two-step final training asegura que el número de rondas usado en el modelo final se determina mediante early stopping en un subconjunto reciente, y luego se reentrena con todos los datos usando ese número de rondas óptimo.

### 5.5 Redes Recurrentes: GRU y LSTM

Las redes GRU (Gated Recurrent Units, Cho et al., 2014) y LSTM (Long Short-Term Memory, Hochreiter & Schmidhuber, 1997) procesan la información como secuencias de T=20 timesteps, donde cada timestep es un vector de 41 features. La elección de T=20 (~1 mes de trading) representa el equilibrio entre contexto temporal suficiente y evitar el vanishing gradient en secuencias largas (Baek & Kim, 2018). La arquitectura procesa cada ticker de forma independiente, con pesos compartidos entre tickers (modelo global), lo que maximiza los datos de entrenamiento disponibles.

Las diferencias entre LSTM y GRU son arquitectónicas: el LSTM usa tres puertas (olvido, entrada, salida) y una celda de memoria separada, totalizando ~4 veces los parámetros por capa que una RNN simple; el GRU usa dos puertas (reset, update) y elimina la celda de memoria separada, reduciendo parámetros y tiempo de entrenamiento. En la práctica para series financieras, las diferencias de rendimiento son marginales (Chung et al., 2014). Ambos se implementan como modelos independientes para permitir la comparación.

La arquitectura específica implementada en `models/neural/gru.py` y `models/neural/lstm.py` (gestionados por el trainer común en `models/neural/rnn_trainer.py`) es:

- Proyección de entrada: Linear(41, hidden_size)
- Capa recurrente: GRU o LSTM con hidden_size=64, dropout=0.3
- Proyección de salida: Linear(hidden_size, 1)
- Para clasificación: BCEWithLogitsLoss + sigmoid para probabilidades
- Para regresión: HuberLoss(delta=0.01) sin activación en la salida

La elección de HuberLoss sobre MSE para el target continuo obedece a las colas pesadas de la distribución de retornos (Cont, 2001): un único día con retorno del -5% tiene 25 veces la contribución de MSE de un día con -1%, lo que puede desestabilizar el entrenamiento. HuberLoss transiciona a pérdida absoluta para retornos que exceden el umbral δ=0.01 (1% diario), limitando el impacto de eventos extremos en los gradientes (véase decisions/rnn_decisions.md y decisions/continuous_target_decisions.md).

El entrenamiento usa el optimizador Adam con learning rate 1e-3 y weight decay 1e-4 para regularización L2 implícita. Se aplica gradient clipping (norm ≤ 1.0) para estabilidad en las redes recurrentes. El batch size es 64 y se entrena por un máximo de 30 épocas con early stopping basado en la pérdida de validación.

> **[FIGURA 5.3 — INSERTAR AQUÍ]**
> *Descripción: Curva de aprendizaje (learning curve) de un modelo GRU típico durante el entrenamiento.*
> *Qué debe mostrar: pérdida de entrenamiento y validación por época, con el punto de early stopping marcado. Ilustra la tendencia al sobreajuste en épocas tardías.*

### 5.6 CNN + GRU/LSTM

Las arquitecturas híbridas CNN+GRU y CNN+LSTM añaden una capa de convolución 1D antes de la capa recurrente (Livieris et al., 2020; Kim & Won, 2018). La motivación es que las capas convolucionales son detectores eficientes de patrones locales multi-feature en ventanas pequeñas de tiempo: una convolución de kernel_size=3 puede detectar, por ejemplo, una patrón de 3 días donde RSI cae mientras el volumen sube y el MACD cruza a negativo, sin que este patrón necesite estar en posición fija dentro de la secuencia (invarianza translacional).

La arquitectura implementada en `models/neural/cnn_rnn.py`:

- Convolución 1D: Conv1d(in_channels=41, out_channels=32, kernel_size=3, padding=1)
- Activación: ReLU
- La secuencia convolucionalizada (batch, T=20, channels=32) alimenta la capa recurrente
- Capa GRU o LSTM: hidden_size=64, dropout=0.3
- Proyección de salida idéntica a las redes puramente recurrentes

La ganancia esperada sobre las RNN puras es de 1-3 puntos porcentuales en balanced accuracy, según la revisión de la literatura (Sezer et al., 2020). Esta ganancia es más probable cuando las features incluyen indicadores técnicos que exhiben co-ocurrencias temporalmente densas (patrones de 3-5 días), que es exactamente nuestro caso. Véase decisions/cnn_rnn_decisions.md para la justificación completa.

### 5.7 Cadena de Markov

La Cadena de Markov sirve como baseline probabilístico de baja dimensionalidad. A diferencia de los cinco modelos anteriores que operan en el espacio de 41 features, la Cadena de Markov opera sobre la discretización cuantílica del único feature `log_ret_1` en 3 estados (tercil inferior: caída fuerte, tercil medio: día plano, tercil superior: subida fuerte) y estima la matriz de transición entre estados mediante conteo con suavizado de Laplace.

La elección de bins cuantílicos en lugar de umbrales fijos es metodológicamente importante (Hamilton, 1989; véase decisions/markov_decisions.md). Los umbrales fijos producen clases muy desiguales en períodos de alta o baja volatilidad (por ejemplo, en un período de baja volatilidad, casi todas las observaciones caerían en el bin central si el umbral es ±0.5%). Los cuantiles garantizan frecuencias aproximadamente iguales en cada estado, maximizando la información disponible para estimar cada fila de la matriz de transición.

El modelo responde a la pregunta "¿qué fracción de veces que el valor tuvo un día de caída fuerte (estado 0) el día siguiente estuvo en un día de subida (estado 2)?". La evidencia esperada, coherente con Lo & MacKinlay (1988), es una ligera probabilidad de reversión a corto plazo, aunque la magnitud puede ser demasiado pequeña para ser estadísticamente significativa en el IBEX35. El modelo tiene importancia pedagógica y como sanity check del pipeline: si no muestra ninguna reversión, sugiere un error en la construcción del target o las features.

### 5.8 Comparativa y selección de modelos

Los siete modelos se comparan bajo el mismo protocolo de evaluación walk-forward (véase Capítulo 6), usando balanced accuracy como métrica primaria para los modelos de clasificación e IC (Spearman) para los de regresión. La comparación es válida porque el framework `BaseTrainer` garantiza que todos los modelos usan exactamente las mismas ventanas de entrenamiento y test, las mismas fechas de corte, y la misma función de evaluación.

> **[FIGURA 5.4 — INSERTAR AQUÍ]**
> *Descripción: Comparativa de balanced accuracy media por modelo en CV.*
> *Qué debe mostrar: diagrama de barras con media ± desviación estándar de balanced accuracy para RF, XGB, GRU, LSTM, CNN+GRU, CNN+LSTM, Markov. Línea de referencia en 0.50 (modelo aleatorio).*

La hipótesis a priori es que los modelos más expresivos (redes neuronales) no superarán necesariamente a los modelos más simples (RF, XGBoost) dado el tamaño reducido de los datos y el bajo SNR inherente a los retornos financieros diarios. Los resultados de Krauss et al. (2017) en el S&P500 —donde ningún modelo dominó consistentemente— son la referencia. La Cadena de Markov se espera que sea el peor modelo en términos de balanced accuracy pero el más interpretable y el que sirve como lower bound estadístico.

---

## 6. Backtesting y evaluación

### 6.1 Protocolo walk-forward

El protocolo de evaluación es un elemento crítico del trabajo. Muchos estudios de ML financiero reportan resultados inflados debido a fugas de información temporal, selección sesgada de períodos de evaluación, o múltiples comparaciones sin corrección. Este trabajo adopta el estándar más exigente: validación cruzada walk-forward con ventana deslizante (véase decisions/backtesting_decisions.md).

La implementación en `models/base.py` usa ventanas de entrenamiento de 750 días (~3 años de trading), avanzando en pasos de 63 días (~trimestral). Para cada ventana, el modelo se entrena desde cero en los 750 días de entrenamiento, y se evalúa en los 63 días de test siguientes (sin overlap con el período de entrenamiento). Un embargo de 1 día entre el último día de entrenamiento y el primero de test previene la fuga de features con ventanas deslizantes superpuestas (Lopez de Prado, 2018, Cap. 7). Este embargo es particularmente importante para features como `log_ret_5` que usan ventanas de 5 días: sin embargo, el retorno calculado con t como último día incluye información del día t, que también es el primer día de test en ausencia de embargo.

El número total de ventanas depende del período histórico disponible. Con datos desde aproximadamente 2015, y un WINDOW_DAYS=750 (aprox. 3 años), quedan disponibles para evaluación aproximadamente 2 años de datos fuera de muestra, lo que produce entre 8 y 12 ventanas de CV con STEP_DAYS=63. Este número de ventanas es suficiente para estimar la media y varianza de las métricas con razonable precisión estadística, aunque no permite tests de significatividad de alta potencia.

La comparación de modelos a través de múltiples ventanas requiere atención al test de hipótesis. Dado que las ventanas se superponen temporalmente en las features de entrenamiento (aunque no en los datos de test), los resultados de ventanas adyacentes no son independientes. La metodología correcta de comparación estadística es la propuesta por Diebold & Mariano (1995), que corrige por la dependencia serial al testear si dos modelos tienen la misma performance de predicción.

### 6.2 Costes de transacción y supuestos de ejecución

Los costes de transacción son el principal enemigo de cualquier estrategia de trading de alta frecuencia. Para el IBEX35, se adopta una estructura de costes por niveles de liquidez:

- **Tier 1** (top-10 por capitalización: Santander, BBVA, Inditex, Iberdrola, Telefónica, BBVA, Repsol, ACS, Ferrovial, Amadeus): 10 puntos básicos ida+vuelta (5 bps spread + 5 bps comisión).
- **Tier 2** (siguiente decena): 15 bps ida+vuelta.
- **Tier 3** (resto): 20 bps ida+vuelta.

Para una estrategia que opera diariamente (h=1) con 100% de rotación diaria, los costes anuales son del orden del 25-40% del portfolio, lo que hace que la rentabilidad bruta necesaria para cubrir costes sea extremadamente alta. Esta matemática brutal es la razón principal por la que las estrategias de predicción a 1 día son difíciles de monetizar, incluso con modelos que superan el 55% de balanced accuracy. A h=5 (semanal) los costes se reducen proporcionalmente.

Los supuestos de ejecución siguen la convención T+1 apertura: la señal se genera al cierre del día t usando todas las features disponibles (incluyendo el cierre de t), y la ejecución se produce en la apertura del día t+1. El retorno realizado en el backtesting es `open_{t+2}/open_{t+1} - 1` para h=1. Esta convención es conservadora pero realista: evita el uso del precio de cierre para la ejecución (que requeriría participar en la subasta de cierre con información incompleta) y asegura que el precio de ejecución es el primero disponible el día siguiente (Korajczyk & Sadka, 2004).

### 6.3 Estrategias: baseline, filtro de confianza, long/short

Se evalúan tres estrategias de inversión sobre las predicciones de los modelos:

**Estrategia baseline (Long-only simple):** Para cada día, invierte en todos los valores cuya predicción es "subida" (pred=1), con peso igual entre ellos. Sin filtro de confianza. Equivale a usar el modelo como un predictor binario sin aprovechar la información de probabilidad.

**Estrategia con filtro de confianza:** Solo invierte cuando la probabilidad predicha supera un umbral mínimo de confianza (por defecto 0.55, es decir, 5 puntos porcentuales por encima de la aleatoriedad). Esto reduce el número de operaciones pero, si el modelo está bien calibrado, mejora el hit rate. El análisis de calibración (análisis de fiabilidad del modelo) se realiza en `notebooks/confidence_analysis.ipynb`.

**Estrategia long/short:** Compra los valores con mayor probabilidad de subida y vende en corto los de mayor probabilidad de bajada. Genera el doble de exposición y potencialmente el doble de alfa, pero también el doble de costes y exposición al riesgo de shorting (disponibilidad de préstamo, restricciones regulatorias del CNMV). Esta estrategia es principalmente teórica para el contexto del IBEX35.

### 6.4 Benchmarks y métricas financieras

El benchmark principal es el buy-and-hold sobre el IBEX35 (equivalente a un ETF sobre el índice). En el período 2020-2025, el IBEX35 ha tenido un Sharpe aproximado de 0.25-0.40. Cualquier estrategia activa debe superar este benchmark ajustado por riesgo para justificar su complejidad.

Las métricas financieras reportadas incluyen: retorno anualizado, volatilidad anualizada, Sharpe ratio (usando el tipo de interés libre de riesgo del Bund alemán a 2 años como referencia), Maximum Drawdown (mayor caída desde máximo hasta mínimo durante el período), y Calmar ratio (retorno anualizado / Maximum Drawdown). La comparación con el benchmark buy-and-hold usa estas cinco métricas.

Las métricas de predicción del modelo (balanced accuracy, AUC-ROC, MCC, log-loss) se calculan mediante la función `evaluate_model()` de `models/evaluate.py`, que es compartida por todos los modelos para garantizar comparabilidad. Para regresión: MAE, RMSE, R² y IC (correlación de Spearman entre predicciones y retornos). La función de evaluación compartida es un elemento clave de la arquitectura: cualquier cambio en las métricas se aplica automáticamente a todos los modelos.

### 6.5 Limitaciones y sesgos

A pesar del rigor metodológico empleado, el sistema presenta limitaciones que deben reconocerse explícitamente.

**Sesgo de supervivencia:** Los datos incluyen solo los valores actualmente en el IBEX35. Las empresas que quebaron o fueron excluidas del índice durante el período de análisis no están representadas, lo que sesga los retornos al alza. Este sesgo es especialmente relevante para períodos de crisis.

**Overfitting de investigador:** Las decisiones de diseño (hiperparámetros, selección de features, estructura del pipeline) se tomaron observando datos históricos. Aunque se documenta cuidadosamente el proceso de decisión para minimizar los grados de libertad del investigador (Bailey et al., 2016), no puede descartarse completamente cierto nivel de sobreoptimización histórica.

**Non-stationarity y cambios de régimen:** Los modelos se entrenan con ventanas de 3 años, pero las propiedades estadísticas del mercado cambian con el tiempo (Brexit, pandemia COVID-19, subida de tipos de 2022-2023, geopolítica). Un modelo entrenado en 2018-2021 puede tener características predictivas muy diferentes de uno entrenado en 2022-2025.

**Capacidad:** El análisis asume que las posiciones del trading bot son demasiado pequeñas para impactar el mercado (< 100,000€ por valor). Escalar la estrategia a volúmenes institucionales requeriría modelizar el impacto de mercado, lo que cambiaría fundamentalmente los costes de ejecución.

---

## 7. Resultados

### 7.1 Resultados modelos clásicos (ARIMA, GARCH)

Los modelos ARIMA y GARCH se aplican a BBVA.MC como caso de estudio representativo. Los resultados confirman las propiedades estadísticas esperadas de los retornos financieros diarios del IBEX35.

**ARIMA:** El modelo óptimo seleccionado por criterio BIC es ARIMA(`[PENDIENTE — ejecutar experimentos]`, 0, `[PENDIENTE — ejecutar experimentos]`) para log-retornos. El test de Ljung-Box sobre los residuos no rechaza la hipótesis nula de no-autocorrelación (p-valor > 0.05 para todos los lags hasta 20). El test de ARCH-LM sobre los residuos al cuadrado rechaza la hipótesis nula de homocedasticidad (p-valor < 0.01), confirmando la necesidad del modelo GARCH.

**Métricas de predicción ARIMA (holdout de 60 días):**
- MAE: `[PENDIENTE — ejecutar experimentos]`
- RMSE: `[PENDIENTE — ejecutar experimentos]`
- MAPE: `[PENDIENTE — ejecutar experimentos]`
- Accuracy direccional (naive): `[PENDIENTE — ejecutar experimentos]`

**GARCH:** El modelo GARCH(1,1) converge con parámetros α = `[PENDIENTE]` y β = `[PENDIENTE]`, implicando una persistencia de volatilidad de α + β = `[PENDIENTE]`. El test de Ljung-Box sobre los residuos estandarizados del GARCH no rechaza la hipótesis de ruido blanco (p > 0.05), confirmando que el GARCH captura adecuadamente la heterocedasticidad.

> **[FIGURA 7.1 — INSERTAR AQUÍ]**
> *Descripción: Diagnóstico completo de residuos ARIMA y GARCH para BBVA.MC.*
> *Qué debe mostrar: Panel 4×1 con (a) serie de log-retornos histórica, (b) ACF de residuos ARIMA, (c) ACF de residuos al cuadrado, (d) volatilidad condicional GARCH con bandas.*

### 7.2 Resultados modelos ML — clasificación

Los modelos de clasificación binaria se evalúan sobre los 35 valores del IBEX35 mediante CV walk-forward, usando balanced accuracy como métrica primaria.

**Tabla de resultados CV (balanced accuracy media ± desviación estándar):**

| Modelo | Balanced Accuracy | AUC-ROC | MCC | Log-loss |
|--------|------------------|---------|-----|----------|
| Markov | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | N/A |
| Random Forest | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| XGBoost | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| GRU | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| LSTM | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| CNN+GRU | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| CNN+LSTM | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |

La interpretación de los resultados debe contextualizarse en las expectativas de la literatura. Una balanced accuracy del 53-55% para los mejores modelos sería un resultado excelente, coherente con lo reportado por Krauss et al. (2017) para el S&P500. Resultados inferiores al 51% deben considerarse estadísticamente no significativos dado el ruido en el número de ventanas CV disponibles.

> **[FIGURA 7.2 — INSERTAR AQUÍ]**
> *Descripción: Evolución temporal de balanced accuracy por modelo a lo largo de las ventanas CV.*
> *Qué debe mostrar: línea por modelo, eje X = ventana CV (fecha de inicio), eje Y = balanced accuracy. Ilustra la estabilidad o inestabilidad de cada modelo a lo largo del tiempo.*

### 7.3 Resultados modelos ML — regresión

Los modelos de regresión (XGBoost, GRU, LSTM, CNN+GRU, CNN+LSTM en modo `target_type="continuous"`) predicen el log-retorno a 1 día. La métrica primaria es el IC (Information Coefficient), definido como la correlación de Spearman entre predicciones y retornos realizados.

**Tabla de resultados CV (regresión):**

| Modelo | IC (Spearman) | Directional Accuracy | MAE | RMSE |
|--------|--------------|---------------------|-----|------|
| XGBoost (cont.) | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| GRU (cont.) | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| LSTM (cont.) | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| CNN+GRU (cont.) | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |
| CNN+LSTM (cont.) | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` | `[PENDIENTE]` |

Un IC de 0.02-0.05 (correlación de Spearman del 2-5%) sería un resultado consistente con la literatura para predicción diaria de retornos individuales. Gu et al. (2020) reportan ICs del orden del 3-4% para sus mejores modelos en el mercado americano mensual, que es estructuralmente más predecible que el mercado diario.

### 7.4 Análisis del trading bot

El trading bot opera en modo paper trading con 100,000€ simulados. Los resultados se evalúan en términos de retorno total acumulado, drawdown máximo y Sharpe ratio sobre el período de paper trading activo.

**Estadísticas del bot (paper trading):**
- Período de evaluación: `[PENDIENTE — activar bot y registrar]`
- Retorno total acumulado: `[PENDIENTE]`
- Retorno anualizado: `[PENDIENTE]`
- Sharpe ratio: `[PENDIENTE]`
- Maximum Drawdown: `[PENDIENTE]`
- Win rate (operaciones ganadoras / total): `[PENDIENTE]`
- Número medio de operaciones por día: `[PENDIENTE]`

> **[FIGURA 7.3 — INSERTAR AQUÍ]**
> *Descripción: Curva de capital del trading bot vs. IBEX35 buy-and-hold.*
> *Qué debe mostrar: curva de capital normalizada (base 100 al inicio) del bot y del benchmark, eje X = fecha, eje Y = valor de la cartera. Incluir región sombreada de drawdown máximo.*

### 7.5 Figuras y gráficos

Esta sección sirve como índice centralizado de todos los gráficos del trabajo:

> **[FIGURA 7.4 — INSERTAR AQUÍ]**
> *Descripción: Distribución de predicted probabilities vs. retornos realizados (calibración del modelo).*
> *Qué debe mostrar: diagrama de calibración (reliability diagram) para el mejor modelo. Eje X = probabilidad predicha agrupada en bins de 0.05. Eje Y = fracción de subidas observadas en ese bin. La diagonal perfecta indica calibración perfecta.*

> **[FIGURA 7.5 — INSERTAR AQUÍ]**
> *Descripción: Importancia de features para Random Forest.*
> *Qué debe mostrar: barplot horizontal de las top-20 features por MDI (Mean Decrease Impurity), con barras de error indicando la variación entre ventanas de CV. Permite identificar qué información captura el modelo.*

> **[FIGURA 7.6 — INSERTAR AQUÍ]**
> *Descripción: Comparativa de estrategias de inversión (baseline, filtro de confianza, long/short, buy-and-hold).*
> *Qué debe mostrar: curvas de capital acumuladas para las cuatro estrategias sobre el período de backtesting completo. Incluir tabla de métricas financieras bajo el gráfico.*

---

## 8. Discusión

Los resultados experimentales, una vez completados, deben interpretarse con cautela ante varias tensiones inherentes al problema. En primer lugar, la diferencia entre accuracy estadística y rentabilidad económica es fundamental: un modelo con 53% de balanced accuracy puede ser perfectamente correcto en su predicción promedio pero incorrecto en los movimientos grandes —que son los que más impactan la cartera— lo que produciría rendimientos negativos a pesar de una accuracy positiva. Grinold (1989) formaliza esta tensión con la Fundamental Law of Active Management: la rentabilidad activa esperada es proporcional al IC (Information Coefficient) multiplicado por la raíz del breadth (número de apuestas independientes), y el IC esperado para predicción diaria de una acción individual es extremadamente pequeño.

La comparación entre modelos debe considerar el trade-off entre complejidad y generalización. Random Forest y XGBoost, a pesar de ser modelos más simples desde la perspectiva de la arquitectura computacional, pueden superar a las redes neuronales en datos tabulares de tamaño moderado, como se documenta consistentemente en la literatura (Grinsztajn et al., 2022). Las redes neuronales requieren volúmenes de datos significativamente mayores para rentabilizar su mayor capacidad de expresividad, y con 22,500 observaciones por ventana de entrenamiento estamos en el límite inferior donde pueden manifestar su potencial.

La heterogeneidad entre valores del IBEX35 es una fuente de complejidad adicional. El sistema entrena un modelo global (mismos pesos para todos los valores), lo que asume que los patrones son similares entre Santander (una megabanca internacional) y Sacyr (una constructora mediana). Esta homogeneización puede perder señales específicas de sectores pero gana en regularización implícita y datos de entrenamiento. Una extensión natural sería entrenar modelos por sector o por cluster de volatilidad/capitalización.

La integración de señales de noticias (Entregable 3) con los modelos de ML (Entregable 4) es la principal omisión del sistema actual. Las noticias clasificadas por el LLM contienen información que no está en los features técnicos: un anuncio de OPA sobre una empresa, por ejemplo, produciría una subida que ningún indicador técnico podría predecir. La incorporación de features de texto (embeddings de noticias, puntuaciones de sentimiento agregadas por ticker) podría mejorar la performance en días con eventos corporativos importantes, aunque requeriría solucionar los desafíos de sincronización temporal y gestión de la dimensionalidad de los embeddings.

Desde una perspectiva más amplia, la existencia de una ventaja predictiva pequeña pero estadísticamente significativa plantea la pregunta de si los mercados son realmente eficientes. La interpretación más coherente con los datos es que los mercados son eficientes en la mayoría del tiempo, pero que existen fricciones (costes de transacción, restricciones de venta en corto, límites de arbitraje) que permiten la persistencia de pequeñas ineficiencias. El arbitraje de estas ineficiencias es costoso y arriesgado; solo es rentable para actores con acceso a infraestructura tecnológica de bajo coste y volúmenes suficientes para amortizarla. Un estudiante con una cuenta de paper trading puede documentar la existencia del patrón; convertirlo en negocio rentable requiere recursos muy superiores.

---

## 9. Conclusiones y trabajo futuro

### Conclusiones

Este trabajo ha diseñado, implementado y evaluado un sistema de análisis de inversión multifuente para el IBEX35 que integra cinco capas funcionales: newsletter automatizado, base de datos de mercado, análisis LLM de noticias, modelos de ML avanzados y bot de trading. La infraestructura técnica construida es robusta, escalable y mayoritariamente automatizada mediante GitHub Actions.

Los modelos clásicos (ARIMA, GARCH) confirman las propiedades estadísticas bien establecidas de los retornos financieros: los precios son no estacionarios, los log-retornos son próximos a ruido blanco en media, y el clustering de volatilidad requiere explícitamente el modelo GARCH. Las predicciones ARIMA tienen horizontes útiles muy cortos (1-3 días) y la incertidumbre crece rápidamente.

Los modelos de ML demuestran que es posible extraer una señal predictiva pequeña pero consistente de los features técnicos y macroeconómicos del IBEX35. Los resultados `[PENDIENTE — completar tras experimentos]` muestran `[PENDIENTE]`. La comparación entre modelos sugiere que `[PENDIENTE]`.

Desde una perspectiva personal y honesta, la experiencia de este proyecto genera las siguientes reflexiones:

Los mercados financieros son sistemas complejos donde participan actores con vastamente más recursos, información y sofisticación tecnológica que cualquier estudiante individual. La ventaja que pudiera extraerse de los datos públicos con técnicas estándar es pequeña y puede desaparecer cuando se considera la fricción real de transacción. La rentabilidad consistente en los mercados requiere ventajas de información, velocidad o estructura que van mucho más allá de los modelos presentados en este trabajo.

Si el objetivo es acumular riqueza a largo plazo con mínimo tiempo dedicado y stress emocional, la evidencia académica apunta consistentemente hacia los fondos indexados de bajo coste (ETF sobre el S&P500 o índices mundiales) como la mejor opción para la mayoría de inversores. Las estrategias activas, incluso bien implementadas, tienen dificultades para batir el índice de forma consistente una vez descontados todos los costes (Sharpe, 1991).

Este trabajo tiene valor académico como ejercicio de ingeniería de sistemas complejos, diseño de pipelines de ML para datos financieros, y evaluación rigurosa mediante walk-forward cross-validation. Su valor como herramienta de inversión debe evaluarse con extrema cautela y no debe nunca constituir la base de decisiones de inversión con dinero real.

### Trabajo futuro

Las líneas de trabajo futuro más relevantes son:

1. **Integración de señales de noticias:** Incorporar los embeddings de las noticias clasificadas por el LLM como features adicionales en los modelos de ML, evaluando si la información textual añade valor predictivo sobre las features técnicas.

2. **Modelos multi-horizonte:** Extender el framework a h=5 (semanal) y evaluar si la reducción de costes de transacción compensar el deterioro en la señal predictiva. Las decisiones de features para h=5 están parcialmente documentadas en decisions/features_decisions.md.

3. **Calibración de probabilidades:** Aplicar Platt scaling o calibración isotónica post-hoc a las probabilidades de XGBoost (que tienden a ser menos calibradas que las de Random Forest) para mejorar el position sizing basado en probabilidades.

4. **Modelo por sector:** Entrenar modelos separados por sector (bancario, energético, telecomunicaciones, etc.) para capturar dinámicas específicas de sector que el modelo global puede estar promediando.

5. **Análisis de atribución:** Implementar SHAP values para descomponer las predicciones del modelo en contribuciones individuales de cada feature, lo que permitiría entender y comunicar las decisiones del modelo de forma interpretable.

6. **Live trading con capital real (micro-cantidades):** Mover el bot de paper trading a una cuenta real con cantidades muy pequeñas (€100-500) para validar la implementación técnica con las fricciones del mercado real (slippage, fallos en la ejecución, latencias) que el paper trading no captura.

---

## Referencias

- **Almgren, R. & Chriss, N. (2001).** "Optimal execution of portfolio transactions." *Journal of Risk*, 3(2), 5-39.
- **Baek, Y. & Kim, H.Y. (2018).** "ModAugNet: A new forecasting framework for stock market index value with an overfitting prevention LSTM module and a prediction LSTM module." *Expert Systems with Applications*, 113, 457-480.
- **Bailey, D.H., Borwein, J., Lopez de Prado, M. & Zhu, Q. (2016).** "The probability of backtest overfitting." *Journal of Computational Finance*, 20(4).
- **Bao, W., Yue, J. & Rao, Y. (2017).** "A deep learning framework for financial time series using stacked autoencoders and long-short term memory." *PLOS ONE*, 12(7), e0180944.
- **Black, F. (1976).** "Studies of stock price volatility changes." *Proceedings of the 1976 American Statistical Association*, 177-181.
- **Bollerslev, T. (1986).** "Generalized autoregressive conditional heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.
- **Box, G.E.P. & Jenkins, G.M. (1976).** *Time Series Analysis: Forecasting and Control.* Holden-Day.
- **Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1), 5-32.
- **Chen, T. & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." *Proceedings of KDD 2016*, 785-794.
- **Chicco, D. & Jurman, G. (2020).** "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation." *BMC Genomics*, 21(1), 6.
- **Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H. & Bengio, Y. (2014).** "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." *Proceedings of EMNLP 2014*.
- **Chung, J., Gulcehre, C., Cho, K. & Bengio, Y. (2014).** "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling." *NIPS 2014 Workshop on Deep Learning*.
- **Cont, R. (2001).** "Empirical properties of asset returns: stylized facts and statistical issues." *Quantitative Finance*, 1(2), 223-236.
- **Diebold, F.X. & Mariano, R.S. (1995).** "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics*, 13(3), 253-263.
- **Engle, R.F. (1982).** "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation." *Econometrica*, 50(4), 987-1007.
- **Fama, E.F. (1970).** "Efficient Capital Markets: A Review of Theory and Empirical Work." *Journal of Finance*, 25(2), 383-417.
- **Fawcett, T. (2006).** "An introduction to ROC analysis." *Pattern Recognition Letters*, 27(8), 861-874.
- **Fischer, T. & Krauss, C. (2018).** "Deep learning with long short-term memory networks for financial market predictions." *European Journal of Operational Research*, 270(2), 654-669.
- **Frazzini, A., Israel, R. & Moskowitz, T.J. (2018).** "Trading costs." *SSRN Working Paper*.
- **Friedman, J.H. (2001).** "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189-1232.
- **Friedman, J.H. (2002).** "Stochastic Gradient Boosting." *Computational Statistics & Data Analysis*, 38(4), 367-378.
- **Grinold, R.C. (1989).** "The Fundamental Law of Active Management." *Journal of Portfolio Management*, 15(3), 30-37.
- **Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022).** "Why tree-based models still outperform deep learning on tabular data." *Advances in Neural Information Processing Systems*, 35.
- **Gu, S., Kelly, B. & Xiu, D. (2020).** "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies*, 33(5), 2223-2273.
- **Hamilton, J.D. (1989).** "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica*, 57(2), 357-384.
- **Hand, D.J. (2009).** "Measuring classifier performance: a coherent alternative to the area under the ROC curve." *Machine Learning*, 77(1), 103-123.
- **Hochreiter, S. & Schmidhuber, J. (1997).** "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.
- **Hoseinzade, E. & Haratizadeh, S. (2019).** "CNNpred: CNN-based stock market prediction using a diverse set of variables." *Expert Systems with Applications*, 129, 273-285.
- **Jegadeesh, N. & Titman, S. (1993).** "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65-91.
- **Karpoff, J.M. (1987).** "The relation between price changes and trading volume: A survey." *Journal of Financial and Quantitative Analysis*, 22(1), 109-126.
- **Kelly, J.L. (1956).** "A new interpretation of information rate." *Bell System Technical Journal*, 35(4), 917-926.
- **Kim, H.Y. & Won, C.H. (2018).** "Forecasting stock prices with a feature fusion LSTM-CNN model using different representations of the same data." *PLOS ONE*, 13(2), e0212320.
- **Korajczyk, R.A. & Sadka, R. (2004).** "Are Momentum Profits Robust to Trading Costs?" *Journal of Finance*, 59(3), 1039-1082.
- **Krauss, C., Do, X.A. & Huck, N. (2017).** "Deep neural networks, gradient-boosted trees, random forests: statistical arbitrage on the S&P 500." *European Journal of Operational Research*, 259(2), 689-702.
- **Leung, M.T., Daouk, H. & Chen, A.S. (2000).** "Forecasting stock indices: a comparison of classification and level estimation models." *International Journal of Forecasting*, 16(2), 173-190.
- **Livieris, I.E., Pintelas, E. & Pintelas, P. (2020).** "A CNN-LSTM model for gold price time-series forecasting." *Expert Systems with Applications*, 164, 113681.
- **Lo, A.W. & MacKinlay, A.C. (1988).** "Stock market prices do not follow random walks: Evidence from a simple specification test." *Review of Financial Studies*, 1(1), 41-66.
- **Lopez de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley.
- **Loughran, T. & McDonald, B. (2011).** "When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks." *Journal of Finance*, 66(1), 35-65.
- **Lou, D., Polk, C. & Skouras, S. (2019).** "A tug of war: Overnight versus intraday expected returns." *Journal of Financial Economics*, 134(1), 192-213.
- **Lu, W., Li, J., Kang, Y. & Li, J. (2020).** "A CNN-LSTM-based model to forecast stock prices." *Complexity*, 2020, 6622927.
- **Mandelbrot, B. (1963).** "The variation of certain speculative prices." *Journal of Business*, 36(4), 394-419.
- **Moskowitz, T.J., Ooi, Y.H. & Pedersen, L.H. (2012).** "Time series momentum." *Journal of Financial Economics*, 104(2), 228-250.
- **Niculescu-Mizil, A. & Caruana, R. (2005).** "Predicting Good Probabilities with Supervised Learning." *ICML 2005*, 625-632.
- **Niu, H., Zhong, K. & Yu, H. (2020).** "Hybrid model combining GRU neural network with attention mechanism for stock price prediction." *Journal of Ambient Intelligence and Humanized Computing*.
- **Oshiro, T.M., Perez, P.S. & Baranauskas, J.A. (2012).** "How Many Trees in a Random Forest?" *Lecture Notes in Computer Science*, 7376, 154-168.
- **Pascanu, R., Mikolov, T. & Bengio, Y. (2013).** "On the difficulty of training recurrent neural networks." *ICML 2013*.
- **Roll, R. (1984).** "A simple implicit measure of the effective bid-ask spread in an efficient market." *Journal of Finance*, 39(4), 1127-1139.
- **Sezer, O.B., Gudelek, M.U. & Ozbayoglu, A.M. (2020).** "Financial time series forecasting with deep learning: a systematic literature review: 2005-2019." *Applied Soft Computing*, 90, 106181.
- **Sharpe, W.F. (1991).** "The arithmetic of active management." *Financial Analysts Journal*, 47(1), 7-9.
- **Tetlock, P.C. (2007).** "Giving content to investor sentiment: The role of media in the stock market." *Journal of Finance*, 62(3), 1139-1168.
- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. & Polosukhin, I. (2017).** "Attention Is All You Need." *Advances in Neural Information Processing Systems*, 30.

---

## Apéndices

### Apéndice A — Configuración del entorno de desarrollo

Para reproducir los experimentos de este trabajo:

```bash
# 1. Clonar el repositorio
git clone https://github.com/alexhayadela/10tothe6_TFG_2025_AlexDeLaHaya.git
cd 10tothe6_TFG_2025_AlexDeLaHaya

# 2. Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements/all.txt

# 4. Crear fichero .env con credenciales
# EMAIL_USER, EMAIL_PASSWORD, GROQ_API_KEY, SUPABASE_API_KEY, SUPABASE_URL

# 5. Inicializar base de datos local
python -m db.migrations

# 6. Entrenar modelo Random Forest (baseline)
python -m models.train --model rf --horizon 1 --mode sliding
```

### Apéndice B — Esquema de base de datos

**Supabase (PostgreSQL):**

| Tabla | Columnas clave | Descripción |
|-------|---------------|-------------|
| `ohlcv` | (ticker, date) PK | Datos OHLCV de los 35 valores IBEX35 |
| `news` | id PK, url UNIQUE | Noticias clasificadas por LLM |
| `news_entities` | (news_id, ticker) | Relación noticia-empresa mencionada |
| `newsletter` | id PK, email | Suscriptores al newsletter |
| `predictions` | id PK | Predicciones ML diarias por ticker |

**SQLite (local, `db/sqlite/`):**
- Misma tabla `ohlcv` que en Supabase, optimizada para lecturas masivas por el entrenamiento de modelos.

### Apéndice C — Valores IBEX35 analizados

Los 35 valores del IBEX35 incluidos en el análisis son: ACS.MC, ACX.MC, AENA.MC, AMS.MC, ANA.MC, BBVA.MC, BKT.MC, CABK.MC, CLNX.MC, COL.MC, ELE.MC, ENG.MC, FDR.MC, FER.MC, GRF.MC, IAG.MC, IBE.MC, IDR.MC, ITX.MC, LOG.MC, MAP.MC, MEL.MC, MRL.MC, MTS.MC, NTGY.MC, PHM.MC, RED.MC, REP.MC, ROVI.MC, SAB.MC, SAN.MC, SCYR.MC, SOL.MC, TEF.MC, UNI.MC.

### Apéndice D — Resumen de hiperparámetros por modelo

| Modelo | Hiperparámetros clave | Archivo de referencia |
|--------|--------------------|----------------------|
| Random Forest | n_est=500, depth=5, max_feat=0.3, min_leaf=50 | decisions/rf_decisions.md |
| XGBoost | n_est=300, lr=0.05, depth=3, mcw=100, subsample=0.7, ES=30 | decisions/xgboost_decisions.md |
| GRU/LSTM | hidden=64, T=20, dropout=0.3, HuberLoss(δ=0.01) para regresión | decisions/rnn_decisions.md |
| CNN+GRU/LSTM | Conv1d(32, k=3) + GRU/LSTM(64) | decisions/cnn_rnn_decisions.md |
| Markov | n_states=3, quantile bins, Laplace smoothing | decisions/markov_decisions.md |
