# 10⁶ — Herramienta de análisis de inversión multifuente con ML para el IBEX 35

---

## Capítulo 1. Introducción

### 1.1 Contexto del proyecto

El proyecto se centra en el análisis de activos financieros pertenecientes al IBEX 35, principal índice bursátil español, compuesto por las empresas con mayor capitalización y liquidez del mercado nacional. A partir de datos históricos de cotización, se estudia la evolución temporal de los distintos activos con el objetivo de identificar patrones y tendencias relevantes.

El análisis se orienta principalmente a la predicción de movimientos de mercado, evaluando si el precio de un activo tenderá a subir o bajar en horizontes temporales determinados. Para ello se emplean técnicas de análisis de series temporales y modelos de aprendizaje automático capaces de extraer información a partir de datos financieros históricos.

### 1.2 Motivación

Este trabajo surge del interés por comprender el funcionamiento de los mercados financieros y analizar hasta qué punto es posible modelar su comportamiento mediante técnicas computacionales. Los mercados constituyen sistemas complejos, dinámicos y altamente influenciados por múltiples factores económicos, sociales y psicológicos, lo que los convierte en un ámbito especialmente atractivo desde el punto de vista analítico y tecnológico.

Además del interés académico, existe una motivación práctica relacionada con la necesidad de preservar y hacer crecer el capital en un contexto económico marcado por la inflación y la pérdida progresiva de poder adquisitivo. En este sentido, el estudio de estrategias de predicción financiera y modelos automatizados de inversión representa una oportunidad para explorar herramientas que puedan contribuir a una mejor toma de decisiones.

Otro de los motivos para desarrollar este proyecto es su potencial de continuidad más allá del propio Trabajo de Fin de Grado. Se trata de un área en constante evolución, con múltiples posibilidades de ampliación y mejora, tanto desde el punto de vista técnico como de investigación. Por ello, el proyecto se plantea no solo como un ejercicio académico, sino como una base sobre la que continuar desarrollando nuevas líneas de trabajo en el futuro.

### 1.3 Objetivos

El objetivo principal de este trabajo es estudiar si los modelos de aprendizaje automático pueden mejorar las capacidades predictivas de los enfoques tradicionales utilizados en los mercados financieros, o al menos competir con ellos en términos de rendimiento y robustez.

De manera más concreta, los objetivos del proyecto son los siguientes:

- Investigar el comportamiento de los mercados financieros mediante técnicas de análisis de datos y aprendizaje automático.
- Diseñar e implementar modelos capaces de realizar predicciones sobre activos financieros y comparar su rendimiento frente a métodos tradicionales y estrategias de referencia.
- Construir un sistema reproducible y automatizado que integre el flujo completo de datos, entrenamiento e inferencia con una intervención manual mínima.
- Desarrollar el proyecto utilizando exclusivamente herramientas, librerías y servicios gratuitos.

### 1.4 Enfoque del proyecto

Uno de los aspectos relevantes de este trabajo es que no se limita al desarrollo de pruebas aisladas en cuadernos interactivos o experimentos puntuales. El objetivo es construir una estructura completa y organizada, similar a un sistema real de análisis financiero automatizado.

Para ello, se plantea una arquitectura compuesta por diferentes etapas: obtención de datos, procesamiento, entrenamiento de modelos, generación de predicciones y evaluación de resultados. Todo el flujo se diseña de forma que pueda ejecutarse de manera automática y reproducible, facilitando tanto la experimentación como la posible ampliación futura del proyecto.

El proyecto se ha desarrollado con un enfoque de presupuesto cero, utilizando únicamente herramientas gratuitas y de código abierto. Esta decisión no solo reduce las barreras de acceso, sino que también favorece la reproducibilidad y la accesibilidad del trabajo para otros estudiantes o investigadores interesados en el área.

---

## Capítulo 2. Estado del arte

### 2.1 Herramientas y plataformas para la inversión en activos financieros

Actualmente existen numerosas plataformas destinadas al análisis de mercados financieros y al apoyo a la inversión. Entre las más populares destacan Yahoo Finance (Yahoo Finance, 2026) y TradingView (TradingView, 2026), que ofrecen distintos niveles de acceso según el tipo de suscripción.

En sus versiones gratuitas, estas plataformas permiten consultar datos históricos básicos, normalmente con frecuencia diaria o con retrasos de entre 15 y 30 minutos respecto al tiempo real. Además, proporcionan herramientas de visualización de gráficos, incorporación de indicadores técnicos y acceso a noticias financieras relacionadas con los mercados o activos específicos.

Las versiones de pago, que van desde los 10 hasta los 200 euros mensuales, amplían estas funcionalidades mediante acceso a un mayor histórico de datos, actualización en tiempo real, creación de alertas y herramientas avanzadas de análisis técnico. Sin embargo, estas plataformas presentan varias limitaciones desde el punto de vista de la investigación y el desarrollo de sistemas predictivos.

Por un lado, muchas de ellas no ofrecen una API pública completa o presentan restricciones importantes para la extracción automatizada de datos históricos y noticias financieras. Asimismo, la integración directa con brókers suele ser limitada o inexistente, dificultando la automatización completa de estrategias de inversión.

En un nivel más profesional destaca Bloomberg Terminal (Bloomberg L.P., 2026), ampliamente utilizado en entornos financieros institucionales. Esta solución proporciona datos de mercado de alta fidelidad en tiempo real, acceso a noticias financieras, herramientas de análisis cuantitativo, pruebas retrospectivas y soporte para trading algorítmico automatizado. No obstante, su coste es elevado, situándose alrededor de los 30.000 euros anuales, lo que limita su accesibilidad fuera del ámbito profesional o institucional.

A pesar de las amplias funcionalidades disponibles, estas plataformas están orientadas principalmente al análisis financiero tradicional y al trading algorítmico basado en reglas definidas manualmente. Es importante diferenciar este enfoque del uso de técnicas de aprendizaje automático. Las pruebas retrospectivas tradicionales consisten en evaluar estrategias predefinidas utilizando datos históricos —por ejemplo, mediante reglas fijas de compra y venta basadas en indicadores técnicos—. En cambio, los modelos de aprendizaje automático buscan aprender patrones complejos directamente a partir de los datos para realizar predicciones futuras, adaptándose dinámicamente a nuevas condiciones del mercado.

Las plataformas analizadas no proporcionan de forma integrada herramientas completas para el entrenamiento, validación y despliegue de modelos de aprendizaje automático. Incluso en entornos avanzados como Bloomberg Terminal, el desarrollo de modelos predictivos suele realizarse externamente mediante lenguajes y bibliotecas especializadas, como Python junto con frameworks de ciencia de datos y aprendizaje automático (Pedregosa et al., 2011). Esta situación justifica el interés en desarrollar soluciones específicas que integren obtención de datos, análisis financiero y modelos de aprendizaje automático dentro de un mismo entorno.

### 2.2 Modelos predictivos para datos financieros

La predicción de series temporales financieras constituye una de las principales aplicaciones del análisis cuantitativo en mercados financieros. Una serie temporal puede definirse como una secuencia de observaciones ordenadas cronológicamente, donde cada valor depende potencialmente de los anteriores. En finanzas, ejemplos habituales son el precio de cierre de una acción, el volumen negociado o los retornos de un activo.

El precio de los activos financieros suele considerarse una serie no estacionaria, ya que sus propiedades estadísticas cambian con el tiempo. Esta no estacionariedad dificulta el modelado y la predicción directa de los precios. Por este motivo, gran parte de la literatura trabaja con los retornos financieros en lugar de los precios absolutos, ya que los retornos presentan un comportamiento más cercano a la estacionariedad y facilitan el uso de modelos estadísticos y de aprendizaje automático.

La predicción de series temporales financieras presenta una dificultad adicional derivada de la hipótesis del mercado eficiente (Fama, 1970). Esta hipótesis establece que el precio de los activos refleja toda la información disponible en el mercado, por lo que resulta extremadamente difícil obtener ventajas predictivas consistentes utilizando únicamente información histórica. En consecuencia, los movimientos futuros del mercado presentan un elevado componente aleatorio y una baja relación señal-ruido.

En el ámbito de los modelos estadísticos clásicos, las cadenas de Markov modelan el sistema mediante un conjunto finito de estados y probabilidades de transición entre ellos, donde el estado futuro depende únicamente del estado actual. En finanzas, se han utilizado para modelar cambios de régimen de mercado y variaciones en volatilidad (Hamilton, 1989). Los modelos ARIMA (*Autoregressive Integrated Moving Average*), aunque ampliamente utilizados en predicción de series temporales (Box et al., 2015), no se emplean en este trabajo al no haberse mostrado competitivos frente a los enfoques basados en aprendizaje automático en el contexto de clasificación de dirección de mercado.

Los modelos basados en árboles de decisión han ganado popularidad en el ámbito financiero debido a su robustez y capacidad para modelar relaciones no lineales. Aunque no incorporan de forma explícita el orden temporal de los datos, pueden utilizar información histórica mediante ingeniería de características. Random Forest (Breiman, 2001) combina múltiples árboles generados a partir de subconjuntos aleatorios de datos y variables, reduciendo el sobreajuste mediante votación. XGBoost (Chen y Guestrin, 2016) construye modelos secuencialmente corrigiendo en cada iteración los errores de los estimadores anteriores. Ambos métodos han sido ampliamente empleados en predicción de retornos y clasificación de tendencias bursátiles (Krauss et al., 2017).

Las redes neuronales recurrentes (RNN) están diseñadas específicamente para procesar datos secuenciales, manteniendo información de estados anteriores mediante conexiones recurrentes. Las redes LSTM (*Long Short-Term Memory*) incorporan mecanismos de memoria y compuertas que permiten conservar información relevante durante periodos prolongados (Hochreiter y Schmidhuber, 1997). Las redes GRU (*Gated Recurrent Unit*) simplifican esta arquitectura reduciendo el número de parámetros y el coste computacional (Cho et al., 2014). Ambos modelos han sido utilizados ampliamente para la predicción de precios, retornos y volatilidad (Fischer y Krauss, 2018). No obstante, el uso de modelos de aprendizaje profundo en finanzas presenta dificultades relacionadas con la disponibilidad limitada de datos y el riesgo de sobreajuste a ruido.

En general, los modelos estadísticos clásicos destacan por su interpretabilidad y por requerir una menor cantidad de datos. Sin embargo, presentan limitaciones para capturar relaciones no lineales complejas. Por otro lado, los métodos de aprendizaje automático ofrecen una mayor capacidad de modelado, aunque requieren más datos, mayor capacidad computacional y presentan una interpretabilidad más reducida.

### 2.3 Evaluación de estrategias predictivas: fundamentos conceptuales

La evaluación de modelos predictivos en el contexto financiero presenta particularidades que van más allá de las métricas estándar de aprendizaje automático. Esta sección introduce los conceptos fundamentales necesarios para interpretar correctamente los resultados del Capítulo 4.

#### Métricas de clasificación y sus limitaciones

En problemas de clasificación binaria de dirección de mercado, la precisión estándar (*accuracy*) puede resultar engañosa cuando las clases están desbalanceadas: un modelo que siempre prediga la clase mayoritaria obtendría una precisión aparentemente aceptable sin aportar información predictiva real. La precisión equilibrada (*balanced accuracy*) corrige este problema promediando la tasa de acierto sobre cada clase. El coeficiente de correlación de Matthews (MCC) considera los cuatro elementos de la matriz de confusión y proporciona una medida más robusta en escenarios desbalanceados. El ROC-AUC evalúa la capacidad discriminativa del modelo a través de todos los umbrales de decisión posibles.

En el contexto de la predicción financiera, López de Prado (2018) señala que incluso valores de precisión equilibrada del orden del 53% pueden ser económicamente significativos si son robustos y estables en el tiempo. Sin embargo, la significancia estadística no implica necesariamente rentabilidad real: una pequeña ventaja estadística puede quedar completamente eliminada por los costes de transacción derivados de la operativa activa.

#### Validación temporal y backtesting

La validación de modelos sobre series temporales financieras requiere respetar estrictamente el orden cronológico de los datos. El método clásico de validación cruzada en *K* particiones no resulta adecuado para series temporales, ya que puede permitir que el modelo se entrene con datos futuros respecto a los empleados en la validación, generando estimaciones artificialmente optimistas (López de Prado, 2018).

El *backtesting* —evaluación retrospectiva de una estrategia sobre datos históricos— constituye la herramienta estándar para estimar el comportamiento de una estrategia en producción. Sin embargo, presenta riesgos propios: el sobreajuste al período de evaluación, el sesgo de selección de la ventana temporal o la no consideración de los costes de fricción reales pueden conducir a estimaciones excesivamente optimistas que no se reproducen en condiciones reales de mercado.

#### De la predicción a la rentabilidad financiera

La brecha entre la capacidad predictiva estadística y la rentabilidad financiera real es uno de los fenómenos mejor documentados en la literatura de *machine learning* aplicado a finanzas. Esta brecha tiene varias causas. En primer lugar, los costes de transacción —comisiones, diferencial entre precio de compra y venta, impacto de mercado— erosionan directamente el margen generado por el modelo. En segundo lugar, la disponibilidad de la información: si la señal predictiva proviene de datos ya incorporados por el mercado antes de la apertura de la sesión, la ventaja estadística detectada puede no ser explotable en la práctica. En tercer lugar, el período de simulación puede no ser representativo del comportamiento general del mercado si coincide con un régimen de mercado particular.

Métricas financieras como el ratio de Sharpe (Sharpe, 1966) —cociente entre el exceso de retorno y la volatilidad de la estrategia— o la pérdida máxima (*maximum drawdown*) permiten contextualizar el rendimiento ajustado al riesgo y compararlo con estrategias de referencia pasivas, como la estrategia de comprar y mantener el índice.

---

## Capítulo 3. Metodología y desarrollo del sistema

### 3.1 Arquitectura general del sistema

La Figura 3 muestra el esquema general de la arquitectura del sistema desarrollado. El sistema parte de dos fuentes principales de información: los datos de mercado y las noticias financieras. Ambos flujos convergen en una base de datos centralizada, desde la cual se ejecutan los modelos de predicción. Las predicciones generadas son consumidas por tres servicios: una página web de visualización, un boletín informativo automatizado y un bot de trading.

![Figura 3](figura_3.png)

*Figura 3. Diagrama de arquitectura del sistema automatizado. Los dos flujos de entrada —datos de mercado y noticias financieras— convergen en una base de datos centralizada, a partir de la cual se generan predicciones que alimentan la página web, el boletín informativo y el bot de trading.*

La arquitectura sigue una separación clara de responsabilidades en bloques independientes. Cada tarea del pipeline se ejecuta de forma desacoplada mediante automatizaciones gestionadas con GitHub Actions (GitHub Inc., 2026). Este enfoque mejora la robustez y mantenibilidad del sistema: un fallo en una etapa concreta no interrumpe necesariamente el resto del flujo de trabajo.

El bloque de datos de mercado construye y mantiene una base de datos histórica de información bursátil actualizada. Las automatizaciones verifican que tanto el mercado español como el estadounidense hayan cerrado antes de actualizar los registros diarios, evitando almacenar valores incompletos. Los datos se almacenan en una base de datos en la nube mediante Supabase (Supabase Inc., 2026), y periódicamente se migran a una base de datos local SQLite para tareas de entrenamiento y experimentación.

El bloque de predicción recupera de la base de datos en la nube las observaciones más recientes de cada activo —aproximadamente 300 sesiones por activo— para construir las características y generar nuevas predicciones con los modelos ya entrenados. Las predicciones resultantes se almacenan de nuevo en la base de datos centralizada, desde donde son consumidas por los servicios finales.

### 3.2 Datos y fuentes de información

#### Series temporales financieras

Se recopilaron series temporales diarias correspondientes a los activos financieros del IBEX 35. Cada registro incluye el precio de apertura, el precio máximo y mínimo de la sesión, el precio de cierre ajustado y el volumen de negociación. Los precios ajustados incorporan correcciones por dividendos y operaciones corporativas, garantizando la consistencia de la serie histórica a efectos de comparación temporal.

Los activos considerados corresponden a las compañías pertenecientes al IBEX 35 durante el período de estudio: ACS, Acerinox, Aena, Amadeus IT Group, Acciona, Acciona Energía, BBVA, Bankinter, CaixaBank, Cellnex Telecom, Colonial, Endesa, Enagás, Fluidra, Ferrovial, Grifols, IAG, Iberdrola, Inditex, Logista, Mapfre, Merlin Properties, ArcelorMittal, Naturgy, Puig, Redeia, Banco Sabadell, Banco Santander, Telefónica y Unicaja Banco.

El conjunto de datos contiene 129.910 registros, comprendidos entre enero de 2006 y abril de 2026. La fecha de inicio no es arbitraria: aunque algunas compañías disponen de datos anteriores, antes de 2006 existen inconsistencias relevantes en determinados activos, como fechas faltantes, volúmenes negativos o nulos, precios erróneos y discontinuidades temporales. Durante el preprocesamiento se filtraron los registros con volumen nulo o negativo. Es importante señalar que el universo de activos está sujeto a **sesgo de supervivencia**, ya que únicamente se incluyen los componentes actuales del IBEX 35, omitiendo empresas históricamente excluidas del índice; este sesgo puede inflar ligeramente las métricas de rendimiento.

Además de las compañías individuales, se recopilaron indicadores de mercado e índices de referencia: el IBEX 35, el S&P 500 y el índice de volatilidad VIX. El histórico de estos índices está disponible desde fechas anteriores a 2006, lo que permite disponer de una ventana de precalentamiento suficiente para construir características basadas en ventanas móviles amplias. Dado que los mercados español y estadounidense operan en calendarios distintos con festivos diferentes, fue necesario alinear ambos calendarios: los flujos (retornos, variaciones) se rellenan con cero en los días en que el mercado correspondiente permanece cerrado, mientras que las variables de estado (niveles, volatilidades) se propagan hacia adelante (*forward fill*).

Todos estos datos fueron obtenidos mediante la librería yfinance (Ranaroussi, 2023), que actúa como interfaz de acceso a Yahoo Finance.

#### Noticias financieras

Adicionalmente, se recopiló un conjunto de noticias financieras procedentes del diario económico Expansión, concretamente de las secciones de economía, mercados, empresas y ahorro, abarcando el período comprendido desde enero de 2026. **Estas noticias no forman parte del entrenamiento de los modelos predictivos actuales**; constituyen una fuente de información cualitativa almacenada en el sistema para futuras extensiones relacionadas con análisis de sentimiento y modelado multimodal.

Para cada noticia se almacenan los siguientes campos originales: título, cuerpo, sección, URL y fecha de publicación. A partir de estos datos se generan automáticamente varios atributos derivados: la categoría de la noticia (específica de compañía, macroeconómica, sentimiento de mercado o ruido general), la relevancia estimada en el rango [0, 1], el sentimiento asociado (positivo, negativo o neutro) y las compañías mencionadas.

### 3.3 Preparación de datos y definición de la señal

#### Definición de la señal

La señal se construye a partir del retorno logarítmico futuro del activo. Para cada día *t*, se calcula el retorno logarítmico entre el precio de cierre en *t* y el precio de cierre en *t* + *h*, donde *h* representa el horizonte temporal considerado:

$$r_{t,h} = \ln\left(\frac{P_{t+h}}{P_t}\right)$$

donde $P_t$ es el precio de cierre ajustado del activo en el instante *t* y $P_{t+h}$ es el precio de cierre ajustado tras un horizonte temporal *h*. A partir de este valor, la variable objetivo se define de forma binaria:

$$y_t = \begin{cases} 1 & \text{si } r_{t,h} > 0 \\ 0 & \text{en caso contrario} \end{cases}$$

Los horizontes considerados son de 1 y 5 días. El horizonte de 1 día permite evaluar la capacidad del modelo para anticipar la dirección inmediata del mercado. El horizonte de 5 días busca capturar tendencias ligeramente más estables y menos sensibles al ruido diario.

#### Exploración de datos

La predicción de series financieras constituye un problema especialmente complejo debido a la propia naturaleza de los retornos bursátiles. Los retornos logarítmicos suelen presentar un comportamiento próximo al de un proceso de ruido blanco, caracterizado por una elevada aleatoriedad y una baja dependencia temporal entre observaciones consecutivas (Fama, 1970).

La Figura 1 muestra la evolución temporal de los retornos logarítmicos de un activo representativo del IBEX 35. Puede observarse que la serie presenta oscilaciones rápidas y aparentemente irregulares, sin patrones visuales persistentes ni tendencias fácilmente identificables.

![Figura 1](figura_1.png)

*Figura 1. Evolución temporal de los retornos logarítmicos diarios de un activo representativo del IBEX 35. La serie presenta oscilaciones rápidas sin tendencia persistente, evidenciando el elevado nivel de ruido característico de las series financieras.*

El análisis de autocorrelación representado en la Figura 2 refuerza esta idea. Los coeficientes de autocorrelación para distintos retardos se mantienen próximos a cero, lo que indica una débil relación lineal entre los retornos actuales y los retornos pasados, en línea con los resultados de Lo y MacKinlay (1988).

![Figura 2](figura_2.png)

*Figura 2. Función de autocorrelación (ACF) de los retornos logarítmicos diarios. Los coeficientes se mantienen próximos a cero para todos los retardos, lo que confirma la débil dependencia temporal de la serie.*

#### Feature engineering

Con el objetivo de mejorar la capacidad predictiva de los modelos y facilitar el aprendizaje de patrones temporales consistentes, se llevó a cabo un proceso de ingeniería de características (*feature engineering*) sobre las series financieras originales. En lugar de utilizar precios absolutos o retornos simples, se emplearon retornos logarítmicos, lo que permite representar las variaciones relativas del precio de forma más consistente a lo largo del tiempo y reduce los problemas asociados a las diferencias de escala entre activos.

Todas las características temporales se construyeron respetando estrictamente la causalidad temporal: cada ventana de datos utiliza únicamente información disponible hasta el instante actual, sin incorporar datos futuros en ninguna transformación o cálculo. Asimismo, se eliminaron variables que pudieran introducir sesgos no deseados, como identificadores explícitos (fecha exacta, nombre de la compañía) y variables intermedias utilizadas únicamente para construir nuevas características.

Las características generadas se agrupan en las siguientes categorías:

**Retornos.** Retorno logarítmico a 1, 5, 10 y 20 días; retorno intradía (cierre menos apertura normalizado); hueco (*gap*) entre apertura del día y cierre del día anterior; momentum 12-1 meses (solo H5).

**Volatilidad.** Desviación estándar de retornos en ventanas de 5, 10 y 20 días; ratio de volatilidades corto/largo plazo (5/20); Average True Range normalizado ATR(14); bandas de Bollinger %B (posición del precio dentro de las bandas); pendiente de regresión lineal del precio (10 días); distancia relativa al máximo y mínimo de las ventanas de 10 y 20 días.

**Volumen y liquidez.** Ratio de volumen reciente frente al promedio (1/5 y 1/20 días); pendiente del OBV (On-Balance Volume, ventana 10 días); ratio de iliquidez de Amihud (2002) sobre ventana de 10 días; retorno ponderado por volumen.

**Indicadores técnicos.** Medias móviles simples y exponenciales en ratios (5/20 y 10/50); histograma MACD normalizado por precio (12, 26, 9); RSI(14); autocorrelación de retornos en ventana de 10 días.

**Variables de mercado.** Amplitud de mercado del IBEX 35 calculada mediante *leave-one-out* (fracción de valores con retorno positivo), tanto a 1 día como media móvil a 10 días; retorno relativo del activo frente al IBEX 35 a 5 y 20 días; volatilidad del IBEX 35 (10 días) y ratio de volatilidades (10/60); volatilidad relativa del activo frente al IBEX 35 (20 días); volatilidad del S&P 500 (20 días) y ratio de volatilidades (20/100); variación estandarizada del VIX en 5 días; percentil del VIX en los últimos 250 días.

**Variables temporales.** Día de la semana (solo H1): codificado como entero para los modelos de árbol, y mediante codificación sinusoidal (*sin/cos*) para las redes neuronales recurrentes, con el objetivo de reflejar la naturaleza cíclica del calendario semanal. Mes del año (solo H5).

La Tabla 1 presenta un resumen de las características finales utilizadas.

**Tabla 1. Características generadas mediante feature engineering. Los números entre paréntesis indican el tamaño de la ventana temporal empleada en cada cálculo.**

| Característica | Explicación breve | Característica | Explicación breve |
|---|---|---|---|
| Logaritmo retorno (1; 5; 10; 20) | Retorno logarítmico en distintos horizontes temporales | Retorno intradía (1) | Diferencia entre precio de cierre y apertura normalizada |
| Volatilidad (5) | Desviación estándar de retornos en ventana de 5 días | Cuerpo | Tamaño absoluto del cuerpo de la vela japonesa |
| Ratio volatilidad (5/20) | Cociente entre volatilidades de corto y largo plazo | Mecha superior | Proporción de la mecha superior respecto al rango total |
| Rango verdadero medio (14) | Average True Range; medida de volatilidad intradía | Mecha inferior | Proporción de la mecha inferior respecto al rango total |
| Ratio media móvil (5/20; 10/50) | Cociente entre medias móviles de distintos periodos | Hueco | Diferencia entre apertura del día y cierre del día anterior |
| MACD (12, 26, 9) | Diferencia entre medias exponenciales de distinta longitud | Día de la semana | Variable cíclica (codificación sinusoidal en RNN) |
| Bandas de Bollinger (20) | Posición del precio dentro de las bandas superior e inferior | Amplitud del mercado (1; 10) | Proporción de valores del IBEX 35 con retorno positivo (leave-one-out) |
| Pendiente (10) | Pendiente de la regresión lineal del precio sobre una ventana de 10 días | Retorno relativo IBEX (5; 20) | Diferencia entre retorno del activo y retorno del índice |
| Distancia al máximo (10; 20) | Distancia relativa al máximo de la ventana | Volatilidad IBEX (10) | Desviación estándar de retornos del IBEX 35 |
| Distancia al mínimo (10; 20) | Distancia relativa al mínimo de la ventana | Ratio volatilidad IBEX (10/60) | Cociente de volatilidades del IBEX 35 en distintos horizontes |
| RSI (14) | Índice de fuerza relativa en ventana de 14 días | Volatilidad relativa IBEX (20) | Cociente entre volatilidad del activo y la del índice |
| Ratio volumen (1/5; 1/20) | Cociente entre el volumen reciente y el volumen promedio | Volatilidad S&P 500 (20) | Desviación estándar de retornos del S&P 500 |
| Pendiente del OBV (10) | Tendencia del volumen on-balance en una ventana de 10 días | Ratio volatilidad S&P 500 (20; 100) | Cociente entre volatilidades del S&P 500 en distintos horizontes |
| Iliquidez Amihud (10) | Ratio de iliquidez de Amihud (2002): impacto del precio por unidad de volumen | Cambio VIX (5) | Variación estandarizada del índice de volatilidad implícita en 5 días |
| Retornos autocorrelacionados (10) | Autocorrelación de los retornos en una ventana de 10 días | Percentil VIX (250) | Percentil del VIX actual respecto al histórico de los últimos 250 días |

### 3.4 Modelos predictivos

#### Modelo de Markov

El modelo de Markov de primer orden discretiza el retorno logarítmico del día anterior (`log_ret_1`) en *n* = 3 estados mediante cuantiles equiprobables, obteniendo tres cubos: retorno bajo (tercil inferior), neutro (tercil central) y alto (tercil superior). El estado del sistema en cada instante *t* queda definido por el cubo al que pertenece el retorno del día anterior. La probabilidad de transición $P(\text{up} \mid s)$ para cada estado *s* se estima como la fracción empírica de movimientos alcistas observados tras ese estado en el período de entrenamiento, aplicando suavizado de Laplace (*alpha* = 1) para gestionar estados no observados. En inferencia, el modelo mapea el retorno observado al cubo correspondiente y devuelve $P(\text{up} \mid s)$; si el estado no fue observado durante el entrenamiento, se devuelve la fracción base de movimientos alcistas del conjunto de entrenamiento. La predicción binaria se obtiene comparando $P(\text{up} \mid s)$ con el umbral 0,5.

#### Modelos de árbol (Random Forest y XGBoost)

Random Forest entrena 500 árboles sobre subconjuntos aleatorios de datos y variables, obteniendo la predicción final por votación mayoritaria. La configuración adoptada limita deliberadamente la complejidad individual de cada árbol (profundidad máxima 5, mínimo de 50 observaciones por hoja, 30% de variables por división) para forzar que la capacidad del modelo emerja de la agregación y no del ajuste individual.

XGBoost construye árboles secuencialmente corrigiendo los errores residuales de los estimadores anteriores mediante potenciación de gradiente. La parada anticipada (*early stopping*) sobre una partición temporal interna del 80/20 determina el número óptimo de rondas de boosting, evitando el sobreajuste sin sacrificar capacidad. Los árboles son poco profundos (profundidad máxima 3) y aplican submuestreo estocástico tanto de filas como de columnas.

#### Redes neuronales recurrentes (LSTM y GRU)

Tanto LSTM como GRU reciben como entrada una secuencia de los *T* = 20 días más recientes de características. Las características se estandarizan (*z-score*) usando únicamente estadísticos del conjunto de entrenamiento. El día de la semana se codifica mediante sin/cos cíclico antes de la normalización. La arquitectura es deliberadamente compacta: una única capa recurrente con estado oculto de 64 unidades y *dropout* de 0,3, seguida de una capa lineal de salida. El modelo se comparte entre los 30 tickers (pooling cross-seccional). El entrenamiento emplea AdamW con *learning rate* 3×10⁻⁴ y parada anticipada con paciencia de 10 épocas sobre una partición temporal interna del 20% del conjunto de entrenamiento.

**Tabla 2. Resumen de los modelos predictivos: entradas, salidas e hiperparámetros principales.**

| Modelo | Entrada | Salida | Hiperparámetros principales |
|---|---|---|---|
| Markov | `log_ret_1` (discretizado, 3 estados) | P(up \| estado) | n\_states=3, order=1, alpha=1.0 (Laplace) |
| Random Forest | Vector de 40 características (H1) / 42 (H5) | P(up), clase binaria | n\_estimators=500, max\_depth=5, max\_features=0.3, min\_samples\_leaf=50 |
| XGBoost | Vector de 40 características (H1) / 42 (H5) | P(up), clase binaria | max\_depth=3, lr=0.05, subsample=0.7, early\_stopping=30 rondas |
| LSTM | Secuencia (T=20, F=41) de características normalizadas | P(up), clase binaria | hidden=64, layers=1, dropout=0.3, ES patience=10, AdamW lr=3×10⁻⁴ |
| GRU | Secuencia (T=20, F=41) de características normalizadas | P(up), clase binaria | hidden=64, layers=1, dropout=0.3, ES patience=10, AdamW lr=3×10⁻⁴ |

### 3.5 Servicios automatizados del sistema

#### Página web

La solución se despliega mediante GitHub Pages como una aplicación web estática basada en HTML, CSS y JavaScript. La interfaz principal muestra predicciones para la siguiente sesión bursátil: cada activo aparece en una tarjeta individual con la dirección esperada del movimiento (verde: alcista, rojo: bajista) y la probabilidad asociada. Al seleccionar un activo, el usuario accede a una vista detallada con cotización actual, gráficos financieros e indicadores técnicos configurables.

En la parte inferior se incorpora un panel de noticias financieras en tiempo real con desplazamiento horizontal automático. La web también describe la arquitectura del modelo empleado e incluye una declaración de transparencia sobre las limitaciones del sistema. Las predicciones se actualizan automáticamente cada día laborable mediante GitHub Actions tras el cierre de los mercados.

#### Boletín informativo

El sistema genera diariamente un boletín por correo electrónico para los usuarios suscritos. El correo incluye las tres predicciones de mayor confianza del modelo y las diez noticias financieras más relevantes del día anterior, ordenadas según su relevancia estimada. El envío se realiza mediante el servidor SMTP de Gmail y soporta listas de distribución de hasta 500 destinatarios. Las direcciones de correo se almacenan en la base de datos en la nube.

#### Procesamiento de noticias

El módulo de procesamiento de noticias emplea un agente basado en modelos de lenguaje para analizar cada noticia y extraer: categoría temática, compañías mencionadas y sentimiento (positivo, negativo, neutro). Se utiliza el modelo openai/gpt-oss-120b (OpenAI, 2025) a través de la infraestructura de inferencia de Groq. Para mejorar la eficiencia dentro de los límites del plan gratuito, se agrupan aproximadamente diez noticias por llamada al modelo. La relevancia se calcula mediante un enfoque híbrido que combina la estimación semántica del modelo de lenguaje con reglas heurísticas basadas en palabras clave (dividendos, OPAs, resultados, adquisiciones, cambios regulatorios).

#### Bot de trading

El bot de trading lee diariamente las predicciones almacenadas y ejecuta las operaciones correspondientes a través de Interactive Brokers. Selecciona las *k* = 3 predicciones de mayor confianza y distribuye el capital proporcionalmente entre ellas. Se conecta localmente al puerto habilitado por la aplicación del bróker una vez el usuario ha iniciado sesión, e implementa lógica de reconexión automática. La tarea se programa mediante PowerShell para ejecutarse cada día laborable a las 9:00 (apertura del IBEX 35) y cerrarse a las 16:45.

### 3.6 Protocolo experimental

#### Validación temporal con ventana expansiva

En problemas de predicción financiera basados en series temporales resulta fundamental respetar el orden cronológico de los datos durante todo el proceso de entrenamiento y validación. Mezclar observaciones futuras en el entrenamiento produce estimaciones artificialmente optimistas que no se mantienen en producción (López de Prado, 2018).

El protocolo de validación combina dos niveles complementarios. El primero es la **validación cruzada purgada con ventana expansiva**, empleada para la estimación de métricas de clasificación. El entrenamiento comienza con una ventana inicial de 750 sesiones bursátiles (aproximadamente tres años) y se amplía progresivamente en pasos trimestrales (63 sesiones). A diferencia de la validación cruzada estándar en *K* particiones, que ignora el orden cronológico y puede mezclar datos futuros con datos pasados, la variante expansiva mantiene estrictamente la causalidad temporal: en cada fold, el conjunto de test siempre es posterior al de entrenamiento. Se introduce además un embargo de un día entre el último día de entrenamiento y el primer día de test, para reducir la correlación entre las observaciones más recientes de cada período.

Para el horizonte de un día se obtienen 70 ventanas de evaluación; para el horizonte de cinco días, 66 ventanas. El escalador de características se ajusta exclusivamente con los datos de entrenamiento de cada fold y se aplica a los conjuntos de validación y test, evitando fugas de información distribucional.

#### Métricas de clasificación

La precisión equilibrada (*balanced accuracy*) se emplea como métrica principal de optimización, ya que corrige el posible desbalance entre clases. El MCC complementa el análisis al considerar los cuatro elementos de la matriz de confusión. El ROC-AUC evalúa la capacidad discriminativa del modelo a través de todos los umbrales posibles.

Para evaluar si la precisión equilibrada media de cada modelo es estadísticamente superior al azar, se aplica una **prueba *t* de una muestra** con hipótesis nula $H_0: \mu = 0{,}5$ e hipótesis alternativa $H_a: \mu > 0{,}5$, utilizando como observaciones las puntuaciones obtenidas en las distintas ventanas de evaluación temporal. Dado que las ventanas de evaluación proceden de una serie temporal y pueden presentar dependencia, este contraste debe interpretarse como una evidencia exploratoria de señal predictiva, no como una prueba definitiva de independencia estadística. Los valores de $\pm$ reportados en la Tabla 3 corresponden a la desviación estándar de las puntuaciones entre ventanas.

#### Métricas financieras y simulación de trading

La evaluación financiera traslada las predicciones del modelo a un entorno de simulación de trading. La rentabilidad acumulada bruta mide el beneficio total antes de considerar costes de transacción; la rentabilidad neta incorpora dichos costes. La pérdida máxima (*drawdown* máximo) cuantifica la mayor caída experimentada desde un máximo hasta un mínimo consecutivo. El ratio de Sharpe (Sharpe, 1966) relaciona el exceso de retorno con la volatilidad de la estrategia.

El período de simulación comprende del 1 de mayo al 1 de junio de 2026. La cartera inicial es de 100.000 euros. La estrategia es exclusivamente *long*: cuando el modelo predice alza (pred=1), se abre una posición larga en el activo; cuando predice baja (pred=0), no se toma posición. Las posiciones se abren al inicio de cada sesión bursátil y se cierran al inicio de la siguiente, en función de las predicciones del día. El tamaño de las posiciones es uniforme entre los activos con predicción positiva, y se rebalancea diariamente. El coste de transacción aplicado es de 10 puntos básicos por operación de compraventa, cubriendo comisiones y diferencial compra-venta.

Se incluyen tres estrategias de referencia: (i) **comprar y mantener**, que consiste en mantener una posición larga en el índice IBEX 35 durante todo el período, sin rebalanceo; (ii) **clasificador aleatorio**, que genera señales de compra con probabilidad 0,5 de forma independiente para cada activo y sesión (ejecutado una sola vez, no promediado); y (iii) **momentum**, que predice que la dirección futura del activo coincidirá con el signo del retorno acumulado durante los últimos 5 días.

---

## Capítulo 4. Resultados experimentales

### 4.1 Resultados predictivos de los modelos

**Tabla 3. Métricas de los modelos en validación cruzada con ventana expansiva, ordenadas de mayor a menor precisión equilibrada. *H* hace referencia al horizonte de predicción en días. Los valores ± corresponden a la desviación estándar entre ventanas de evaluación. La columna "Significativo" indica si la precisión equilibrada es estadísticamente superior al 50% según la prueba *t* de una muestra (α = 0,05).**

| Modelo | Precisión equilibrada (%) | ROC-AUC | MCC | Significativo |
|---|---|---|---|---|
| Random Forest (H1) | 53,16 ± 3,28 | 0,5448 | 0,0649 | SÍ |
| GRU (H1) | 52,74 ± 3,06 | 0,5431 | 0,0564 | SÍ |
| XGBoost (H1) | 52,55 ± 3,28 | 0,5381 | 0,0541 | SÍ |
| LSTM (H1) | 52,34 ± 2,64 | 0,5383 | 0,0486 | SÍ |
| Random Forest (H5) | 51,76 ± 4,64 | 0,5301 | 0,0399 | SÍ |
| LSTM (H5) | 51,68 ± 3,64 | 0,5245 | 0,0371 | SÍ |
| GRU (H5) | 51,58 ± 3,59 | 0,5261 | 0,0356 | SÍ |
| XGBoost (H5) | 50,70 ± 3,46 | 0,5250 | 0,0145 | NO |
| Markov (H5) | 50,10 ± 1,18 | 0,5048 | 0,0016 | NO |
| Markov (H1) | 49,92 ± 0,92 | 0,4974 | −0,0016 | NO |

La Tabla 3 recoge el rendimiento medio en validación cruzada con ventana expansiva de los modelos evaluados para los horizontes de un día y cinco días. Los resultados sugieren la presencia de una **señal predictiva débil pero estadísticamente distinguible del azar** para los modelos de árbol y las redes recurrentes.

En el horizonte de un día, Random Forest, XGBoost, GRU y LSTM muestran precisiones equilibradas significativamente superiores al 50%, con valores *t* elevados. La magnitud absoluta de la diferencia es reducida: todos los modelos se sitúan entre el 52% y el 54%. López de Prado (2018) señala que en contextos financieros una precisión del orden del 53% puede ser indicativa de señal explotable si es robusta y estable, aunque no garantiza rentabilidad real. Este resultado es coherente con el rango de 52-55% reportado en la literatura para predicción diaria de renta variable (Krauss et al., 2017). El ROC-AUC y el MCC refuerzan este patrón: los valores de MCC, aunque positivos, son reducidos en términos absolutos (siempre por debajo de 0,07), lo que es consistente con la literatura sobre eficiencia de mercado en índices bursátiles desarrollados (Fama, 1970).

El modelo de Markov en ambos horizontes se mantiene prácticamente en el 50% de precisión equilibrada; la prueba *t* no rechaza la hipótesis nula, lo que indica que la información contenida en el estado discreto del retorno pasado no es suficiente para predecir la dirección futura del mercado.

Para el horizonte de cinco días, el rendimiento general se deteriora moderadamente. XGBoost pierde su significación estadística (p > 0,05) con un MCC de 0,0145, lo que sugiere que su ventaja predictiva es predominantemente de muy corto plazo.

Ha de tenerse en cuenta que la prueba estadística utilizada asume independencia entre ventanas, lo que puede no sostenerse en series temporales. Los resultados deben interpretarse, por tanto, como evidencia exploratoria de la presencia de señal predictiva, no como una confirmación definitiva.

### 4.2 Importancia de variables

El análisis de importancia de variables en los modelos de árbol —calculada como la reducción media de impureza (*mean decrease in impurity*) normalizada sobre todos los árboles del ensemble— revela un patrón consistente con implicaciones interpretativas relevantes.

![Figura 6](figura_6.png)

*Figura 6. Importancia relativa de las características en los modelos de árbol (Random Forest y XGBoost) para los horizontes H1 y H5. Las variables macroeconómicas, en particular la variación estandarizada del VIX y las métricas de volatilidad del S&P 500, dominan los rankings de importancia en ambos horizontes.*

En el horizonte de un día, la variable más influyente es la variación estandarizada del VIX estadounidense, seguida de métricas de volatilidad relativa del S&P 500. En el horizonte de cinco días, este patrón se mantiene con las variables macroeconómicas de origen externo encabezando los rankings. La preponderancia de estas señales sobre los indicadores técnicos individuales de cada acción sugiere que la dirección del mercado español está condicionada estructuralmente por el contexto macroeconómico global.

Estos resultados son directamente interpretables para los modelos de árbol. Para las arquitecturas recurrentes LSTM y GRU, la estructura interna es inherentemente más opaca y no permite extraer una importancia de variables equivalente, por lo que no es posible extrapolar directamente estas conclusiones a dichos modelos.

### 4.3 Resultados de la simulación financiera

**Tabla 4. Resultados de la simulación de trading para distintas estrategias durante el período de prueba (1 de mayo – 1 de junio de 2026). La estrategia "Mejor modelo" corresponde a Random Forest con horizonte H1. Los valores de rentabilidad son acumulados en el período.**

| Estrategia | Rentabilidad bruta (%) | Rentabilidad neta (%) | Pérdida máxima (%) | Ratio de Sharpe |
|---|---|---|---|---|
| Mejor modelo (RF H1) | +0,25% | −1,74% | −2,70% | −5,150 |
| Comprar y mantener | +1,15% | +1,15% | −2,93% | 0,825 |
| Clasificador aleatorio | +3,40% | +0,54% | −3,20% | 0,484 |
| Momentum | +2,30% | −0,52% | −4,31% | −0,768 |

La Tabla 4 muestra los resultados de la simulación de trading. La estrategia basada en Random Forest con horizonte de un día presenta una rentabilidad bruta ligeramente positiva (+0,25%), lo que indica que el modelo conserva cierta capacidad predictiva. Sin embargo, tras incorporar los costes de transacción, la rentabilidad neta pasa a ser negativa (−1,74%), con un ratio de Sharpe muy desfavorable (−5,150). La estrategia de comprar y mantener obtiene un mejor comportamiento tanto en rentabilidad neta (+1,15%) como en ratio de Sharpe (0,825).

Conviene señalar que el período de simulación es de un mes, lo que constituye una muestra reducida, y que los resultados pueden estar fuertemente condicionados por el régimen de mercado específico de ese período. En particular, el clasificador aleatorio alcanza una rentabilidad bruta de +3,40% antes de costes, resultado que refleja el comportamiento alcista general del mercado durante esas semanas y no una capacidad predictiva real. La simulación debe interpretarse principalmente como un ejercicio exploratorio orientado a ilustrar la diferencia entre capacidad predictiva estadística y rentabilidad financiera real, no como una evaluación concluyente del sistema.

### 4.4 Discusión

Los resultados de los capítulos anteriores permiten extraer el mensaje central de este trabajo: **los modelos de aprendizaje automático muestran una mejora estadística pequeña pero indicativa respecto al azar, pero esta mejora no se traduce en rentabilidad financiera positiva una vez se incorporan los costes de transacción reales y se compara con estrategias de referencia pasivas**.

La señal predictiva identificada proviene principalmente de variables macroeconómicas externas —variación del VIX y volatilidad del S&P 500—, lo que sugiere una dependencia estructural del IBEX 35 respecto al contexto global. Sin embargo, esta información queda disponible tras el cierre de Wall Street, y el mercado europeo puede incorporarla parcialmente durante los intercambios nocturnos antes de la apertura del IBEX 35 del día siguiente. En consecuencia, aunque el modelo detecte patrones estadísticamente significativos durante el entrenamiento, dicha ventaja puede no ser completamente explotable en condiciones reales.

El impacto acumulado de los costes de transacción es determinante: la operativa diaria sobre una cartera diversificada implica un elevado número de operaciones que erosionan completamente el reducido margen predictivo. Este resultado es coherente con la hipótesis del mercado eficiente en su forma débil (Fama, 1970) y recuerda una limitación fundamental del aprendizaje automático en finanzas: la significancia estadística es condición necesaria pero no suficiente para la rentabilidad práctica.

---

## Capítulo 5. Conclusiones

Este trabajo ha abordado el problema de la predicción de dirección del mercado bursátil español mediante técnicas de aprendizaje automático, integrando los resultados en un sistema automatizado de extremo a extremo. A continuación se presentan las principales conclusiones derivadas del estudio.

**Sobre la capacidad predictiva de los modelos.** Los resultados demuestran que los modelos basados en árboles de decisión (Random Forest, XGBoost) y redes neuronales recurrentes (GRU, LSTM) presentan una señal predictiva débil pero estadísticamente distinguible del azar sobre la dirección del IBEX 35, con precisiones equilibradas situadas entre el 52% y el 54% para el horizonte de un día. Este resultado es coherente con el rango de 52-55% reportado en la literatura para predicción diaria de renta variable (Krauss et al., 2017). El modelo de Markov, por el contrario, no supera el umbral del azar en ningún horizonte.

**Sobre la naturaleza de la señal predictiva.** La señal predictiva identificada proviene principalmente de variables macroeconómicas externas, en particular de la variación del VIX y de métricas de actividad del S&P 500. Este hallazgo sugiere que la dirección del mercado español está condicionada en mayor medida por el contexto global que por los indicadores técnicos individuales de cada acción, resultado consistente con la literatura sobre transmisión de información entre mercados internacionales.

**Sobre la brecha entre capacidad estadística y rentabilidad real.** A pesar de que los modelos presentan una precisión equilibrada estadísticamente significativa, la operativa diaria genera un número elevado de transacciones cuyos costes de fricción eliminan completamente el reducido margen predictivo. La estrategia de comprar y mantener supera en rentabilidad neta a todas las estrategias activas durante el período analizado.

**Sobre el horizonte temporal.** El rendimiento predictivo se deteriora de forma moderada al ampliar el horizonte de predicción de uno a cinco días. XGBoost pierde su significación estadística en H5, lo que sugiere que la señal que capta este modelo es predominantemente de muy corto plazo.

**Sobre el sistema construido.** Más allá de los resultados predictivos, este trabajo demuestra la viabilidad de construir un sistema de análisis financiero automatizado y completo —desde la ingesta de datos hasta la ejecución de operaciones— utilizando exclusivamente herramientas gratuitas y de código abierto. La arquitectura integra un pipeline de datos en la nube, múltiples modelos de aprendizaje automático, una página web con predicciones actualizadas diariamente, un boletín informativo automatizado y un bot de trading conectado a un bróker real.

**Limitaciones del estudio.** El trabajo presenta diversas limitaciones. El universo de activos está sujeto a sesgo de supervivencia, ya que únicamente se incluyen los componentes actuales del IBEX 35. El período de simulación financiera es reducido (un mes), por lo que los resultados pueden estar condicionados por el régimen de mercado específico de ese período. Los modelos no incorporan las noticias financieras recopiladas como variables de entrada. Los costes de transacción utilizados son fijos y no modelizan el impacto de mercado de órdenes de tamaño elevado.

En definitiva, este trabajo confirma que los mercados financieros desarrollados presentan ineficiencias débiles estadísticamente detectables mediante modelos de aprendizaje automático, pero que la traducción de estas señales en rentabilidad real es muy difícil debido a los costes de transacción, la rapidez con que el mercado incorpora información pública y la baja magnitud del efecto predictivo.

---

## Capítulo 6. Trabajo futuro

En una línea de trabajo futura, uno de los aspectos más relevantes consiste en ampliar de forma significativa la base de datos utilizada, incorporando un mayor volumen de información histórica de noticias para capturar mejor la evolución de los patrones informativos y su relación con los movimientos del mercado. La integración de noticias en el proceso de predicción posibilitaría incorporar información cualitativa mediante representaciones vectoriales de los textos (*embeddings*), capturando matices semánticos más ricos que una simple clasificación de sentimiento.

Asimismo, resulta pertinente ampliar el conjunto de variables explicativas mediante la inclusión de indicadores macroeconómicos adicionales, como la tasa de empleo, la inflación u otras métricas de actividad económica. También se abre la puerta a explorar combinaciones de indicadores menos convencionales, dado que muchas de las variables utilizadas en este trabajo son indicadores técnicos clásicos ampliamente seguidos por el mercado, lo que puede limitar su capacidad de generar ventaja diferencial.

Desde una perspectiva de gestión del riesgo, sería interesante incorporar reglas de gestión dinámica del tamaño de las posiciones en función de la volatilidad y definir zonas de entrada y salida del mercado basadas en la confianza del modelo, lo que puede contribuir tanto a mejorar la rentabilidad ajustada al riesgo como a dotar al sistema de mayor robustez en distintos regímenes de mercado.

Finalmente, la calibración de probabilidades del modelo —mediante técnicas como la escala de Platt o la regresión isotónica— podría mejorar la utilidad de las probabilidades generadas para el dimensionamiento de posiciones, ya que actualmente estas probabilidades no pueden interpretarse en sentido estadístico estricto.

---

## Bibliografía

Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*, (5), 31-56.

Bloomberg L.P. (2026). *Bloomberg Terminal*. Recuperat 15 de maig del 2026, des de https://www.bloomberg.com/professional/solution/bloomberg-terminal/

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., y Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5.ª ed.). Hoboken: John Wiley & Sons.

Breiman, L. (2001). Random forests. *Machine Learning*, (45), 5-32.

Chen, T., y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., y Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 1724-1734.

Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, (25), 383-417.

Fischer, T., y Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, (270), 654-669.

GitHub Inc. (2026). *GitHub Actions — Automate your workflow from idea to production*. Recuperat 15 de maig del 2026, des de https://docs.github.com/en/actions

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, (57), 357-384.

Hochreiter, S., y Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, (9), 1735-1780.

Krauss, C., Do, X. A., y Huck, N. (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. *European Journal of Operational Research*, (259), 689-702.

Lo, A. W., y MacKinlay, A. C. (1988). Stock market prices do not follow random walks: Evidence from a simple specification test. *The Review of Financial Studies*, (1), 41-66.

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Hoboken: John Wiley & Sons.

OpenAI. (2025). *GPT-oss-120b* (versió 2025). OpenAI. https://openai.com

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., y Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, (12), 2825-2830.

Ranaroussi, R. (2023). *yfinance: Yahoo! Finance market data downloader*. Recuperat 15 de maig del 2026, des de https://github.com/ranaroussi/yfinance

Sharpe, W. F. (1966). Mutual fund performance. *The Journal of Business*, (39), 119-138.

Supabase Inc. (2026). *Supabase: The Open Source Firebase Alternative*. Recuperat 15 de maig del 2026, des de https://supabase.com

TradingView Inc. (2026). *TradingView — Track All Markets*. Recuperat 15 de maig del 2026, des de https://www.tradingview.com

Yahoo Finance. (2026). *Yahoo Finance — Stock Market Live, Quotes, Business & Finance News*. Recuperat 15 de maig del 2026, des de https://finance.yahoo.com
