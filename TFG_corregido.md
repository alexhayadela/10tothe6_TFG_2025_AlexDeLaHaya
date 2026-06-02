ANÁLISIS, PREDICCIÓN Y EJECUCIÓN AUTOMATIZADA DE INVERSIONES EN EMPRESAS DEL IBEX 35

Trabajo de Fin de Grado de

Alex De La Haya Gutiérrez

Director: Vicenç Gómez Cerdà

Grado en Ingeniería Matemática en Ciencia de Datos

Curs 2025-2026

**Agradecimientos**

A mi familia y amigos por su apoyo constante y su confianza a lo largo de todo este proceso.

A mi tutor, Vicenç Gómez, por su orientación y ayuda en cada etapa del desarrollo de este trabajo.

A todos los programadores que han contribuido al desarrollo de librerías de código abierto, cuya labor ha hecho posible construir este proyecto sobre una base sólida.

Este trabajo no habría sido posible sin vuestra ayuda. Muchas gracias a todos.

**Resumen**

Lorem ipsum dolor sit amet, consectetur adipisici elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquid ex ea commodi consequat. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint obcaecat cupiditat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.Lorem ipsum dolor sit amet, consectetur adipisici elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquid ex ea commodi consequat. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint obcaecat cupiditat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.Lorem ipsum dolor sit amet, consectetur adipisici elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua.

**Resum** 

Lorem ipsum dolor sit amet, consectetur adipisici elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquid ex ea commodi consequat. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint obcaecat cupiditat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.Lorem ipsum dolor sit amet, consectetur adipisici elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquid ex ea commodi consequat. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint obcaecat cupiditat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.Lorem ipsum dolor sit amet, consectetur adipisici elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. 

**Abstract**

Lorem ipsum dolor sit amet, consectetur adipisici elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquid ex ea commodi consequat. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint obcaecat cupiditat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.Lorem ipsum dolor sit amet, consectetur adipisici elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquid ex ea commodi consequat. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint obcaecat cupiditat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.Lorem ipsum dolor sit amet, consectetur adipisici elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua.

---

**Índice**

[**Introducción	1**](#introducción)
[1.1  Contexto	1](#contexto)
[1.2  Motivación	1](#motivación)
[1.3  Objetivos	2](#objetivos)
[1.4  Enfoque del proyecto	2](#enfoque)

[**Marco Conceptual y Contexto	3**](#marco)
[2.1  Herramientas y plataformas para la inversión en activos financieros	3](#herramientas)
[2.2  Series temporales financieras y retos de predicción	4](#series)
[2.3  Modelos predictivos para datos financieros	5](#modelos-marco)
[2.3.1  Modelos estadísticos clásicos	5](#arima)
[2.3.2  Árboles de decisión	6](#arboles)
[2.3.3  Redes neuronales recurrentes	6](#rnn)
[2.4  De la predicción al rendimiento financiero	7](#prediccion-rendimiento)

[**Metodología y desarrollo del sistema	9**](#metodologia)
[3.1  Arquitectura general del sistema	9](#arquitectura)
[3.2  Datos y fuentes de información	9](#datos)
[3.3  Preparación de datos y definición de la señal	11](#preparacion)
[3.3.1  Definición de la señal	11](#senal)
[3.3.2  Exploración de datos	12](#exploracion)
[3.3.3  Feature engineering	12](#features)
[3.4  Modelos predictivos	14](#modelos-impl)
[3.5  Servicios automatizados del sistema	17](#servicios)
[3.5.1  Procesamiento de noticias	17](#noticias)
[3.5.2  Página web	18](#web)
[3.5.3  Boletín informativo	19](#boletin)
[3.5.4  Almacenamiento de predicciones y bot de trading	19](#trading)
[3.6  Protocolo experimental	20](#protocolo)

[**Resultados experimentales	23**](#resultados)
[4.1  Resultados predictivos de los modelos	23](#resultados-pred)
[4.2  Importancia de variables	25](#importancia)
[4.3  Resultados de la simulación financiera	26](#simulacion)
[4.4  Discusión de los resultados	27](#discusion)

[**Conclusiones	29**](#conclusiones)
[**Trabajo futuro	31**](#trabajo-futuro)
[**Bibliografía	32**](#bibliografia)
[**Apéndice	35**](#apendice)

---

**Capítulo 1**

# **Introducción**

## **1.1  Contexto**

El proyecto se centra en el análisis de activos financieros pertenecientes al IBEX 35, principal índice bursátil español, compuesto por las empresas con mayor capitalización y liquidez del mercado nacional. A partir de datos históricos de cotización, se estudia la evolución temporal de los distintos activos con el objetivo de identificar patrones y tendencias relevantes.

El análisis se orienta principalmente a la predicción de movimientos de mercado, evaluando si el precio de un activo tenderá a subir o bajar en horizontes temporales determinados. Para ello se emplean técnicas de análisis de series temporales y modelos de aprendizaje automático capaces de extraer información a partir de datos financieros históricos.

## **1.2  Motivación**

Este trabajo surge del interés por comprender el funcionamiento de los mercados financieros y analizar hasta qué punto es posible modelar su comportamiento mediante técnicas computacionales. Los mercados constituyen sistemas complejos, dinámicos y altamente influenciados por múltiples factores económicos, sociales y psicológicos, lo que los convierte en un ámbito especialmente atractivo desde el punto de vista analítico y tecnológico.

Además del interés académico, existe una motivación práctica relacionada con la necesidad de preservar y hacer crecer el capital en un contexto económico marcado por la inflación y la pérdida progresiva de poder adquisitivo. En este sentido, el estudio de estrategias de predicción financiera y modelos automatizados de inversión representa una oportunidad para explorar herramientas que puedan contribuir a una mejor toma de decisiones.

Otro de los motivos para desarrollar este proyecto es su potencial de continuidad más allá del propio Trabajo de Fin de Grado. Se trata de un área en constante evolución, con múltiples posibilidades de ampliación y mejora, tanto desde el punto de vista técnico como de investigación. Por ello, el proyecto se plantea no solo como un ejercicio académico, sino como una base sobre la que continuar desarrollando nuevas líneas de trabajo en el futuro.

## **1.3  Objetivos**

El objetivo principal de este trabajo es estudiar si los modelos de aprendizaje automático pueden mejorar las capacidades predictivas de los enfoques tradicionales utilizados en los mercados financieros, o al menos competir con ellos en términos de rendimiento y robustez.

Para ello, se analizan y comparan diferentes aproximaciones, incluyendo modelos estadísticos clásicos, estrategias basadas en reglas simples y técnicas de aprendizaje automático aplicadas a series temporales financieras. A través de esta comparación se pretende evaluar las ventajas, limitaciones y viabilidad real de cada enfoque en escenarios de mercado.

De manera más concreta, los objetivos del proyecto son los siguientes:

* Desarrollar un sistema automatizado y reproducible de análisis financiero que integre la obtención de datos, la generación de predicciones y su explotación mediante una página web, un boletín informativo y un bot de trading.  
* Diseñar, implementar y comparar distintos modelos predictivos para activos del IBEX 35, incluyendo métodos clásicos, modelos basados en reglas y modelos de aprendizaje automático.  
* Evaluar la viabilidad predictiva y financiera del sistema, analizando si las predicciones sobre la dirección de los activos en horizontes de uno y cinco días permiten obtener una mejora respecto a estrategias de referencia, tanto en métricas de clasificación como en simulaciones de trading con costes de transacción.

## **1.4  Enfoque del proyecto**

Uno de los aspectos relevantes de este trabajo es que no se limita al desarrollo de pruebas aisladas en cuadernos interactivos o experimentos puntuales. El objetivo es construir una estructura completa y organizada, similar a un sistema real de análisis financiero automatizado.

Para ello, se plantea una arquitectura compuesta por diferentes etapas: obtención de datos, procesamiento, entrenamiento de modelos, generación de predicciones y evaluación de resultados. Todo el flujo se diseña de forma que pueda ejecutarse de manera automática y reproducible, facilitando tanto la experimentación como la posible ampliación futura del proyecto.

El proyecto se ha desarrollado con un enfoque de presupuesto cero, utilizando únicamente herramientas gratuitas y de código abierto tanto para el procesamiento de datos como para el entrenamiento y el despliegue de modelos. Esta decisión no solo reduce las barreras de acceso, sino que también favorece la reproducibilidad y la accesibilidad del trabajo para otros estudiantes o investigadores interesados en el área.

Asimismo, el proyecto sigue una filosofía de ciencia abierta, priorizando la claridad en la documentación y la reproducibilidad de los experimentos. De este modo, cualquier persona interesada podría comprender la metodología empleada, replicar los resultados obtenidos o utilizar el sistema como base para futuras investigaciones o desarrollos.

---

**Capítulo 2**

# **Marco Conceptual y Contexto**

Este capítulo sitúa el trabajo en su contexto teórico y tecnológico. En primer lugar se describen las herramientas y plataformas existentes para el análisis de activos financieros, identificando sus limitaciones desde el punto de vista de la investigación. A continuación se exponen las características propias de las series temporales financieras que hacen de la predicción un problema especialmente difícil. El capítulo continúa con una revisión de los principales enfoques modelizadores —estadísticos clásicos, métodos de conjunto y redes neuronales recurrentes— para cerrar con una discusión conceptual sobre la evaluación de estrategias predictivas y la brecha entre capacidad estadística y rendimiento financiero real.

## **2.1  Herramientas y plataformas para la inversión en activos financieros**

Actualmente existen numerosas plataformas destinadas al análisis de mercados financieros y al apoyo a la inversión. Entre las más populares destacan Yahoo Finance (Yahoo Finance, 2026) y TradingView (TradingView, 2026), que ofrecen distintos niveles de acceso según el tipo de suscripción.

En sus versiones gratuitas, estas plataformas permiten consultar datos históricos básicos, normalmente con frecuencia diaria o con retrasos de entre 15 y 30 minutos respecto al tiempo real. Además, proporcionan herramientas de visualización de gráficos, incorporación de indicadores técnicos y acceso a noticias financieras relacionadas con los mercados o activos específicos.

Las versiones de pago, que van desde los 10 hasta los 200 euros mensuales, amplían estas funcionalidades mediante acceso a un mayor histórico de datos, actualización en tiempo real, creación de alertas y herramientas avanzadas de análisis técnico. Sin embargo, estas plataformas presentan varias limitaciones desde el punto de vista de la investigación y el desarrollo de sistemas predictivos.

Por un lado, muchas de ellas no ofrecen una API pública completa o presentan restricciones importantes para la extracción automatizada de datos históricos y noticias financieras. Asimismo, la integración directa con brókers suele ser limitada o inexistente, dificultando la automatización completa de estrategias de inversión.

En un nivel más profesional destaca Bloomberg Terminal (Bloomberg L.P., 2026), ampliamente utilizado en entornos financieros institucionales. Esta solución proporciona datos de mercado de alta fidelidad en tiempo real, acceso a noticias financieras, herramientas de análisis cuantitativo, pruebas retrospectivas y soporte para trading algorítmico automatizado. No obstante, su coste es elevado, situándose alrededor de los 30.000 euros anuales, lo que limita su accesibilidad fuera del ámbito profesional o institucional.

A pesar de las amplias funcionalidades disponibles, estas plataformas están orientadas principalmente al análisis financiero tradicional y al trading algorítmico basado en reglas definidas manualmente. Es importante diferenciar este enfoque del uso de técnicas de aprendizaje automático. Las pruebas retrospectivas tradicionales consisten en evaluar estrategias predefinidas utilizando datos históricos —por ejemplo, mediante reglas fijas de compra y venta basadas en indicadores técnicos—. En cambio, los modelos de aprendizaje automático buscan aprender patrones complejos directamente a partir de los datos para realizar predicciones futuras, adaptándose dinámicamente a nuevas condiciones del mercado.

Las plataformas analizadas no proporcionan de forma integrada herramientas completas para el entrenamiento, validación y despliegue de modelos de aprendizaje automático. Incluso en entornos avanzados como Bloomberg Terminal, el desarrollo de modelos predictivos suele realizarse externamente mediante lenguajes y bibliotecas especializadas, como Python junto con frameworks de ciencia de datos y aprendizaje automático (Pedregosa et al., 2011). Posteriormente, los resultados deben reintegrarse en la plataforma o conectarse directamente con brókers para ejecutar operaciones. Esta situación justifica el interés en desarrollar soluciones específicas que integren obtención de datos, análisis financiero y modelos de aprendizaje automático dentro de un mismo entorno.

## **2.2  Series temporales financieras y retos de predicción**

La predicción de series temporales financieras constituye una de las principales aplicaciones del análisis cuantitativo en mercados financieros. En este contexto, una serie temporal puede definirse como una secuencia de observaciones ordenadas cronológicamente, donde cada valor depende potencialmente de los anteriores. En finanzas, ejemplos habituales son el precio de cierre de una acción, el volumen negociado o los retornos de un activo.

El precio de los activos financieros suele considerarse una serie no estacionaria, ya que sus propiedades estadísticas cambian con el tiempo debido a factores económicos, políticos o sociales. Esta no estacionariedad dificulta el modelado y la predicción directa de los precios. Por este motivo, gran parte de la literatura trabaja con los retornos financieros en lugar de los precios absolutos, ya que los retornos presentan un comportamiento más cercano a la estacionariedad y facilitan el uso de modelos estadísticos y de aprendizaje automático.

En este contexto, el objetivo del trabajo no es predecir con precisión el precio futuro de los activos, sino evaluar si existe una señal débil que permita anticipar la dirección del movimiento en horizontes cortos. Por ello, el problema se formula como una tarea de clasificación binaria sobre el signo del retorno futuro, y no como una regresión directa sobre precios.

En las series temporales financieras resulta especialmente relevante el estudio de la autocorrelación, es decir, la dependencia existente entre valores observados en distintos instantes temporales. Analizar si los valores pasados contienen información útil para predecir los futuros es fundamental para determinar la aplicabilidad de determinados modelos predictivos. Lo y MacKinlay (1988) demostraron que los precios de las acciones no siguen exactamente un paseo aleatorio (*random walk*), aunque las desviaciones observadas son pequeñas y su explotabilidad práctica es limitada.

La predicción de series temporales financieras presenta una dificultad adicional derivada de la hipótesis del mercado eficiente (Fama, 1970). Esta hipótesis establece que el precio de los activos refleja toda la información disponible en el mercado, por lo que resulta extremadamente difícil obtener ventajas predictivas consistentes utilizando únicamente información histórica. En consecuencia, los movimientos futuros del mercado presentan un elevado componente aleatorio y una baja relación señal-ruido, lo que limita la capacidad predictiva de muchos modelos.

## **2.3  Modelos predictivos para datos financieros**

La literatura sobre predicción financiera cuantitativa ha desarrollado una amplia variedad de enfoques, que van desde los modelos estadísticos clásicos de series temporales hasta las arquitecturas de aprendizaje profundo más recientes. Cada familia de modelos refleja hipótesis distintas sobre la estructura de dependencia de los datos financieros: los métodos clásicos asumen formas de dependencia paramétrica y bien caracterizadas, mientras que los modelos de aprendizaje automático buscan aprender esas estructuras directamente a partir de los datos. En las subsecciones siguientes se describen los enfoques relevantes para este trabajo.

### **2.3.1  Modelos estadísticos clásicos**

Entre los enfoques tradicionales para la predicción de series temporales destacan los modelos ARIMA (*Autoregressive Integrated Moving Average*). Estos modelos combinan componentes autorregresivos y de medias móviles para capturar dependencias temporales en la serie, y han sido ampliamente utilizados en finanzas para la predicción de precios y volatilidad debido a su simplicidad e interpretabilidad (Box et al., 2015). Sin embargo, su capacidad para capturar relaciones no lineales es limitada, lo que restringe su aplicabilidad en el contexto de este trabajo, donde el problema se formula como clasificación binaria sobre el signo del retorno.

Otro enfoque clásico son las cadenas de Markov, en las que el sistema se modela mediante un conjunto finito de estados y probabilidades de transición entre ellos. La hipótesis fundamental es que el estado futuro depende únicamente del estado actual y no de toda la trayectoria previa. En finanzas, los modelos de Markov se han utilizado para modelar cambios de régimen de mercado, tendencias alcistas y bajistas o variaciones en volatilidad (Hamilton, 1989). A diferencia de ARIMA, este enfoque se adapta directamente al problema de clasificación binaria considerado en este trabajo, razón por la cual se incluye como modelo de referencia en los experimentos.

### **2.3.2  Árboles de decisión**

Los modelos basados en árboles de decisión han ganado popularidad en el ámbito financiero debido a su robustez y capacidad para modelar relaciones no lineales. Aunque estos modelos no incorporan de forma explícita el orden temporal de los datos, pueden utilizar información histórica mediante ingeniería de características, incluyendo retardos, medias móviles o indicadores técnicos como variables de entrada.

Random Forest (Breiman, 2001) es un método de aprendizaje ensamblado que combina múltiples árboles de decisión generados a partir de subconjuntos aleatorios de datos y variables. La predicción final se obtiene mediante votación o promedio entre los distintos árboles, lo que reduce el sobreajuste y mejora la capacidad de generalización. En finanzas se ha utilizado para la predicción de retornos y movimientos de mercado debido a su estabilidad frente al ruido característico de los datos financieros (Krauss et al., 2017).

XGBoost (Chen y Guestrin, 2016) es una técnica de potenciación de gradiente basada en árboles de decisión que construye modelos secuencialmente, corrigiendo en cada iteración los errores cometidos por los árboles anteriores. Este enfoque permite capturar relaciones complejas y suele ofrecer un alto rendimiento predictivo. XGBoost ha sido ampliamente empleado en problemas financieros, especialmente en predicción de precios y clasificación de tendencias bursátiles (Krauss et al., 2017).

### **2.3.3  Redes neuronales recurrentes**

Las redes neuronales recurrentes (RNN) están diseñadas específicamente para procesar datos secuenciales, ya que mantienen información de estados anteriores mediante conexiones recurrentes. Esto las hace especialmente adecuadas para el análisis de series temporales financieras.

Sin embargo, las RNN tradicionales presentan dificultades para capturar dependencias a largo plazo debido a problemas como el desvanecimiento del gradiente. Para solucionar estas limitaciones surgieron arquitecturas más avanzadas: las redes LSTM (*Long Short-Term Memory*) incorporan mecanismos de memoria y compuertas que permiten conservar información relevante durante periodos prolongados (Hochreiter y Schmidhuber, 1997). De forma similar, las redes GRU (*Gated Recurrent Unit*) simplifican esta arquitectura reduciendo el número de parámetros y el coste computacional (Cho et al., 2014). Ambos modelos reciben habitualmente como entrada ventanas temporales de observaciones pasadas para predecir valores futuros de la serie.

En el ámbito financiero, LSTM y GRU se han utilizado ampliamente para la predicción de precios, retornos y volatilidad debido a su capacidad para capturar patrones temporales complejos (Fischer y Krauss, 2018).

No obstante, el uso de modelos de aprendizaje profundo en finanzas presenta ciertas dificultades relacionadas con la disponibilidad limitada de datos. Aunque las series financieras contienen miles de observaciones, esta cantidad sigue siendo reducida en comparación con otros dominios del aprendizaje profundo, donde suelen emplearse millones de ejemplos. Como consecuencia, estos modelos pueden aprender ruido o variaciones aleatorias en lugar de patrones generales. López de Prado (2018) recomienda complementar las redes neuronales con ingeniería de características, incorporando indicadores técnicos, variables macroeconómicas o transformaciones estadísticas que faciliten el aprendizaje y mejoren la capacidad de generalización del modelo.

En general, los modelos estadísticos clásicos destacan por su interpretabilidad y por requerir una menor cantidad de datos para su entrenamiento. Sin embargo, presentan limitaciones para capturar relaciones no lineales complejas presentes en los mercados financieros. Por otro lado, los métodos de aprendizaje automático ofrecen una mayor capacidad de modelado y pueden detectar patrones más sofisticados, aunque requieren más datos, mayor capacidad computacional y presentan una interpretabilidad más reducida.

## **2.4  De la predicción al rendimiento financiero**

La construcción de un modelo predictivo es solo el primer paso hacia un sistema de inversión algorítmica operativo. Entre la capacidad estadística de un clasificador y su rentabilidad financiera real existe una brecha conceptual y práctica que conviene entender antes de interpretar los resultados experimentales.

**Backtesting y evaluación fuera de muestra.** El *backtesting* es el proceso mediante el cual se simula el comportamiento histórico de una estrategia de inversión utilizando datos pasados. Su objetivo es estimar cómo habría funcionado la estrategia si se hubiera ejecutado en tiempo real. En el contexto de modelos de aprendizaje automático, la evaluación fuera de muestra mediante una ventana temporal expansiva (o deslizante) emula el proceso de decisión real: el modelo se entrena exclusivamente con información disponible hasta el momento de la predicción y se evalúa sobre un período posterior. En este trabajo, cada ventana de evaluación comprende aproximadamente 63 sesiones bursátiles, equivalentes a un trimestre de mercado. Esta granularidad trimestral permite acumular un número suficiente de observaciones para obtener estimaciones estables de las métricas de clasificación mientras se mantiene una frecuencia de actualización razonable para el sistema.

**De la clasificación al trading.** Una vez que el modelo genera predicciones binarias sobre la dirección de cada activo, es necesario traducirlas en operaciones concretas. En este trabajo se adopta una estrategia *long-only*: cuando el modelo predice una subida (señal = 1), se abre una posición larga; cuando predice una bajada (señal = 0), no se mantiene posición en ese activo. Las posiciones se ponderan de forma uniforme entre todos los activos con señal activa en cada sesión, y se reequilibran diariamente.

**Costes de transacción.** Los mercados financieros reales implican fricciones que no están presentes en la evaluación estadística. Las comisiones de intermediación y la horquilla entre precio de compra y venta suponen un coste por operación que, en estrategias de alta rotación como la predicción diaria, puede llegar a superar el exceso de rentabilidad generado por el modelo. En este trabajo se aplica un coste fijo de 10 puntos básicos por lado (20 puntos básicos por operación de ida y vuelta), valor representativo del mercado español para carteras de tamaño moderado.

**Métricas financieras.** Para evaluar el comportamiento de la estrategia de trading se utilizan cuatro métricas principales. La *rentabilidad bruta* mide el beneficio acumulado antes de considerar costes de transacción. La *rentabilidad neta* refleja el beneficio real una vez descontados dichos costes. La *pérdida máxima* (*maximum drawdown*) cuantifica la mayor caída experimentada desde un máximo hasta un mínimo consecutivo, indicando el riesgo de pérdida en el peor escenario histórico. El *ratio de Sharpe* (Sharpe, 1966) relaciona el exceso de rentabilidad diaria sobre el activo sin riesgo con la volatilidad de los retornos, proporcionando una medida de rentabilidad ajustada al riesgo.

**Estrategias de referencia.** Los resultados del modelo deben contextualizarse frente a estrategias pasivas o naïve que no requieren capacidad predictiva. Se consideran tres referencias: (i) *comprar y mantener* el IBEX 35, que actúa como benchmark del comportamiento del mercado durante el período; (ii) un *clasificador aleatorio*, que genera señales de compra o venta de forma aleatoria, cuya distribución de resultados se obtiene mediante 1.000 simulaciones bootstrap; y (iii) una estrategia de *momentum* ingenuo, que predice que la dirección futura del activo coincidirá con la tendencia observada durante los últimos cinco días.

**La brecha estadística-financiera.** Una precisión equilibrada estadísticamente significativa no garantiza rentabilidad financiera. Por un lado, la magnitud del efecto predictivo en mercados eficientes es intrínsecamente pequeña —diferencias de pocas décimas de punto porcentual sobre el 50%—, lo que deja poco margen antes de que los costes la anulen. Por otro, la señal puede provenir de información que el mercado ya ha procesado parcialmente antes de que se ejecuten las órdenes, reduciendo su explotabilidad real. Esta distinción entre significancia estadística y viabilidad económica es fundamental para interpretar los resultados del Capítulo 4.

---

**Capítulo 3**

# **Metodología y desarrollo del sistema**

Este capítulo describe el sistema desarrollado, desde su arquitectura general hasta los detalles de implementación de cada componente. La estructura sigue el flujo natural del sistema: primero se presenta la arquitectura de conjunto, luego las fuentes de datos y su preprocesamiento, a continuación los modelos y los servicios automatizados que consumen sus predicciones, y finalmente el protocolo experimental empleado para evaluarlos.

## **3.1  Arquitectura general del sistema**

La Figura 3 muestra el diagrama de arquitectura del sistema completo. El pipeline parte de dos fuentes de datos externas: los datos de mercado OHLCV (*open, high, low, close, volume*) de los activos del IBEX 35 y los índices de referencia, y un flujo continuo de noticias financieras procedentes de la prensa económica española. Ambas fuentes se almacenan en una base de datos centralizada en la nube (Supabase) y en una réplica local SQLite para entrenamiento y análisis.

A partir de los datos de mercado se genera el conjunto de características que alimenta los modelos predictivos. Los modelos, entrenados periódicamente sobre el histórico disponible, producen predicciones diarias de dirección y probabilidad para cada activo. Estas predicciones se almacenan en la base de datos y son consumidas por tres servicios: una página web de visualización, un boletín informativo automatizado y un bot de trading. Las noticias, por su parte, se procesan mediante un módulo de inteligencia artificial generativa para extraer metadatos de relevancia y sentimiento, y se integran en la página web y el boletín, pero no forman parte del entrenamiento de los modelos predictivos actuales.

Todas las etapas del pipeline —actualización de datos, generación de predicciones, procesamiento de noticias y envío del boletín— se ejecutan como tareas automatizadas mediante GitHub Actions (GitHub Inc., 2026), de forma desacoplada e independiente entre sí.

*Figura 3. Diagrama de arquitectura del sistema automatizado.*

![][image3]

## **3.2  Datos y fuentes de información**

### **3.2.1  Datos financieros**

Se recopilaron series temporales diarias de precios ajustados (*adjusted close*) y volumen para los activos del IBEX 35 y para los índices de referencia IBEX 35, S&P 500 y VIX. Los precios ajustados incorporan correcciones por dividendos y ampliaciones de capital, de modo que los retornos calculados sobre ellos reflejan fielmente el rendimiento económico total del activo. Todos los datos fueron obtenidos mediante la librería yfinance (Ranaroussi, 2023), que actúa como interfaz hacia Yahoo Finance.

Las compañías incluidas corresponden a los integrantes del IBEX 35 durante el período de estudio: ACS, Acerinox, Aena, Amadeus IT Group, Acciona, Acciona Energía, BBVA, Bankinter, CaixaBank, Cellnex Telecom, Colonial, Endesa, Enagás, Fluidra, Ferrovial, Grifols, IAG, Iberdrola, Inditex, Logista, Mapfre, Merlin Properties, ArcelorMittal, Naturgy, Puig, Redeia, Banco Sabadell, Banco Santander, Telefónica y Unicaja Banco. El universo fijado presenta un potencial sesgo de supervivencia, ya que incluye únicamente empresas que han permanecido en el índice a lo largo del período; compañías que fueron excluidas o entraron en concurso de acreedores no están representadas. Este sesgo tiende a inflar ligeramente las métricas de rendimiento en datos históricos.

El conjunto de datos contiene 129.910 registros comprendidos entre enero de 2006 y abril de 2026. La fecha de inicio no es arbitraria: antes de 2006 existen inconsistencias en determinados activos, como fechas faltantes, volúmenes nulos o precios erróneos. Durante el preprocesamiento se eliminaron sesiones con volumen nulo o negativo para evitar que registros correspondientes a días de mercado cerrado o errores de datos contaminen el conjunto de entrenamiento.

La alineación de calendarios bursátiles constituye un aspecto técnico relevante: el mercado español y el estadounidense no comparten exactamente los mismos días festivos. Cuando el mercado americano permanece cerrado en una fecha en que el español sí opera, se aplica un relleno hacia adelante (*forward fill*) sobre las variables procedentes del S&P 500 y el VIX; cuando la ausencia de movimiento es estructural, el retorno se sustituye por cero. De este modo se evitan discontinuidades artificiales en las series de variables de contexto de mercado.

### **3.2.2  Noticias financieras**

Adicionalmente, se recopiló un conjunto de noticias financieras procedentes del diario económico Expansión, concretamente de las secciones de economía, mercados, empresas y ahorro, abarcando el período desde enero de 2026. Para cada noticia se almacenan los campos originales —título, cuerpo, sección, URL y fecha de publicación— así como atributos derivados generados automáticamente: categoría temática, relevancia estimada en [0,1], sentimiento (positivo, negativo o neutro) y compañías mencionadas.

Es importante destacar que las noticias no forman parte del entrenamiento de los modelos predictivos actuales. Su procesamiento y almacenamiento se orientan a futuras extensiones del sistema relacionadas con análisis de sentimiento y modelado multimodal, y a su integración en los servicios de cara al usuario —página web y boletín informativo—.

El sistema también almacena información del servicio de boletín, incluyendo la dirección de correo electrónico y la fecha de suscripción de cada usuario, para gestionar el envío automatizado de informes.

## **3.3  Preparación de datos y definición de la señal**

### **3.3.1  Definición de la señal**

La variable objetivo se construye a partir del retorno logarítmico futuro del activo. Para cada día $t$ se calcula el retorno logarítmico entre el precio de cierre en $t$ y el precio de cierre en $t+h$, donde $h$ es el horizonte temporal:

$$r_{t,h} = \ln\left(\frac{P_{t+h}}{P_t}\right)$$

donde $P_t$ es el precio de cierre ajustado en el instante $t$. A partir de este valor, la variable objetivo se define de forma binaria:

$$y_t = \begin{cases} 1 & \text{si } r_{t,h} > 0 \\ 0 & \text{en caso contrario} \end{cases}$$

Una señal igual a 1 indica que el precio futuro es superior al precio actual (subida). Los horizontes considerados son $h = 1$ día y $h = 5$ días. El horizonte de un día evalúa la capacidad del modelo para anticipar la dirección inmediata del mercado; el de cinco días busca modelar tendencias ligeramente más estables y menos sensibles al ruido diario, aproximándose al comportamiento semanal del activo.

### **3.3.2  Exploración de datos**

La predicción de series financieras constituye un problema especialmente complejo debido a la propia naturaleza de los retornos bursátiles. En particular, los retornos logarítmicos suelen presentar un comportamiento próximo al de un proceso de ruido blanco, caracterizado por una elevada aleatoriedad y una baja dependencia temporal entre observaciones consecutivas. Esto implica que los movimientos pasados del mercado contienen una cantidad limitada de información útil para anticipar movimientos futuros (Fama, 1970).

La Figura 1 muestra la evolución temporal de los retornos logarítmicos de un activo representativo del IBEX 35. Puede observarse que la serie presenta oscilaciones rápidas y aparentemente irregulares, sin patrones visuales persistentes ni tendencias fácilmente identificables, lo que evidencia el elevado nivel de ruido presente en la señal financiera.

El análisis de autocorrelación representado en la Figura 2 refuerza esta idea. Los coeficientes de autocorrelación para distintos retardos se mantienen próximos a cero, lo que indica una débil relación lineal entre los retornos actuales y los retornos pasados, en línea con los resultados de Lo y MacKinlay (1988).

![][image1]![][image2]  
*Figura 1. Evolución temporal de los retornos logarítmicos diarios de un activo representativo del IBEX 35.*  
*Figura 2. Función de autocorrelación (ACF) de los retornos logarítmicos diarios.*

Estas características suponen un desafío importante para los modelos de aprendizaje automático. En problemas donde la señal contiene una alta proporción de ruido y una estructura temporal limitada, la capacidad predictiva de los modelos tiende a reducirse considerablemente (López de Prado, 2018).

### **3.3.3  Feature engineering**

Con el objetivo de mejorar la capacidad predictiva de los modelos y facilitar el aprendizaje de patrones temporales consistentes, se llevó a cabo un proceso de ingeniería de características (*feature engineering*) sobre las series financieras originales. En lugar de utilizar precios absolutos o volúmenes sin procesar, se emplearon retornos logarítmicos, ratios y transformaciones normalizadas, evitando magnitudes absolutas cuya escala cambia con el tiempo y puede introducir artefactos en los modelos basados en árboles de decisión (López de Prado, 2018).

Todas las características se construyeron respetando estrictamente la causalidad temporal: cada ventana de datos utiliza únicamente información disponible hasta el instante actual, sin incorporar datos futuros en ningún cálculo. Se eliminaron también identificadores explícitos como la fecha exacta o el nombre de la compañía, para evitar que el modelo aprenda patrones específicos de empresa o dependencias temporales espurias.

Las variables generadas se agrupan en seis categorías:

**Retornos.** Retorno logarítmico en horizontes de 1, 5, 10 y 20 días; retorno intradía (diferencia cierre-apertura normalizada); y hueco (*gap*) entre la apertura del día y el cierre del día anterior.

**Volatilidad y estructura de vela.** Desviación estándar de retornos en ventana de 5 días; ratio de volatilidades corto/largo plazo (5/20); *Average True Range* (ATR, 14 días); y variables de vela japonesa: tamaño del cuerpo, mecha superior e inferior.

**Volumen y liquidez.** Ratios de volumen reciente frente a volumen promedio (1/5 y 1/20 días); pendiente del *On-Balance Volume* (OBV, 10 días); e iliquidez de Amihud (2002) en ventana de 10 días.

**Indicadores técnicos.** Ratio de medias móviles (5/20 y 10/50); MACD (12, 26, 9); posición relativa en las Bandas de Bollinger (20 días); pendiente de regresión lineal del precio (10 días); distancia relativa al máximo (10 y 20 días); distancia relativa al mínimo (10 y 20 días); RSI (14 días); y autocorrelación de retornos en ventana de 10 días.

**Variables de mercado.** Amplitud del mercado IBEX 35 en 1 y 10 días (fracción de valores con retorno positivo); retorno relativo del activo frente al índice (5 y 20 días); volatilidad del IBEX 35 (10 días) y su ratio corto/largo plazo (10/60); volatilidad relativa del activo frente al índice (20 días); volatilidad del S&P 500 (20 días) y su ratio (20/100); variación del VIX en 5 días; y percentil del VIX sobre los últimos 250 días.

**Variables temporales.** Día de la semana, representado mediante codificación categórica en modelos de árbol y mediante codificación sinusoidal en las redes neuronales recurrentes para preservar la naturaleza cíclica del calendario.

La Tabla 1 recoge el listado completo de características con su definición.

*Tabla 1. Características generadas mediante feature engineering. Los números entre paréntesis indican el tamaño de la ventana temporal empleada.*

| Característica | Explicación breve | Característica | Explicación breve |
| :---- | :---- | :---- | :---- |
| Logaritmo retorno (1;5;10;20) | Retorno logarítmico en distintos horizontes temporales | Retorno intradía (1) | Diferencia entre precio de cierre y apertura normalizada |
| Volatilidad (5) | Desviación estándar de retornos en ventana de 5 días | Cuerpo | Tamaño absoluto del cuerpo de la vela japonesa |
| Ratio volatilidad (5/20) | Cociente entre volatilidades de corto y largo plazo | Mecha superior | Proporción de la mecha superior respecto al rango total |
| Rango verdadero medio (14) | Average True Range; medida de volatilidad intradía | Mecha inferior | Proporción de la mecha inferior respecto al rango total |
| Ratio media móvil (5/20;10/50) | Cociente entre medias móviles de distintos periodos | Hueco | Diferencia entre apertura del día y cierre del día anterior |
| MACD (12,26,9) | Diferencia entre medias exponenciales de distinta longitud | Día de la semana | Variable cíclica (codificación sinusoidal en RNN) |
| Bandas de Bollinger (20) | Distancia del precio respecto a la banda superior e inferior | Amplitud del mercado (1;10) | Proporción de valores del IBEX 35 con retorno positivo |
| Pendiente (10) | Pendiente de la regresión lineal del precio sobre una ventana de 10 días | Retorno relativo IBEX (5;20) | Diferencia entre retorno del activo y retorno del índice |
| Distancia al máximo (10;20) | Distancia relativa al máximo de la ventana | Volatilidad IBEX (10) | Desviación estándar de retornos del IBEX 35 |
| Distancia al mínimo (10;20) | Distancia relativa al mínimo de la ventana | Ratio volatilidad IBEX (10/60) | Cociente de volatilidades del IBEX 35 en distintos horizontes |
| RSI (14) | Índice de fuerza relativa en ventana de 14 días | Volatilidad relativa IBEX (20) | Cociente entre volatilidad del activo y la del índice |
| Ratio volumen (1/5;1/20) | Cociente entre el volumen reciente y el volumen promedio | Volatilidad S&P (20) | Desviación estándar de retornos del S&P 500 |
| Pendiente del OBV (10) | Tendencia del volumen on-balance en una ventana de 10 días | Ratio volatilidad S&P (20;100) | Cociente entre volatilidades del S&P 500 en distintos horizontes |
| Iliquidez Amihud (10) | Ratio de iliquidez de Amihud (2002): impacto del precio por unidad de volumen | Cambio VIX (5) | Variación del índice de volatilidad implícita en 5 días |
| Retornos autocorrelacionados (10) | Autocorrelación de los retornos en una ventana de 10 días | Percentil VIX (250) | Percentil del VIX actual respecto al histórico de los últimos 250 días |

## **3.4  Modelos predictivos**

Se evalúan cinco familias de modelos que representan distintas hipótesis sobre la estructura de los datos financieros: cadenas de Markov, Random Forest, XGBoost, LSTM y GRU. En todos los casos el problema se formula como clasificación binaria (subida/bajada) y la salida es una probabilidad $P(\text{subida} \mid X_t)$; la predicción discreta se obtiene umbralando en 0,5. La Tabla 2 resume los inputs, outputs e hiperparámetros principales de cada modelo.

*Tabla 2. Resumen de los modelos implementados: entradas, salidas e hiperparámetros principales.*

| Modelo | Entradas | Salida | Hiperparámetros principales |
| :---- | :---- | :---- | :---- |
| Markov | log_ret_1 (retorno del día anterior) | P(subida) | n_states=3 (cuantiles), order=1, α=1,0 (Laplace) |
| Random Forest | 41 características (matriz plana) | P(subida) | n_estimators=500, max_depth=5, max_features=0,3, min_samples_leaf=50 |
| XGBoost | 41 características (matriz plana) | P(subida) | n_estimators=300, learning_rate=0,05, max_depth=3, min_child_weight=100, subsample=0,7 |
| LSTM | Ventana temporal × 41 características | P(subida) | 1 capa recurrente, hidden units moderado, dropout, early stopping |
| GRU | Ventana temporal × 41 características | P(subida) | 1 capa recurrente, hidden units moderado, dropout, early stopping |

**Modelo de Markov.** El modelo discretiza el espacio de retornos en $n = 3$ estados cuantílicos (tercil inferior, medio y superior) que representan movimientos bajistas, neutros y alcistas. El estado actual queda determinado por el retorno logarítmico del día anterior (log_ret_1). A partir del conjunto de entrenamiento se estima la matriz de transición $P(\text{subida} \mid \text{estado})$ como la frecuencia empírica de movimientos alcistas en cada estado, con suavizado de Laplace ($\alpha = 1$) para gestionar estados no observados. La predicción es, por tanto, $P(\text{subida}) = P_{s_t \to 1}$, donde $s_t$ es el estado actual. El modelo emplea solo dos clases de predicción: subida ($y=1$) o bajada ($y=0$), en coherencia con la formulación binaria del resto de modelos.

**Random Forest.** Método de aprendizaje ensamblado que combina 500 árboles de decisión poco profundos (profundidad máxima 5) entrenados sobre subconjuntos aleatorios de datos y variables (30% de características por partición). La restricción explícita de la complejidad individual de cada árbol —junto con el límite de 50 observaciones mínimas por hoja— favorece la generalización y previene el ajuste excesivo a patrones espurios en datos financieros de alta varianza.

**XGBoost.** Técnica de potenciación de gradiente que construye modelos secuencialmente, corrigiendo los errores del árbol anterior. Los árboles son muy superficiales (profundidad máxima 3) para que la complejidad emerja de la acumulación de muchos estimadores débiles. Se emplea parada temprana (*early stopping*) con una partición temporal 80/20 dentro de cada ventana de entrenamiento para determinar el número óptimo de iteraciones y evitar el sobreajuste.

**LSTM y GRU.** Las redes LSTM y GRU reciben como entrada una ventana temporal de observaciones pasadas expresadas como secuencias de las 41 características. Se utilizan arquitecturas deliberadamente compactas —una única capa recurrente con capacidad de representación moderada, *dropout* y parada temprana— para limitar la flexibilidad del modelo y priorizar la estabilidad fuera de muestra sobre señales de baja relación señal-ruido. Las características de entrada se normalizan con un escalador ajustado exclusivamente sobre los datos de entrenamiento de cada ventana para evitar fugas de información.

## **3.5  Servicios automatizados del sistema**

Los modelos entrenados no solo sirven para evaluación académica: sus predicciones se integran en tres servicios de cara al usuario y en un sistema de almacenamiento centralizado. Todos estos componentes son operados por automatizaciones de GitHub Actions que se ejecutan diariamente en días laborables.

### **3.5.1  Procesamiento de noticias**

El módulo de procesamiento de noticias extrae de forma automática las publicaciones del diario Expansión mediante sus feeds RSS (secciones economía, mercados y empresas) y las enriquece con metadatos mediante un agente de inteligencia artificial generativa. El agente, basado en el modelo openai/gpt-oss-120b (OpenAI, 2025) ejecutado en la infraestructura de Groq (Groq, 2026), procesa las noticias en lotes de aproximadamente diez artículos por llamada para equilibrar eficiencia y calidad de las respuestas. Para cada artículo extrae: categoría temática (noticia específica de compañía, macroeconómica, sentimiento de mercado o ruido general), compañías del IBEX 35 mencionadas, sentimiento (positivo, negativo o neutro) y relevancia estimada en [0,1]. La relevancia combina la evaluación semántica del modelo de lenguaje con reglas heurísticas basadas en palabras clave de eventos corporativos relevantes (dividendos, OPAs, resultados, fusiones, cambios regulatorios), lo que hace el sistema más robusto e interpretable que un enfoque puramente basado en inteligencia artificial generativa. Los metadatos resultantes se almacenan en Supabase y se consultan desde la página web y el boletín informativo.

### **3.5.2  Página web**

La interfaz de usuario se ha desplegado como aplicación web estática en GitHub Pages, implementada en HTML, CSS y JavaScript sin dependencias de servidor. Al cargarse, el cliente JavaScript realiza dos solicitudes asíncronas: (i) lee el fichero `predictions.json` desde el repositorio, que contiene las predicciones del día generadas por el modelo RF h=1; (ii) consume los feeds RSS de Expansión a través de un proxy público para el panel de noticias en directo.

Las predicciones se muestran como tarjetas individuales ordenadas por probabilidad descendente (8 tarjetas por página, con navegación), con código de colores verde/rojo para señales alcistas/bajistas y la probabilidad $P(\text{Buy}|X_t)$ mostrada explícitamente. Cada tarjeta enlaza directamente al perfil del activo en Yahoo Finance. El panel inferior reproduce titulares del día mediante un *ticker* de desplazamiento horizontal continuo, similar a los cintillos informativos televisivos, con pausa al pasar el ratón. La web también incluye una sección de descripción del modelo y una declaración de responsabilidad que precisa que las predicciones no constituyen asesoramiento financiero.

![][image4]  
*Figura 4. Interfaz principal de la página web.*

### **3.5.3  Boletín informativo**

Cada mañana, los usuarios suscritos reciben por correo electrónico un informe generado automáticamente con las tres predicciones de mayor confianza de la sesión anterior y las diez noticias de mayor relevancia estimada del día precedente. El boletín se construye como un documento HTML con las mismas convenciones visuales que la web —tipografía, paleta de colores y estructura de tarjetas— adaptado a las restricciones de los clientes de correo. El envío se realiza mediante el servidor SMTP de Gmail y puede llegar a hasta 500 destinatarios. Las direcciones de suscripción se gestionan en la base de datos Supabase, desde la que el script de envío recupera la lista de destinatarios activos en cada ejecución.

![][image5]![][image6]  
*Figura 5. Ejemplo del boletín informativo diario.*

### **3.5.4  Almacenamiento de predicciones y bot de trading**

El bloque de predicción constituye el núcleo del pipeline automatizado. Cada día laborable, una vez que tanto el mercado español como el estadounidense han cerrado, la automatización recupera las observaciones más recientes de cada activo desde Supabase (aproximadamente 300 sesiones por activo, suficientes para calcular todas las características con ventanas amplias), genera las predicciones con los modelos entrenados, y las almacena en la base de datos en la nube y en el fichero `predictions.json` del repositorio para su consumo por la web.

El bot de trading lee diariamente las predicciones almacenadas y ejecuta las operaciones correspondientes al inicio de cada sesión bursátil mediante la API de Interactive Brokers (Interactive Brokers, 2026). Se conecta localmente al puerto habilitado por la aplicación del bróker, consulta el estado de la cuenta y envía órdenes de compra para los activos con señal alcista. Todas las posiciones reciben el mismo peso dentro de la cartera. La automatización corre como tarea programada de PowerShell entre las 9:00 y las 16:45 (hora española). Las pruebas iniciales se realizaron sobre una cuenta de simulación con 100.000 euros de capital ficticio proporcionado por el bróker.

## **3.6  Protocolo experimental**

Esta sección describe el procedimiento completo de entrenamiento y evaluación utilizado para obtener los resultados del Capítulo 4. El protocolo se diseña para simular de forma rigurosa las condiciones de producción, respetando en todo momento la causalidad temporal.

**Validación cruzada purgada con ventana expansiva.** El método clásico de validación cruzada en $K$ particiones no es adecuado para series temporales porque mezcla datos futuros con datos pasados, generando estimaciones irreales. En su lugar se emplea validación cruzada purgada (López de Prado, 2018): las particiones de entrenamiento y evaluación respetan estrictamente el orden cronológico, y se introduce un período de embargo de un día entre la última observación de entrenamiento y la primera de evaluación para reducir correlaciones entre muestras adyacentes.

La validación cruzada purgada y la evaluación con ventana expansiva son dos caras del mismo procedimiento: en cada iteración se amplía el conjunto de entrenamiento con los datos más recientes disponibles, se genera un artefacto entrenado hasta ese punto temporal, y se evalúa sobre la ventana inmediatamente siguiente. De esta forma, la validación cruzada proporciona estimaciones de rendimiento en múltiples ventanas temporales, acumulando evidencia sobre la estabilidad del modelo, mientras que la evaluación final fuera de muestra utiliza el artefacto entrenado hasta el punto temporal más reciente disponible.

**Diseño de las ventanas.** El entrenamiento comienza con una primera ventana inicial de 750 sesiones bursátiles (aproximadamente tres años de mercado). A continuación, las ventanas avanzan en pasos trimestrales de aproximadamente 63 sesiones bursátiles. Cada ventana de evaluación comprende también 63 sesiones. Este diseño produce 70 ventanas de evaluación para el horizonte de un día y 66 para el de cinco días.

**Métricas de clasificación.** La precisión equilibrada (*balanced accuracy*) actúa como métrica principal de optimización, ya que corrige el posible desbalance entre clases y proporciona una medida equilibrada del rendimiento global. El coeficiente de correlación de Matthews (MCC) complementa este análisis al considerar los cuatro elementos de la matriz de confusión, siendo robusto ante desbalances. El ROC-AUC evalúa la capacidad discriminativa del modelo a través de todos los umbrales de decisión posibles.

**Contraste de significación.** Para evaluar si la precisión equilibrada media de cada modelo es estadísticamente superior al azar, se aplica una prueba $t$ de una muestra con hipótesis nula $H_0: \mu = 0{,}5$ e hipótesis alternativa $H_a: \mu > 0{,}5$, utilizando como observaciones las puntuaciones obtenidas en las distintas ventanas de evaluación temporal. Este contraste mide si las predicciones son mejores que el azar, pero no permite comparar directamente un modelo frente a otro; para ello se emplean contrastes pareados entre modelos (Sección 4.1).

**Métricas financieras y protocolo de trading.** La simulación financiera fuera de muestra se ejecuta sobre el período del 1 de mayo al 1 de junio de 2026. La cartera parte de 100.000 euros con las siguientes reglas: (i) estrategia *long-only*: se abren posiciones largas en todos los activos con predicción de subida (señal = 1) y no se toman posiciones en activos con predicción de bajada (señal = 0); (ii) la señal se genera al cierre del día $t$ con datos disponibles hasta ese momento; (iii) la entrada se ejecuta a la apertura del día $t+1$ y la salida al cierre de $t+1$ para $h=1$; (iv) las posiciones se ponderan de forma uniforme entre todos los activos activos y se reequilibran diariamente; (v) se aplica un coste de transacción de 10 puntos básicos por lado (20 puntos básicos por operación de ida y vuelta). El clasificador aleatorio se promedia sobre 1.000 simulaciones bootstrap para obtener estimaciones estables de sus métricas.

---

**Capítulo 4**

# **Resultados experimentales**

Este capítulo presenta los resultados obtenidos en la evaluación experimental. Se organiza en cuatro secciones: resultados de clasificación en validación cruzada expandida y contrastes estadísticos entre modelos, análisis de importancia de variables, resultados de la simulación financiera, y una discusión integradora que relaciona ambas dimensiones de la evaluación.

## **4.1  Resultados predictivos de los modelos**

La Tabla 3 recoge el rendimiento medio en validación cruzada expandida de los modelos evaluados para los horizontes de predicción de uno y cinco días. Los valores de ± corresponden al error estándar de la media entre ventanas de evaluación.

*Tabla 3. Métricas de los modelos en validación cruzada expandida, ordenadas de mayor a menor precisión equilibrada. H hace referencia al horizonte de predicción en días. La columna "Significativo" indica si la precisión equilibrada es estadísticamente superior al 50% según la prueba t de una muestra (α = 0,05).*

| Modelo | Precisión equilibrada (%) | ROC AUC | MCC | Significativo |
| :---- | :---- | :---- | :---- | ----- |
| Random Forest (H1) | 53,16 ± 0,39 | 0,5448 | 0,0649 | SÍ |
| GRU (H1) | 52,74 ± 0,36 | 0,5431 | 0,0564 | SÍ |
| XGBoost (H1) | 52,55 ± 0,39 | 0,5381 | 0,0541 | SÍ |
| LSTM (H1) | 52,34 ± 0,31 | 0,5383 | 0,0486 | SÍ |
| Random Forest (H5) | 51,76 ± 0,57 | 0,5301 | 0,0399 | SÍ |
| LSTM (H5) | 51,68 ± 0,45 | 0,5245 | 0,0371 | SÍ |
| GRU (H5) | 51,58 ± 0,44 | 0,5261 | 0,0356 | SÍ |
| XGBoost (H5) | 50,70 ± 0,43 | 0,5250 | 0,0145 | NO |
| Markov (H5) | 50,10 ± 0,15 | 0,5048 | 0,0016 | NO |
| Markov (H1) | 49,92 ± 0,11 | 0,4974 | -0,0016 | NO |

Los resultados sugieren la existencia de una señal predictiva débil pero detectable en la dirección del IBEX 35. En términos de precisión equilibrada, todos los modelos superan el umbral del azar —a excepción del modelo de Markov en ambos horizontes, que se mantiene prácticamente en el 50% (49,92% y 50,10%, respectivamente)—. La prueba $t$ de una muestra confirma que las diferencias del modelo de Markov no son estadísticamente significativas, lo que indica que la información contenida en los estados discretos del retorno pasado no es suficiente para predecir de forma sistemática la dirección futura del mercado.

En el horizonte de un día, los cuatro modelos restantes muestran una precisión equilibrada significativamente superior al azar (p < 0,001 en todos los casos), con valores superiores al 52%. Aunque las diferencias absolutas respecto al 50% son reducidas, Random Forest alcanza el mejor desempeño (53,16%), seguido de GRU y XGBoost. Este resultado es coherente con el benchmark de la literatura, que sitúa el rendimiento de modelos de aprendizaje automático en predicción diaria de renta variable entre el 52% y el 55% de precisión equilibrada (Krauss et al., 2017). El ROC-AUC y el MCC refuerzan este patrón: los valores de MCC, siempre inferiores a 0,07, confirman que la magnitud del efecto es reducida en términos absolutos, consistente con la hipótesis del mercado eficiente en su forma débil (Fama, 1970).

Para el horizonte de cinco días, el rendimiento general se deteriora moderadamente. Random Forest, LSTM y GRU mantienen su significación estadística, si bien con mayor variabilidad entre ventanas. XGBoost no puede rechazar la hipótesis nula a nivel α = 0,05 en el horizonte de cinco días, lo que sugiere que su ventaja predictiva es predominantemente de corto plazo.

**Comparación pareada entre modelos.** El contraste de una muestra frente a 0,5 permite evaluar si cada modelo supera el azar, pero no si un modelo es significativamente mejor que otro. Para abordar esta cuestión se realizaron contrastes $t$ pareados entre todos los pares de modelos, utilizando las puntuaciones por ventana de validación como observaciones. Las Tablas 4 y 5 resumen los resultados en forma de matrices de victorias para los horizontes H1 y H5, respectivamente, donde "Mejor" indica que el modelo de fila supera significativamente al modelo de columna (p < 0,05, bilateral), y "Ns" indica ausencia de diferencia significativa.

*Tabla 4. Matriz de comparaciones pareadas (prueba t bilateral) para H=1. "Mejor": el modelo fila supera significativamente al modelo columna; "Ns": diferencia no significativa.*

|  | Markov (H1) | RF (H1) | XGB (H1) | GRU (H1) | LSTM (H1) |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Markov (H1) | — | Peor | Peor | Peor | Peor |
| RF (H1) | Mejor | — | Mejor | Ns | Mejor |
| XGB (H1) | Mejor | Peor | — | Ns | Ns |
| GRU (H1) | Mejor | Ns | Ns | — | Ns |
| LSTM (H1) | Mejor | Peor | Ns | Ns | — |

*Tabla 5. Matriz de comparaciones pareadas (prueba t bilateral) para H=5.*

|  | Markov (H5) | RF (H5) | XGB (H5) | GRU (H5) | LSTM (H5) |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Markov (H5) | — | Peor | Ns | Peor | Peor |
| RF (H5) | Mejor | — | Mejor | Ns | Ns |
| XGB (H5) | Ns | Peor | — | Ns | Peor |
| GRU (H5) | Mejor | Ns | Ns | — | Ns |
| LSTM (H5) | Mejor | Mejor | Ns | Ns | — |

En H1, Random Forest se distingue de forma significativa de Markov, XGBoost y LSTM, mientras que su diferencia con GRU no alcanza significación estadística. En H5, Random Forest supera significativamente a Markov y XGBoost, y no muestra diferencias significativas frente a GRU ni LSTM. La lectura de estas matrices confirma que no existe un modelo claramente dominante: la mayoría de las diferencias entre los modelos de aprendizaje automático no son estadísticamente significativas, especialmente en H5.

Para la implementación en producción se optó por Random Forest tanto en H1 como en H5, dado que presenta la mayor precisión equilibrada media en ambos horizontes (53,16% y 51,76%, respectivamente) y, como se muestra en la Sección 4.3, también produce los mejores resultados financieros fuera de muestra.

## **4.2  Importancia de variables**

La importancia de variables se calculó a partir del atributo `feature_importances_` de los modelos Random Forest y XGBoost, que mide la reducción media de impureza de Gini ponderada por el número de muestras que pasan por cada nodo en todos los árboles del conjunto. Este método proporciona una estimación de la contribución relativa de cada característica al rendimiento predictivo del modelo de árbol, y se puede calcular directamente a partir de la estructura interna del modelo sin necesidad de técnicas de perturbación externas.

La Figura 6 muestra la importancia relativa de las características en los modelos de árbol para los horizontes H1 y H5.

![][image7]  
*Figura 6. Importancia relativa de las características en los modelos de árbol (Random Forest y XGBoost) para los horizontes H1 y H5.*

El patrón es consistente en ambos horizontes: las variables más influyentes son la variación estandarizada del VIX estadounidense y métricas de volatilidad y actividad del S&P 500, seguidas de indicadores de amplitud del mercado IBEX 35. Este resultado indica que el estado del mercado americano —su nivel de tensión y actividad— constituye la señal más informativa para predecir la dirección del IBEX 35. La preponderancia de estas señales sobre los indicadores técnicos individuales de cada acción sugiere que la dirección del mercado español está fuertemente condicionada por el contexto macroeconómico global, y que los modelos han aprendido a explotar esta dependencia estructural.

Cabe destacar que la importancia de Gini es específica de los modelos de árbol y no puede extrapolarse directamente a las arquitecturas recurrentes LSTM y GRU, cuya opacidad interna impide este tipo de análisis sin recurrir a técnicas de explicabilidad adicionales como SHAP o permutation importance.

## **4.3  Resultados de la simulación financiera**

La Tabla 6 muestra los resultados obtenidos en la simulación de trading fuera de muestra (1 de mayo – 1 de junio de 2026) para las distintas estrategias. Las reglas de operativa son las descritas en la Sección 3.6: estrategia *long-only*, posición abierta a la apertura de $t+1$ y cerrada al cierre de $t+1$, ponderación uniforme, coste de 10 puntos básicos por lado, y clasificador aleatorio promediado sobre 1.000 simulaciones bootstrap.

*Tabla 6. Resultados de la simulación de trading para distintas estrategias durante el período de prueba (1 de mayo – 1 de junio de 2026). Los valores de rentabilidad son acumulados en el período.*

| Estrategia | Rentabilidad bruta (%) | Rentabilidad neta (%) | Pérdida máxima (%) | Ratio Sharpe |
| :---- | ----- | ----- | ----- | ----- |
| RF (H5) | +12,42% | +9,12% | −5,24% | 5,566 |
| Comprar y mantener | +3,27% | +3,27% | −2,93% | 1,908 |
| Clasificador aleatorio | +3,03% | −0,81% | −3,28% | −0,970 |
| Momentum | +1,92% | −1,88% | −4,31% | −2,008 |
| RF (H1) | −0,50% | −3,26% | −4,21% | −7,793 |

Los resultados revelan una diferencia notable entre los dos horizontes del mismo modelo. La estrategia basada en Random Forest con horizonte de cinco días es la única que supera a todas las referencias, con una rentabilidad neta del +9,12% y un ratio de Sharpe de 5,566, muy por encima del *buy & hold* (+3,27%, Sharpe 1,908). La estrategia de RF H1, en cambio, presenta una rentabilidad bruta ligeramente negativa (−0,50%), que se convierte en −3,26% neta una vez aplicados los costes de transacción, con un ratio de Sharpe muy desfavorable (−7,793).

Este contraste entre H1 y H5 ilustra el papel crítico de los costes de transacción en estrategias de alta frecuencia diaria. La operativa diaria sobre una cartera diversificada de 30 activos implica un número elevado de operaciones que erosionan completamente el pequeño exceso de rentabilidad bruta del modelo H1. El horizonte de cinco días reduce sustancialmente la rotación de la cartera, manteniendo posiciones abiertas durante más tiempo y amortizando mejor los costes fijos por operación sobre un retorno acumulado mayor.

La estrategia de comprar y mantener obtiene un buen comportamiento relativo (Sharpe 1,908), coherente con el período de mercado analizado. El clasificador aleatorio, aunque con rentabilidad bruta positiva, resulta negativo en términos netos, confirmando que la operativa activa consume los retornos de mercado si no existe señal predictiva real. La estrategia de momentum presenta un rendimiento intermedio en términos brutos pero termina en negativo tras costes, lo que sugiere que la inercia de cinco días no es suficientemente robusta para compensar la fricción transaccional.

## **4.4  Discusión de los resultados**

Los resultados presentados en este capítulo permiten articular un mensaje principal: los modelos de aprendizaje automático detectan una señal predictiva débil pero estadísticamente significativa en la dirección del IBEX 35, pero esa señal solo se traduce en rentabilidad real cuando se gestiona adecuadamente la frecuencia operativa y, con ella, el impacto de los costes de transacción.

Desde la perspectiva estadística, Random Forest H1 y H5 lideran el ranking de precisión equilibrada y son significativamente superiores al modelo de Markov en ambos horizontes. Las diferencias entre los modelos de aprendizaje automático entre sí son en su mayor parte no significativas, lo que indica que en el margen de rendimiento disponible en un mercado desarrollado todas las arquitecturas convergen hacia un límite similar. Los valores de MCC, siempre inferiores a 0,07, confirman que la magnitud del efecto es reducida en términos absolutos, consistente con la hipótesis del mercado eficiente en su forma débil (Fama, 1970).

Desde la perspectiva financiera, el horizonte de predicción importa tanto como el modelo. RF H5 supera ampliamente a todas las referencias con un Sharpe de 5,566, mientras que RF H1 —a pesar de tener la mayor precisión estadística— no es explotable con costes realistas. Este resultado es consistente con la discusión conceptual de la Sección 2.4: una pequeña ventaja estadística puede no ser suficiente para superar la fricción de la operativa diaria.

El análisis de importancia de variables refuerza la hipótesis de que la señal proviene principalmente del contexto macroeconómico global —en particular del VIX y del S&P 500—. Dado que esta información se hace pública tras el cierre de Wall Street, el mercado europeo tiene tiempo de incorporarla parcialmente antes de la apertura del IBEX 35, lo que limita su explotabilidad en H1 pero deja más margen en H5, donde el modelo integra señales de varios días consecutivos.

En conjunto, el sistema desarrollado demuestra que es posible construir un pipeline de inversión algorítmica completo con herramientas de código abierto y presupuesto cero, y obtener resultados financieros positivos en una simulación fuera de muestra con el horizonte adecuado. Sin embargo, el período de evaluación es corto (un mes) y los resultados pueden estar condicionados por el régimen de mercado específico de ese período; la validación en ventanas temporales más largas es necesaria antes de extraer conclusiones generalizables.

---

**Capítulo 5**

# **Conclusiones**

Este trabajo ha abordado el problema de la predicción de dirección del mercado bursátil español mediante técnicas de aprendizaje automático, integrando los resultados en un sistema automatizado de extremo a extremo. A continuación se presentan las principales conclusiones derivadas del estudio, actualizadas para incluir los resultados de los modelos en el horizonte de cinco días.

**Sobre la capacidad predictiva de los modelos.** Los resultados obtenidos demuestran que los modelos de aprendizaje automático basados en árboles de decisión (Random Forest, XGBoost) y redes neuronales recurrentes (GRU, LSTM) poseen una capacidad predictiva estadísticamente superior al azar sobre la dirección del IBEX 35, aunque la magnitud del efecto es reducida. Random Forest es el modelo con mejor rendimiento en el horizonte de un día (precisión equilibrada 53,16%) y en el de cinco días (51,76%), resultado coherente con el rango de 52-55% reportado en la literatura para predicción diaria de renta variable (Krauss et al., 2017). El modelo de Markov, por el contrario, no supera el umbral del azar en ningún horizonte, lo que indica que la información contenida en los estados discretos del retorno pasado no es suficiente para predecir de forma sistemática la dirección futura del mercado. La mayor parte de las diferencias entre los modelos de aprendizaje automático no son estadísticamente significativas en los contrastes pareados, lo que sugiere que en el margen de rendimiento disponible en un mercado eficiente todas las arquitecturas convergen hacia un límite similar.

**Sobre el impacto del horizonte de predicción.** La extensión del análisis al horizonte de cinco días revela diferencias sustanciales tanto en la significación estadística de los modelos como, especialmente, en su viabilidad financiera. Mientras que en H1 los cuatro modelos de aprendizaje automático son estadísticamente significativos, en H5 XGBoost pierde su ventaja sobre el azar. Más relevante aún es el impacto en la simulación financiera: RF H5 genera una rentabilidad neta del +9,12% con un ratio de Sharpe de 5,566, superando ampliamente al *buy & hold* (+3,27%, Sharpe 1,908) y a todas las demás referencias. Este resultado contrasta con RF H1, cuya operativa diaria sobre una cartera diversificada acumula costes de transacción que convierten una pequeña ventaja estadística en una rentabilidad neta negativa (−3,26%). La reducción de la rotación de cartera al pasar de H1 a H5 es, por tanto, el factor determinante para la viabilidad económica del sistema.

**Sobre la naturaleza de la señal predictiva.** El análisis de importancia de variables mediante la impureza de Gini de los modelos de árbol revela que la señal predictiva proviene principalmente de variables macroeconómicas externas, en particular de la variación del VIX y de métricas de actividad del S&P 500. Este hallazgo sugiere que la dirección del mercado español está condicionada en mayor medida por el contexto macroeconómico global que por los indicadores técnicos individuales de cada acción. Sin embargo, esta dependencia implica también que la señal puede ser parcialmente anticipada por el propio mercado antes de la apertura de la sesión bursátil, lo que contribuye a limitar su explotabilidad en operativas de muy corto plazo.

**Sobre la brecha entre capacidad estadística y rentabilidad real.** La evaluación financiera evidencia con claridad la diferencia entre significancia estadística y explotabilidad económica. A pesar de que todos los modelos de aprendizaje automático presentan una precisión equilibrada estadísticamente significativa en H1, la operativa diaria genera un número elevado de transacciones cuyos costes terminan por eliminar completamente el reducido margen predictivo. Es solo en el horizonte de cinco días, donde la rotación de cartera es menor y el coste de transacción se amortiza sobre retornos acumulados mayores, donde el sistema demuestra ser financieramente rentable.

**Sobre el sistema desarrollado.** Más allá de los resultados predictivos, el trabajo demuestra que es posible construir un sistema de análisis financiero automatizado completo —datos, modelos, predicciones, visualización, boletín y bot de trading— utilizando exclusivamente herramientas gratuitas y de código abierto. La arquitectura desacoplada basada en GitHub Actions garantiza la reproducibilidad y facilita la incorporación de nuevos componentes o mejoras futuras.

En definitiva, este trabajo confirma que los mercados financieros desarrollados presentan ineficiencias débiles y estadísticamente detectables mediante modelos de aprendizaje automático, pero que la traducción de estas señales en rentabilidad real depende críticamente del horizonte temporal y del control de los costes de transacción. El valor del sistema desarrollado reside no solo en los resultados obtenidos, sino en la demostración de que un pipeline de inversión algorítmica completo puede construirse con recursos accesibles y metodología rigurosa.

---

**Capítulo 6**

# **Trabajo futuro**

En una línea de trabajo futura, uno de los aspectos más relevantes consiste en ampliar de forma significativa la base de datos utilizada, incorporando un mayor volumen de información histórica de noticias. El incremento del horizonte temporal, abarcando períodos de varios años, permitiría capturar mejor la evolución de los patrones informativos y su relación con los movimientos del mercado. En este contexto, la integración de noticias en el proceso de predicción adquiere un papel especialmente relevante, ya que posibilitaría incorporar información cualitativa que complementa los datos estrictamente financieros.

En cuanto al tratamiento de la información textual, en lugar de limitarse a una clasificación simplificada del sentimiento asociado a las noticias, se plantea la posibilidad de utilizar representaciones vectoriales de los textos. Este enfoque permite capturar matices semánticos más ricos y relaciones más complejas entre documentos, lo que podría traducirse en una mejora en la calidad de las señales generadas.

Asimismo, resulta pertinente ampliar el conjunto de variables explicativas mediante la inclusión de indicadores macroeconómicos adicionales, como la tasa de empleo, la inflación u otras métricas de actividad económica. Este tipo de variables pueden aportar una visión más global del entorno financiero, mejorando la capacidad del modelo para adaptarse a diferentes regímenes de mercado.

Por otro lado, se abre la puerta a explorar nuevas fuentes de información o combinaciones de indicadores menos convencionales que puedan identificar señales con mayor capacidad explicativa y potencial predictivo. Muchas de las variables utilizadas en este trabajo son indicadores técnicos clásicos ampliamente seguidos por el mercado, lo que puede limitar su capacidad de generar ventaja si son anticipados por otros participantes.

Desde una perspectiva de gestión del riesgo y la operativa, sería interesante incorporar reglas de gestión dinámica del tamaño de las posiciones en función de la volatilidad, así como definir zonas de entrada y salida del mercado basadas en la confianza del modelo. Estos enfoques, propios de la gestión de carteras, pueden contribuir tanto a mejorar la rentabilidad ajustada al riesgo como a dotar al sistema de mayor robustez en distintos regímenes de mercado.

Finalmente, la calibración de probabilidades del modelo —mediante técnicas como la escala de Platt o la regresión isotónica— podría mejorar la utilidad de las probabilidades generadas para el dimensionamiento de posiciones, ya que actualmente estas probabilidades no pueden interpretarse en sentido estadístico estricto.

---

# **Bibliografía**

Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. Journal of Financial Markets, 5(1), 31-56. https://doi.org/10.1016/S1386-4181(01)00024-6

Bloomberg L.P. (s. f.). Bloomberg Terminal. Bloomberg Professional Services. Recuperado el 15 de mayo de 2026, de https://professional.bloomberg.com/products/bloomberg-terminal/

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., y Ljung, G. M. (2015). Time series analysis: Forecasting and control (5.ª ed.). John Wiley & Sons.

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

Chen, T., y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16) (pp. 785-794). Association for Computing Machinery. https://doi.org/10.1145/2939672.2939785

Cho, K., van Merriënboer, B., Gülçehre, Ç., Bahdanau, D., Bougares, F., Schwenk, H., y Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. En Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 1724-1734). Association for Computational Linguistics. https://aclanthology.org/D14-1179/

Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. The Journal of Finance, 25(2), 383-417. https://doi.org/10.1111/j.1540-6261.1970.tb00518.x

Fischer, T., y Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654-669. https://doi.org/10.1016/j.ejor.2017.11.054

GitHub. (s. f.). GitHub Actions documentation. Recuperado el 15 de mayo de 2026, de https://docs.github.com/actions

Groq. (s. f.). GroqCloud documentation. Recuperado el 15 de mayo de 2026, de https://console.groq.com/docs/overview

Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, 57(2), 357-384. https://doi.org/10.2307/1912559

Hochreiter, S., y Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735

Interactive Brokers. (s. f.). IB Gateway. Recuperado el 15 de mayo de 2026, de https://www.interactivebrokers.com/en/trading/ibgateway-stable.php

Krauss, C., Do, X. A., y Huck, N. (2017). Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500. European Journal of Operational Research, 259(2), 689-702. https://doi.org/10.1016/j.ejor.2016.10.031

Lo, A. W., y MacKinlay, A. C. (1988). Stock market prices do not follow random walks: Evidence from a simple specification test. The Review of Financial Studies, 1(1), 41-66. https://doi.org/10.1093/rfs/1.1.41

López de Prado, M. (2018). Advances in financial machine learning. John Wiley & Sons.

OpenAI. (2025, 5 de agosto). gpt-oss-120b & gpt-oss-20b model card. OpenAI. https://openai.com/index/gpt-oss-model-card/

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., VanderPlas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., y Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830. https://www.jmlr.org/papers/v12/pedregosa11a.html

Aroussi, R. (s. f.). yfinance: Download market data from Yahoo! Finance's API. GitHub. Recuperado el 15 de mayo de 2026, de https://github.com/ranaroussi/yfinance

Sharpe, W. F. (1966). Mutual fund performance. The Journal of Business, 39(1, Part 2), 119-138. https://doi.org/10.1086/294846

Supabase Inc. (s. f.). Supabase documentation. Recuperado el 15 de mayo de 2026, de https://supabase.com/docs

TradingView. (s. f.). TradingView. Recuperado el 15 de mayo de 2026, de https://www.tradingview.com

Yahoo Finance. (s. f.). Yahoo Finance. Recuperado el 15 de mayo de 2026, de https://finance.yahoo.com

Welch, B. L. (1947). The generalization of 'Student's' problem when several different population variances are involved. Biometrika, 34(1/2), 28-35. https://doi.org/10.2307/2332510

---

# **Apéndice**
