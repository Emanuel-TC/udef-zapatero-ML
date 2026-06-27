# Análisis Forense de Redes de Influencia mediante Machine Learning (Caso Práctico: UDEF-zapatero-ML)

Este repositorio contiene el código fuente, la metodología y los resultados del análisis de datos aplicado a un sumario policial (UDEF) sobre una red de influencia financiera. El proyecto demuestra la aplicación de técnicas avanzadas de **Data Science, Teoría de Grafos y Aprendizaje No Supervisado (Unsupervised Machine Learning)** para extraer inteligencia estructurada a partir de texto libre.

## 📌 Contexto del Proyecto

El objetivo principal es auditar, estructurar y analizar las comunicaciones extraídas de dispositivos móviles en un contexto de investigación financiera. Para evitar el sesgo cognitivo humano en la lectura de los mensajes, se han diseñado modelos puramente matemáticos que perfilan a los actores y detectan anomalías de comportamiento basándose en la topología de la red y los metadatos temporales.

## 🏗 Arquitectura y Pipeline de Datos

El proyecto sigue una estructura modular orientada a objetos (OOP), separando la extracción de datos, el preprocesamiento, el modelado y la visualización.

### Fase 1: Ingesta Híbrida y Resolución de Entidades
* **Extracción Híbrida (RegEx + LLM Groq):** Para superar las limitaciones del OCR y el formato desestructurado del PDF policial, se implementó un pipeline que combina Expresiones Regulares para volcados limpios y un modelo LLM (Groq) como mecanismo de rescate o *fallback* para páginas corruptas.
* **Entity Resolution (NLP con spaCy):** Se desarrolló un módulo (`src/entities.py`) para desambiguar alias y unificar identidades en la red, evitando la fragmentación de nodos.

### Fase 2: Casos de Uso (Machine Learning)
Se han desarrollado dos casos de uso analíticos independientes:

1.  **Machine Learning sobre Grafos (Graph ML):** * **Metodología:** Construcción de un grafo heterogéneo (comunicaciones directas + menciones NLP).
    * **Modelado:** Extracción de características topológicas (Betweenness, PageRank, Degree Centrality) y segmentación de la red mediante un algoritmo **K-Means**.
    * **Objetivo:** Identificar roles estructurales de forma no supervisada (Líderes, Brokers de Información, Peones).
2.  **Detección de Anomalías (Temporal Anomaly Detection):**
    * **Metodología:** Feature Engineering sobre metadatos temporales (Deltas de tiempo con suavizado logarítmico, longitud de mensaje, horas del día).
    * **Modelado:** Aplicación de **Isolation Forest** para aislar ventanas de tiempo atípicas.
    * **Objetivo:** Detectar picos de crisis, coordinación clandestina y frenesí comunicativo sin evaluar el texto o el sentimiento de los mensajes (evitando el sesgo de lenguaje encriptado).

## 📂 Estructura del Repositorio

```bash
├── data/
│   ├── raw/                  # PDF original y datos en crudo
│   └── processed/            # Datasets intermedios y finales (Grafos, Features, etc.)
├── notebooks/
│   ├── 01_data_extraction_and_eda.ipynb  # Pipeline de Ingesta y Limpieza Quirúrgica
│   ├── 02_case_of_use_1.ipynb            # Graph ML y Clustering de Roles (K-Means)
│   └── 03_case_of_use_2.ipynb            # Detección de Anomalías Temporales (Isolation Forest)
├── src/
│   ├── parser.py             # Módulo de extracción híbrida PDF -> CSV
│   ├── entities.py           # Módulo de Resolución de Entidades (NLP)
│   ├── graphs.py             # Constructor de Grafos y Extracción de Topología
│   ├── models.py             # Clases envolventes para Modelos Predictivos
│   ├── anomalies.py          # Feature Engineering Temporal
│   └── utils.py              # Helpers y Visualizaciones (Seaborn, Matplotlib)
├── requirements.txt          # Dependencias del proyecto
├── .env                      # Variables de entorno (API Keys - NO SUBIR AL REPO)
└── README.md
```

## 🚀 Instalación y Ejecución

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/Emanuel-TC/udef-zapatero-ML
    cd udef-zapatero-ML
    ```
2.  **Crear un entorno virtual e instalar dependencias:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python -m spacy download es_core_news_md
    ```
3.  **Configurar Variables de Entorno:**
    * Crea un archivo `.env` en la raíz del proyecto.
    * Añade tu API Key de Groq: `GROQ_API_KEY=tu_api_key_aqui`
4.  **Ejecución:**
    * Ejecuta los Jupyter Notebooks en la carpeta `/notebooks` de forma secuencial (`01` -> `02` -> `03`).

## 💡 Próximos Pasos (En Desarrollo)
* Integración de una capa de explicabilidad semántica mediante LLMs para traducir los hallazgos matemáticos (Clústeres y Anomalías) a informes de inteligencia natural.

---
*Desarrollado para la Defensa del Trabajo de Fin de Máster.*