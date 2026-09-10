# Detección de fraude en transacciones con tarjeta de crédito

**Trabajo Final de Maestría — Análisis y Visualización de Datos Masivos**
Universidad Internacional de La Rioja (UNIR)

**Equipo 3_G:** José Luis Acosta Bendeck · David Juárez Vásquez

---

## Descripción

Este repositorio contiene el prototipo del proyecto de detección de fraude en
transacciones con tarjeta de crédito. Reúne el pipeline completo que va desde el
conjunto de datos crudo hasta un cuadro de mando interactivo, y permite recrear
íntegramente los resultados presentados en la memoria del TFM.

El proyecto aborda un problema de **clasificación binaria fuertemente
desbalanceada**: sobre 283.726 transacciones, solo 473 son fraudulentas (0,167 %).

## Resultados de referencia

Modelo seleccionado: **Random Forest** con ponderación de clases, evaluado sobre
las 85.118 transacciones del conjunto de prueba (142 fraudes).

| Métrica | Valor |
|---|---|
| AUPRC | 0,808 |
| AUROC | 0,969 |
| Precisión | 0,876 |
| Recall | 0,746 |
| F1 | 0,806 |
| Falsas alarmas (FP) | 15 |
| Fraudes detectados (TP) | 106 |

## Estructura del repositorio

```
TFM_UNIR/
├── src/
│   ├── train_model.py       # Pipeline: carga, limpieza, partición, entrenamiento
│   └── build_database.py    # Construye el modelo de datos en SQLite
├── app/
│   └── dashboard.py         # Cuadro de mando interactivo (Streamlit)
├── models/
│   ├── modelo_rf.pkl        # Modelo entrenado y serializado
│   ├── scaler.pkl           # Escalador robusto ajustado
│   ├── metricas.json        # Métricas de evaluación
│   └── importancia_variables.csv
├── data/
│   ├── muestra_test.csv     # Muestra del conjunto de prueba (para el dashboard)
│   └── fraude.db            # Base de datos SQLite
├── docs/                    # Capturas del prototipo
├── requirements.txt
└── README.md
```

## Instalación

```bash
git clone https://github.com/bendeck07/TFM_UNIR.git
cd TFM_UNIR
pip install -r requirements.txt
```

## Obtención de los datos

El conjunto de datos **no se incluye** en el repositorio por su tamaño (~150 MB).
Descárgalo de Kaggle y colócalo en la carpeta `data/`:

- Fuente: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Archivo esperado: `data/creditcard.csv`

## Uso

**1. Entrenar el modelo y generar los artefactos**

```bash
python src/train_model.py
```

Genera el modelo serializado, el escalador, las métricas y la muestra de prueba.

**2. Construir el modelo de datos**

```bash
python src/build_database.py
```

Crea `data/fraude.db` con las tablas `transacciones`, `predicciones`,
`alertas` y `ejecuciones`.

**3. Lanzar el cuadro de mando**

```bash
streamlit run app/dashboard.py
```

Abre el navegador en `http://localhost:8501`.

## Funcionalidades del cuadro de mando

- **Ajuste del umbral de decisión** en tiempo real, con recálculo inmediato de
  métricas, matriz de confusión y coste.
- **Análisis de coste económico** configurable: permite modificar el coste
  asignado a cada tipo de error y observar qué umbral minimiza el coste total.
- **Curva precisión-exhaustividad** con el punto de operación actual señalado.
- **Importancia de variables** del modelo.
- **Cola de alertas** priorizada por probabilidad de fraude, filtrable por
  nivel de prioridad.

## Modelo de datos

La base SQLite implementa el siguiente esquema:

- `transacciones` — datos de cada operación y su etiqueta real.
- `ejecuciones` — trazabilidad de cada ejecución del modelo (versión, umbral, métricas).
- `predicciones` — score de fraude asignado a cada transacción en cada ejecución.
- `alertas` — operaciones marcadas para revisión, con prioridad y estado de gestión.

## Metodología

El proyecto sigue el marco **CRISP-DM**. La partición es estratificada (70/30),
el escalado se ajusta exclusivamente sobre el conjunto de entrenamiento para
evitar fugas de información, y el desbalanceo se aborda mediante ponderación de
clases. La reproducibilidad se garantiza fijando la semilla aleatoria en 42.

## Nota sobre la muestra del dashboard

Para mantener el repositorio ligero, `data/muestra_test.csv` contiene los 142
fraudes del conjunto de prueba más 5.000 operaciones legítimas de las 84.976
totales. Las métricas mostradas en el dashboard sobre esta muestra son, por
tanto, más favorables que las de referencia indicadas arriba. Para reproducir
las métricas oficiales, ejecuta `src/train_model.py` con el conjunto completo.

## Licencia

Proyecto académico desarrollado en el marco del Trabajo Final de Maestría de UNIR.
El conjunto de datos original pertenece al Machine Learning Group de la
Université Libre de Bruxelles (ULB).
