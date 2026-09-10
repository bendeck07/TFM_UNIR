# -*- coding: utf-8 -*-
"""
TFM - Detección de fraude con tarjeta de crédito
Equipo 3_G — José Luis Acosta Bendeck y David Juárez Vásquez

Modelo de datos del prototipo.

Crea una base de datos SQLite con el esquema que soporta el almacenamiento
de las transacciones, las predicciones del modelo y las alertas generadas
para revisión por parte del analista de fraude.

Esquema:
    transacciones : datos de cada operación (Time, V1-V28, Amount, Class)
    predicciones  : score de fraude asignado por el modelo a cada transacción
    alertas       : operaciones marcadas para revisión, con su estado de gestión
    ejecuciones   : trazabilidad de cada ejecución del modelo (versión, umbral, métricas)

Uso:
    python src/build_database.py
"""

import os
import sqlite3
import json
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(DATA_DIR, "fraude.db")

ESQUEMA = """
DROP TABLE IF EXISTS alertas;
DROP TABLE IF EXISTS predicciones;
DROP TABLE IF EXISTS transacciones;
DROP TABLE IF EXISTS ejecuciones;

CREATE TABLE transacciones (
    id_transaccion INTEGER PRIMARY KEY AUTOINCREMENT,
    tiempo         REAL    NOT NULL,
    importe        REAL    NOT NULL,
    clase_real     INTEGER,              -- 0 legítima, 1 fraude (etiqueta conocida)
    origen         TEXT    DEFAULT 'creditcard.csv',
    fecha_carga    TEXT    NOT NULL
);

CREATE TABLE ejecuciones (
    id_ejecucion   INTEGER PRIMARY KEY AUTOINCREMENT,
    modelo         TEXT    NOT NULL,
    version        TEXT    NOT NULL,
    umbral         REAL    NOT NULL,
    auprc          REAL,
    recall         REAL,
    precision_val  REAL,
    fecha          TEXT    NOT NULL
);

CREATE TABLE predicciones (
    id_prediccion  INTEGER PRIMARY KEY AUTOINCREMENT,
    id_transaccion INTEGER NOT NULL,
    id_ejecucion   INTEGER NOT NULL,
    score_fraude   REAL    NOT NULL,     -- probabilidad estimada de fraude
    prediccion     INTEGER NOT NULL,     -- 0/1 según el umbral aplicado
    FOREIGN KEY (id_transaccion) REFERENCES transacciones(id_transaccion),
    FOREIGN KEY (id_ejecucion)   REFERENCES ejecuciones(id_ejecucion)
);

CREATE TABLE alertas (
    id_alerta      INTEGER PRIMARY KEY AUTOINCREMENT,
    id_transaccion INTEGER NOT NULL,
    score_fraude   REAL    NOT NULL,
    prioridad      TEXT    NOT NULL,     -- alta / media / baja
    estado         TEXT    DEFAULT 'pendiente',  -- pendiente/confirmada/descartada
    fecha_alerta   TEXT    NOT NULL,
    FOREIGN KEY (id_transaccion) REFERENCES transacciones(id_transaccion)
);

CREATE INDEX idx_pred_score ON predicciones(score_fraude);
CREATE INDEX idx_alertas_estado ON alertas(estado);
"""


def prioridad(score):
    if score >= 0.80:
        return "alta"
    if score >= 0.50:
        return "media"
    return "baja"


def main():
    ruta_muestra = os.path.join(DATA_DIR, "muestra_test.csv")
    if not os.path.exists(ruta_muestra):
        raise FileNotFoundError(
            "No existe data/muestra_test.csv. Ejecuta antes 'python src/train_model.py'."
        )

    df = pd.read_csv(ruta_muestra)
    print(f"[1] Muestra cargada: {len(df):,} transacciones")

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript(ESQUEMA)
    print("[2] Esquema creado (transacciones, ejecuciones, predicciones, alertas)")

    ahora = datetime.now().isoformat(timespec="seconds")

    # --- transacciones ---
    filas = [(float(r.Time), float(r.Amount), int(r.Class), "creditcard.csv", ahora)
             for r in df.itertuples()]
    cur.executemany(
        "INSERT INTO transacciones (tiempo, importe, clase_real, origen, fecha_carga) "
        "VALUES (?,?,?,?,?)", filas)
    print(f"[3] {len(filas):,} transacciones insertadas")

    # --- ejecucion ---
    metricas = {}
    ruta_met = os.path.join(MODEL_DIR, "metricas.json")
    if os.path.exists(ruta_met):
        metricas = json.load(open(ruta_met, encoding="utf-8"))
    umbral = metricas.get("umbral", 0.5)
    cur.execute(
        "INSERT INTO ejecuciones (modelo, version, umbral, auprc, recall, precision_val, fecha) "
        "VALUES (?,?,?,?,?,?,?)",
        ("RandomForestClassifier", "1.0", umbral, metricas.get("AUPRC"),
         metricas.get("recall"), metricas.get("precision"), ahora))
    id_ejec = cur.lastrowid
    print(f"[4] Ejecución registrada (id={id_ejec}, umbral={umbral})")

    # --- predicciones ---
    preds = []
    for i, r in enumerate(df.itertuples(), start=1):
        score = float(r.score_fraude)
        preds.append((i, id_ejec, score, int(score >= umbral)))
    cur.executemany(
        "INSERT INTO predicciones (id_transaccion, id_ejecucion, score_fraude, prediccion) "
        "VALUES (?,?,?,?)", preds)
    print(f"[5] {len(preds):,} predicciones almacenadas")

    # --- alertas (solo las que superan el umbral) ---
    alertas = [(i, s, prioridad(s), "pendiente", ahora)
               for (i, _, s, p) in preds if p == 1]
    cur.executemany(
        "INSERT INTO alertas (id_transaccion, score_fraude, prioridad, estado, fecha_alerta) "
        "VALUES (?,?,?,?,?)", alertas)
    print(f"[6] {len(alertas):,} alertas generadas para revisión")

    con.commit()

    # --- verificación ---
    print("\n[7] Verificación del contenido:")
    for tabla in ["transacciones", "predicciones", "alertas", "ejecuciones"]:
        n = cur.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
        print(f"    {tabla}: {n:,} registros")

    print("\n[8] Consulta de ejemplo — alertas de prioridad alta:")
    q = """
        SELECT a.id_alerta, t.importe, a.score_fraude, a.prioridad, t.clase_real
        FROM alertas a
        JOIN transacciones t ON t.id_transaccion = a.id_transaccion
        WHERE a.prioridad = 'alta'
        ORDER BY a.score_fraude DESC
        LIMIT 5;
    """
    for fila in cur.execute(q).fetchall():
        print("   ", fila)

    con.close()
    print(f"\nBase de datos creada en {DB_PATH}")


if __name__ == "__main__":
    main()
