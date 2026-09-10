# -*- coding: utf-8 -*-
"""
TFM - Detección de fraude con tarjeta de crédito
Equipo 3_G — José Luis Acosta Bendeck y David Juárez Vásquez
Universidad Internacional de La Rioja (UNIR)

Fase de entrenamiento del prototipo.

Este script recrea el pipeline completo del proyecto:
    1. Carga el conjunto 'creditcard.csv'.
    2. Elimina duplicados exactos.
    3. Particiona de forma estratificada (70/30).
    4. Escala Time y Amount con RobustScaler ajustado solo en entrenamiento.
    5. Entrena un Random Forest con ponderación de clases.
    6. Evalúa sobre el conjunto de prueba.
    7. Serializa el modelo, el escalador y la muestra de prueba.

Uso:
    python src/train_model.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.30

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def localizar_csv():
    """Busca creditcard.csv en las ubicaciones habituales."""
    candidatos = [
        os.path.join(DATA_DIR, "creditcard.csv"),
        os.path.join(BASE_DIR, "creditcard.csv"),
        os.path.join(os.getcwd(), "creditcard.csv"),
    ]
    for ruta in candidatos:
        if os.path.exists(ruta):
            return ruta
    raise FileNotFoundError(
        "No se encontró 'creditcard.csv'. Descárgalo de Kaggle y colócalo en la carpeta data/. "
        "Ver instrucciones en el README.md"
    )


def cargar_y_limpiar():
    ruta = localizar_csv()
    print(f"[1] Cargando datos desde {ruta}")
    df = pd.read_csv(ruta)
    print(f"    Dimensiones originales: {df.shape[0]:,} x {df.shape[1]}")

    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[2] Duplicados eliminados: {n_dup:,}")
    print(f"    Dimensiones tras limpieza: {df.shape[0]:,} x {df.shape[1]}")
    print(f"    Fraudes: {int(df.Class.sum())} ({100*df.Class.mean():.3f} %)")
    return df


def preparar(df):
    X = df.drop(columns=["Class"])
    y = df["Class"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[3] Partición estratificada {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
    print(f"    Entrenamiento: {X_tr.shape[0]:,} ({int(y_tr.sum())} fraudes)")
    print(f"    Prueba:        {X_te.shape[0]:,} ({int(y_te.sum())} fraudes)")

    scaler = RobustScaler()
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    X_tr[["Time", "Amount"]] = scaler.fit_transform(X_tr[["Time", "Amount"]])
    X_te[["Time", "Amount"]] = scaler.transform(X_te[["Time", "Amount"]])
    print("[4] Escalado robusto aplicado (ajustado solo con entrenamiento)")

    return X_tr, X_te, y_tr, y_te, scaler


def entrenar(X_tr, y_tr):
    print("[5] Entrenando Random Forest (class_weight='balanced')...")
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    modelo.fit(X_tr, y_tr)
    print("    Entrenamiento completado.")
    return modelo


def evaluar(modelo, X_te, y_te, umbral=0.5):
    proba = modelo.predict_proba(X_te)[:, 1]
    pred = (proba >= umbral).astype(int)
    cm = confusion_matrix(y_te, pred)

    metricas = {
        "umbral": umbral,
        "AUPRC": float(average_precision_score(y_te, proba)),
        "AUROC": float(roc_auc_score(y_te, proba)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
    }
    print("[6] Evaluación sobre el conjunto de prueba:")
    for k, v in metricas.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    return metricas, proba


def main():
    df = cargar_y_limpiar()
    X_tr, X_te, y_tr, y_te, scaler = preparar(df)
    modelo = entrenar(X_tr, y_tr)
    metricas, proba = evaluar(modelo, X_te, y_te)

    # --- Serialización de artefactos ---
    joblib.dump(modelo, os.path.join(MODEL_DIR, "modelo_rf.pkl"), compress=3)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"[7] Modelo y escalador serializados en {MODEL_DIR}/")

    # Importancia de variables
    imp = pd.Series(modelo.feature_importances_, index=X_tr.columns)
    imp.sort_values(ascending=False).to_csv(
        os.path.join(MODEL_DIR, "importancia_variables.csv"), encoding="utf-8-sig"
    )

    # --- Muestra del conjunto de prueba para el dashboard ---
    # Se incluyen todos los fraudes y una muestra de legítimas, para que el
    # repositorio sea ligero pero el prototipo funcione al clonarlo.
    test = X_te.copy()
    test["Class"] = y_te
    test["score_fraude"] = proba

    fraudes = test[test.Class == 1]
    legitimas = test[test.Class == 0].sample(
        n=min(5000, (test.Class == 0).sum()), random_state=RANDOM_STATE
    )
    muestra = pd.concat([fraudes, legitimas]).sample(frac=1, random_state=RANDOM_STATE)
    muestra.to_csv(os.path.join(DATA_DIR, "muestra_test.csv"), index=False, encoding="utf-8")
    print(f"[8] Muestra de prueba guardada: {len(muestra):,} filas "
          f"({int(muestra.Class.sum())} fraudes) en data/muestra_test.csv")

    print("\nPROCESO COMPLETADO")


if __name__ == "__main__":
    main()
