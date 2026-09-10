# -*- coding: utf-8 -*-
"""
TFM - Detección de fraude con tarjeta de crédito
Equipo 3_G — José Luis Acosta Bendeck y David Juárez Vásquez
Universidad Internacional de La Rioja (UNIR)

Prototipo: cuadro de mando interactivo para el analista de fraude.

Permite:
    - Ajustar el umbral de decisión y observar su efecto en tiempo real.
    - Consultar métricas, matriz de confusión y coste económico estimado.
    - Explorar la cola de alertas priorizada por score de fraude.

Uso:
    streamlit run app/dashboard.py
"""

import os
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
)

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

COLOR_LEGIT = "#2E74B5"
COLOR_FRAUD = "#C00000"

# Supuestos de coste (editables desde la barra lateral)
COSTE_FN_DEF = 123.87   # importe medio de fraude no detectado
COSTE_FP_DEF = 8.0      # revisión manual de una falsa alarma
COSTE_TP_DEF = 5.0      # gestión de un fraude detectado

st.set_page_config(page_title="Detección de fraude — TFM Equipo 3_G",
                   page_icon="💳", layout="wide")


@st.cache_data
def cargar_datos():
    ruta = os.path.join(DATA_DIR, "muestra_test.csv")
    if not os.path.exists(ruta):
        return None
    return pd.read_csv(ruta)


@st.cache_data
def cargar_metricas():
    ruta = os.path.join(MODEL_DIR, "metricas.json")
    if os.path.exists(ruta):
        return json.load(open(ruta, encoding="utf-8"))
    return {}


@st.cache_data
def cargar_importancia():
    ruta = os.path.join(MODEL_DIR, "importancia_variables.csv")
    if os.path.exists(ruta):
        return pd.read_csv(ruta, index_col=0)
    return None


# --------------------------------------------------------------------------
# Cabecera
# --------------------------------------------------------------------------
st.title("💳 Detección de fraude en transacciones con tarjeta de crédito")
st.caption("Prototipo del Trabajo Final de Maestría — Equipo 3_G · "
           "José Luis Acosta Bendeck y David Juárez Vásquez · UNIR")

df = cargar_datos()
if df is None:
    st.error("No se encuentra `data/muestra_test.csv`. "
             "Ejecuta primero `python src/train_model.py`.")
    st.stop()

y_true = df["Class"].values
scores = df["score_fraude"].values

# --------------------------------------------------------------------------
# Barra lateral: parámetros
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Parámetros de operación")
umbral = st.sidebar.slider(
    "Umbral de decisión", min_value=0.05, max_value=0.95, value=0.30, step=0.05,
    help="Probabilidad a partir de la cual una transacción se marca como sospechosa.")

st.sidebar.markdown("---")
st.sidebar.subheader("Supuestos de coste (EUR)")
c_fn = st.sidebar.number_input("Fraude no detectado (FN)", value=COSTE_FN_DEF, step=10.0)
c_fp = st.sidebar.number_input("Falsa alarma (FP)", value=COSTE_FP_DEF, step=1.0)
c_tp = st.sidebar.number_input("Fraude detectado (TP)", value=COSTE_TP_DEF, step=1.0)

st.sidebar.markdown("---")
st.sidebar.warning(
    "**Nota metodológica:** los datos mostrados son una muestra del conjunto de "
    "prueba (los 142 fraudes más 5.000 operaciones legítimas de las 84.976 "
    "totales). Por ello la precisión y el número de falsas alarmas que aquí se "
    "muestran son más favorables que los del conjunto completo. Las métricas de "
    "referencia del proyecto, calculadas sobre las 85.118 transacciones de "
    "prueba, son: AUPRC 0,808 · precisión 0,876 · recall 0,746 · 15 falsas alarmas."
)

# --------------------------------------------------------------------------
# Cálculos según el umbral seleccionado
# --------------------------------------------------------------------------
pred = (scores >= umbral).astype(int)
cm = confusion_matrix(y_true, pred)
tn, fp, fn, tp = cm.ravel()

prec = precision_score(y_true, pred, zero_division=0)
rec = recall_score(y_true, pred, zero_division=0)
f1 = f1_score(y_true, pred, zero_division=0)
auprc = average_precision_score(y_true, scores)
coste_total = fn * c_fn + fp * c_fp + tp * c_tp

# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------
st.subheader("Indicadores clave")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("AUPRC", f"{auprc:.3f}", help="Área bajo la curva precisión-exhaustividad")
k2.metric("Recall", f"{rec:.1%}", help="Porcentaje de fraudes detectados")
k3.metric("Precisión", f"{prec:.1%}", help="De las alertas emitidas, cuántas son fraude real")
k4.metric("Falsas alarmas", f"{fp:,}", help="Operaciones legítimas marcadas por error")
k5.metric("Coste estimado", f"{coste_total:,.0f} €", help="Coste total bajo los supuestos definidos")

st.markdown("---")

# --------------------------------------------------------------------------
# Fila 1: matriz de confusión + coste según umbral
# --------------------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Matriz de confusión")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow([[0, 1], [1, 0]], cmap="Blues", alpha=0.12)
    etiquetas = [[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]]
    for i in range(2):
        for j in range(2):
            color = COLOR_FRAUD if (i, j) in [(0, 1), (1, 0)] else "#1a1a1a"
            ax.text(j, i, etiquetas[i][j], ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. legítima", "Pred. fraude"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Real legítima", "Real fraude"])
    for s in ax.spines.values():
        s.set_visible(False)
    st.pyplot(fig)
    st.caption(f"Con umbral {umbral:.2f}: se detectan **{tp}** fraudes, "
               f"se escapan **{fn}**, y se revisan **{fp}** operaciones legítimas.")

with c2:
    st.subheader("Coste estimado según el umbral")
    rango = np.arange(0.05, 1.0, 0.05)
    costes = []
    for u in rango:
        p = (scores >= u).astype(int)
        m = confusion_matrix(y_true, p)
        _tn, _fp, _fn, _tp = m.ravel()
        costes.append(_fn * c_fn + _fp * c_fp + _tp * c_tp)
    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    ax2.plot(rango, costes, color=COLOR_FRAUD, lw=2, marker="o", markersize=4)
    ax2.axvline(umbral, ls="--", color="grey", lw=1.4)
    i_min = int(np.argmin(costes))
    ax2.scatter([rango[i_min]], [costes[i_min]], color="#2E8B57", s=120, zorder=5,
                label=f"Óptimo: {rango[i_min]:.2f} ({costes[i_min]:,.0f} €)")
    ax2.set_xlabel("Umbral de decisión"); ax2.set_ylabel("Coste estimado (€)")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    st.pyplot(fig2)
    st.caption("La línea discontinua marca el umbral seleccionado en la barra lateral.")

st.markdown("---")

# --------------------------------------------------------------------------
# Fila 2: curva PR + importancia de variables
# --------------------------------------------------------------------------
c3, c4 = st.columns(2)

with c3:
    st.subheader("Curva precisión-exhaustividad")
    pr, rc, _ = precision_recall_curve(y_true, scores)
    fig3, ax3 = plt.subplots(figsize=(5.5, 4))
    ax3.plot(rc, pr, color=COLOR_LEGIT, lw=2, label=f"AUPRC = {auprc:.3f}")
    ax3.scatter([rec], [prec], color=COLOR_FRAUD, s=120, zorder=5,
                label=f"Punto de operación (umbral {umbral:.2f})")
    ax3.set_xlabel("Recall"); ax3.set_ylabel("Precisión")
    ax3.legend(fontsize=9); ax3.grid(alpha=0.3)
    st.pyplot(fig3)

with c4:
    st.subheader("Variables más influyentes")
    imp = cargar_importancia()
    if imp is not None:
        top = imp.iloc[:12].iloc[::-1]
        fig4, ax4 = plt.subplots(figsize=(5.5, 4))
        ax4.barh(top.index, top.iloc[:, 0], color=COLOR_LEGIT)
        ax4.set_xlabel("Importancia")
        st.pyplot(fig4)
        st.caption("Importancia calculada por reducción de impureza en el Random Forest.")
    else:
        st.info("Ejecuta `python src/train_model.py` para generar la importancia de variables.")

st.markdown("---")

# --------------------------------------------------------------------------
# Cola de alertas
# --------------------------------------------------------------------------
st.subheader("🚨 Cola de alertas para revisión")
st.write("Transacciones marcadas como sospechosas con el umbral actual, "
         "ordenadas por probabilidad de fraude descendente.")

alertas = df[pred == 1].copy()
alertas["prioridad"] = pd.cut(
    alertas["score_fraude"], bins=[0, 0.5, 0.8, 1.01],
    labels=["baja", "media", "alta"], right=False)
alertas = alertas.sort_values("score_fraude", ascending=False)

col_a, col_b = st.columns([1, 3])
with col_a:
    filtro = st.multiselect("Filtrar por prioridad",
                            options=["alta", "media", "baja"],
                            default=["alta", "media", "baja"])
mostrar = alertas[alertas["prioridad"].isin(filtro)]

tabla = mostrar[["Amount", "score_fraude", "prioridad", "Class"]].head(50).rename(columns={
    "Amount": "Importe (escalado)", "score_fraude": "Probabilidad de fraude",
    "prioridad": "Prioridad", "Class": "Etiqueta real"})
st.dataframe(tabla, use_container_width=True)
st.caption(f"Mostrando hasta 50 de {len(mostrar):,} alertas. "
           "La columna «Etiqueta real» solo está disponible en este entorno de "
           "validación; en producción no se conocería en el momento de la alerta.")
