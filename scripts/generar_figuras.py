"""
Genera las 4 figuras de la tesis en formato PNG (300 DPI).
Ejecutar desde la raíz del proyecto:
    python docs/generar_figuras.py
"""

import json
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Rutas ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figuras"
OUT.mkdir(exist_ok=True)

PHASE1_CSV = ROOT / "logs" / "geotermia_v7_phase1.csv"
PHASE2_CSV = ROOT / "logs" / "geotermia_v7_phase2.csv"
EVAL_JSON  = ROOT / "results" / "metrics" / "evaluation_metrics.json"

# ── Paleta de colores consistente ─────────────────────────────────
C_PRIMARY   = "#2563EB"   # azul principal
C_SECONDARY = "#7C3AED"   # violeta
C_ACCENT    = "#059669"   # verde
C_WARN      = "#D97706"   # ámbar
C_DARK      = "#1E293B"   # casi negro
C_LIGHT     = "#F1F5F9"   # gris muy claro
C_GRAY      = "#64748B"   # gris medio
C_RED       = "#DC2626"   # rojo


def _style():
    """Estilo global limpio para todas las figuras."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "axes.edgecolor": C_GRAY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


# ══════════════════════════════════════════════════════════════════
# FIGURA 1 — Pipeline de procesamiento
# ══════════════════════════════════════════════════════════════════

def _rounded_box(ax, x, y, w, h, text, color, text_color="white",
                 fontsize=9, bold=False, subtext=None):
    """Dibuja un rectángulo redondeado con texto centrado."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="none",
        linewidth=0, zorder=2
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y + (0.06 if subtext else 0), text,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=3)
    if subtext:
        ax.text(x, y - 0.15, subtext,
                ha="center", va="center", fontsize=7,
                color=text_color, alpha=0.85, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color=C_GRAY):
    """Flecha entre cajas."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14),
                zorder=1)


def figura_1_pipeline():
    """Pipeline de procesamiento de datos."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.2, 2.0)
    ax.axis("off")

    # Título
    ax.text(5, 1.7, "Pipeline de procesamiento de datos",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=C_DARK)

    # Fila superior: flujo principal
    steps = [
        (0.8,  0.7, "Descarga\nGEE API",       "2.019 .tif",    C_PRIMARY),
        (2.8,  0.7, "Aumento\nde datos (×10)",  "22.209 imgs",   C_SECONDARY),
        (4.8,  0.7, "Filtrado\nNoData",          "Mediana/descarte", C_WARN),
        (6.8,  0.7, "Redimensionar\n224 × 224",  "Bicúbica + AA", C_ACCENT),
        (8.8,  0.7, "Normalización\nz-score",    "Global/banda",  C_PRIMARY),
    ]

    for x, y, txt, sub, col in steps:
        _rounded_box(ax, x, y, 1.7, 0.75, txt, col, subtext=sub, fontsize=9, bold=True)

    # Flechas entre pasos superiores
    for i in range(len(steps) - 1):
        _arrow(ax, steps[i][0] + 0.85, steps[i][1],
               steps[i+1][0] - 0.85, steps[i+1][1])

    # Fila inferior
    steps_low = [
        (2.8, -0.5, "Particionado\n.npy",       "~500 imgs/lote", C_GRAY),
        (4.8, -0.5, "Entrenamiento\nGPU",        "2 fases, 80 ép.", C_RED),
        (6.8, -0.5, "Evaluación\n+ Bootstrap",   "3.619 test imgs", C_ACCENT),
        (8.8, -0.5, "Despliegue\nStreamlit",     "Inferencia web",  C_SECONDARY),
    ]

    for x, y, txt, sub, col in steps_low:
        _rounded_box(ax, x, y, 1.7, 0.75, txt, col, subtext=sub, fontsize=9, bold=True)

    # Flechas fila inferior
    for i in range(len(steps_low) - 1):
        _arrow(ax, steps_low[i][0] + 0.85, steps_low[i][1],
               steps_low[i+1][0] - 0.85, steps_low[i+1][1])

    # Flecha de conexión (z-score → particionado)
    _arrow(ax, 8.8, 0.7 - 0.375, 2.8, -0.5 + 0.375 + 0.05, color=C_GRAY)

    # Datos de entrada (izquierda)
    _rounded_box(ax, 0.8, -0.5, 1.7, 0.75,
                 "ASTER GED\nAG100 v003", "#334155",
                 subtext="7 bandas, 90 m/px", fontsize=9, bold=True)
    _arrow(ax, 0.8, -0.5 + 0.375, 0.8, 0.7 - 0.375, color=C_GRAY)

    path = OUT / "figura_1_pipeline.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ══════════════════════════════════════════════════════════════════
# FIGURA 2 — Arquitectura del modelo
# ══════════════════════════════════════════════════════════════════

def figura_2_arquitectura():
    """Arquitectura EfficientNetB0 + Channel Adapter."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-1.5, 3.5)
    ax.axis("off")

    ax.text(6.25, 3.1, "Arquitectura: EfficientNetB0 + Channel Adapter",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=C_DARK)

    # ── INPUT ──
    _rounded_box(ax, 0.8, 1.5, 2.0, 1.0,
                 "INPUT", "#334155",
                 subtext="224 × 224 × 7", fontsize=11, bold=True)

    # ── CHANNEL ADAPTER ──
    ax.add_patch(FancyBboxPatch(
        (2.3, 0.2), 2.6, 2.6,
        boxstyle="round,pad=0.15",
        facecolor="#EEF2FF", edgecolor=C_PRIMARY,
        linewidth=1.5, linestyle="--", zorder=1
    ))
    ax.text(3.6, 2.55, "Channel Adapter", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_PRIMARY)
    _rounded_box(ax, 3.6, 1.9, 2.1, 0.55,
                 "Conv2D(16, 3×3)", C_PRIMARY,
                 subtext="BN + ReLU", fontsize=8.5, bold=True)
    _rounded_box(ax, 3.6, 1.1, 2.1, 0.55,
                 "Conv2D(3, 1×1)", C_PRIMARY,
                 subtext="BN + ReLU", fontsize=8.5, bold=True)
    _arrow(ax, 3.6, 1.9 - 0.275, 3.6, 1.1 + 0.275)

    # ── BACKBONE ──
    ax.add_patch(FancyBboxPatch(
        (5.5, 0.2), 2.6, 2.6,
        boxstyle="round,pad=0.15",
        facecolor="#F5F3FF", edgecolor=C_SECONDARY,
        linewidth=1.5, linestyle="--", zorder=1
    ))
    ax.text(6.8, 2.55, "EfficientNetB0 (ImageNet)", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_SECONDARY)
    _rounded_box(ax, 6.8, 1.9, 2.1, 0.55,
                 "237 capas MBConv", C_SECONDARY,
                 subtext="SE blocks", fontsize=8.5, bold=True)
    _rounded_box(ax, 6.8, 1.1, 2.1, 0.55,
                 "Feature maps", C_SECONDARY,
                 subtext="7 × 7 × 1.280", fontsize=8.5, bold=True)
    _arrow(ax, 6.8, 1.9 - 0.275, 6.8, 1.1 + 0.275)

    # ── CLASSIFICATION HEAD ──
    ax.add_patch(FancyBboxPatch(
        (8.7, 0.2), 3.8, 2.6,
        boxstyle="round,pad=0.15",
        facecolor="#ECFDF5", edgecolor=C_ACCENT,
        linewidth=1.5, linestyle="--", zorder=1
    ))
    ax.text(10.6, 2.55, "Classification Head", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_ACCENT)
    _rounded_box(ax, 9.6, 1.5, 1.3, 0.55,
                 "GAP", C_ACCENT,
                 subtext="→ 1.280", fontsize=8.5, bold=True)
    _rounded_box(ax, 11.1, 2.0, 1.5, 0.45,
                 "Dense(256)+Drop(0.5)", C_ACCENT,
                 fontsize=7.5, bold=True)
    _rounded_box(ax, 11.1, 1.4, 1.5, 0.45,
                 "Dense(64)+Drop(0.3)", C_ACCENT,
                 fontsize=7.5, bold=True)
    _rounded_box(ax, 11.1, 0.8, 1.5, 0.45,
                 "Dense(1, sigmoid)", "#065F46",
                 fontsize=7.5, bold=True)
    # Flechas internas
    _arrow(ax, 9.6 + 0.65, 1.5, 11.1 - 0.75, 2.0)
    _arrow(ax, 11.1, 2.0 - 0.225, 11.1, 1.4 + 0.225)
    _arrow(ax, 11.1, 1.4 - 0.225, 11.1, 0.8 + 0.225)

    # ── OUTPUT ──
    _rounded_box(ax, 11.1, -0.2, 2.0, 0.6,
                 "P(geotérmico)", C_RED,
                 subtext="0.0 – 1.0", fontsize=9, bold=True)
    _arrow(ax, 11.1, 0.8 - 0.225, 11.1, -0.2 + 0.3)

    # ── Flechas principales entre bloques ──
    _arrow(ax, 0.8 + 1.0, 1.5, 3.6 - 1.05 - 0.25, 1.5)
    _arrow(ax, 3.6 + 1.05 + 0.25, 1.5, 6.8 - 1.05 - 0.25, 1.5)
    _arrow(ax, 6.8 + 1.05 + 0.25, 1.5, 9.6 - 0.65, 1.5)

    # ── Etiquetas de dimensiones en las flechas ──
    ax.text(2.15, 1.82, "224×224×7", ha="center", fontsize=7, color=C_GRAY)
    ax.text(5.2, 1.82, "224×224×3", ha="center", fontsize=7, color=C_GRAY)
    ax.text(8.35, 1.82, "7×7×1280", ha="center", fontsize=7, color=C_GRAY)

    # ── Leyenda de parámetros ──
    info = [
        "Total: 4.396.112 parámetros",
        "Adapter: 0,03 %  |  Backbone: 92,1 %  |  Head: 7,8 %",
    ]
    ax.text(6.25, -0.85, "  ·  ".join(info),
            ha="center", va="center", fontsize=8.5, color=C_GRAY,
            style="italic")

    path = OUT / "figura_2_arquitectura.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ══════════════════════════════════════════════════════════════════
# FIGURA 3 — Curvas de entrenamiento
# ══════════════════════════════════════════════════════════════════

def _read_csv(filepath):
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def figura_3_curvas():
    """Curvas de entrenamiento — AUC y loss por época (datos reales)."""
    phase1 = _read_csv(PHASE1_CSV)
    phase2 = _read_csv(PHASE2_CSV)

    # Extraer val_auc y val_loss
    p1_val_auc  = [float(r["val_auc"]) for r in phase1]
    p2_val_auc  = [float(r["val_auc"]) for r in phase2]
    p1_val_loss = [float(r["val_loss"]) for r in phase1]
    p2_val_loss = [float(r["val_loss"]) for r in phase2]
    p1_tr_auc   = [float(r["auc"]) for r in phase1]
    p2_tr_auc   = [float(r["auc"]) for r in phase2]
    p1_tr_loss  = [float(r["loss"]) for r in phase1]
    p2_tr_loss  = [float(r["loss"]) for r in phase2]

    epochs_p1 = list(range(1, len(p1_val_auc) + 1))
    epochs_p2 = list(range(31, 31 + len(p2_val_auc)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel izquierdo: AUC ──
    ax1.plot(epochs_p1, p1_tr_auc, color=C_PRIMARY, alpha=0.4, lw=1.2,
             label="Train AUC (Fase 1)")
    ax1.plot(epochs_p1, p1_val_auc, color=C_PRIMARY, lw=2,
             label="Val AUC (Fase 1)")
    ax1.plot(epochs_p2, p2_tr_auc, color=C_SECONDARY, alpha=0.4, lw=1.2,
             label="Train AUC (Fase 2)")
    ax1.plot(epochs_p2, p2_val_auc, color=C_SECONDARY, lw=2,
             label="Val AUC (Fase 2)")

    # línea de transición
    ax1.axvline(x=30.5, color=C_GRAY, linestyle="--", lw=0.8, alpha=0.6)
    ax1.text(30.5, 0.58, "  Transición\n  F1 → F2", fontsize=7.5,
             color=C_GRAY, va="bottom")

    # Marcar mejores épocas
    best_p1_idx = np.argmax(p1_val_auc)
    best_p2_idx = np.argmax(p2_val_auc)
    ax1.scatter(epochs_p1[best_p1_idx], p1_val_auc[best_p1_idx],
                color=C_PRIMARY, s=60, zorder=5, edgecolors="white", linewidths=1.5)
    ax1.annotate(f"Ép. {epochs_p1[best_p1_idx]}: {p1_val_auc[best_p1_idx]:.4f}",
                 xy=(epochs_p1[best_p1_idx], p1_val_auc[best_p1_idx]),
                 xytext=(epochs_p1[best_p1_idx] - 8, p1_val_auc[best_p1_idx] + 0.03),
                 fontsize=8, color=C_PRIMARY,
                 arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=0.8))
    ax1.scatter(epochs_p2[best_p2_idx], p2_val_auc[best_p2_idx],
                color=C_SECONDARY, s=60, zorder=5, edgecolors="white", linewidths=1.5)
    ax1.annotate(f"Ép. {epochs_p2[best_p2_idx]}: {p2_val_auc[best_p2_idx]:.4f}",
                 xy=(epochs_p2[best_p2_idx], p2_val_auc[best_p2_idx]),
                 xytext=(epochs_p2[best_p2_idx] - 14, p2_val_auc[best_p2_idx] - 0.04),
                 fontsize=8, color=C_SECONDARY,
                 arrowprops=dict(arrowstyle="->", color=C_SECONDARY, lw=0.8))

    ax1.set_xlabel("Época")
    ax1.set_ylabel("AUC-ROC")
    ax1.set_title("a) Evolución del AUC-ROC")
    ax1.set_ylim(0.55, 1.0)
    ax1.legend(loc="lower right", framealpha=0.9, edgecolor=C_LIGHT)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel derecho: Loss ──
    ax2.plot(epochs_p1, p1_tr_loss, color=C_PRIMARY, alpha=0.4, lw=1.2,
             label="Train Loss (Fase 1)")
    ax2.plot(epochs_p1, p1_val_loss, color=C_PRIMARY, lw=2,
             label="Val Loss (Fase 1)")
    ax2.plot(epochs_p2, p2_tr_loss, color=C_SECONDARY, alpha=0.4, lw=1.2,
             label="Train Loss (Fase 2)")
    ax2.plot(epochs_p2, p2_val_loss, color=C_SECONDARY, lw=2,
             label="Val Loss (Fase 2)")

    ax2.axvline(x=30.5, color=C_GRAY, linestyle="--", lw=0.8, alpha=0.6)

    ax2.set_xlabel("Época")
    ax2.set_ylabel("Loss (BinaryCrossentropy)")
    ax2.set_title("b) Evolución de la función de pérdida")
    ax2.legend(loc="upper right", framealpha=0.9, edgecolor=C_LIGHT)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Curvas de entrenamiento — Fases 1 y 2",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.01)
    fig.tight_layout()

    path = OUT / "figura_3_curvas.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ══════════════════════════════════════════════════════════════════
# FIGURA 4 — Ejemplo de predicción paso a paso
# ══════════════════════════════════════════════════════════════════

def figura_4_prediccion():
    """Flujo de predicción paso a paso (Nevado del Ruiz)."""
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-0.5, 9.5)
    ax.axis("off")

    ax.text(5, 9.1, "Ejemplo de predicción paso a paso",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=C_DARK)

    steps = [
        (5, 8.0, "1. Entrada de coordenadas",
         "(4.895° N, −75.322° W) — Nevado del Ruiz, Colombia", "#334155"),
        (5, 6.8, "2. Descarga ASTER GED",
         "7 bandas TIR + temp. + NDVI → imagen 111 × 111 px", C_PRIMARY),
        (5, 5.6, "3. Preprocesamiento",
         "Resize 224 × 224 (bicúbica) + normalización z-score global", C_WARN),
        (5, 4.4, "4. Channel Adapter",
         "7 bandas → Conv2D(16) → Conv2D(3) → tensor 224 × 224 × 3", C_PRIMARY),
        (5, 3.2, "5. EfficientNetB0 backbone",
         "237 capas MBConv + SE → mapa de features 7 × 7 × 1.280", C_SECONDARY),
        (5, 2.0, "6. Classification Head",
         "GAP → Dense(256) → Dense(64) → Dense(1, sigmoid)", C_ACCENT),
    ]

    for x, y, title, desc, col in steps:
        _rounded_box(ax, x, y, 8.5, 0.7, title, col,
                     subtext=desc, fontsize=11, bold=True)

    # Flechas entre pasos
    for i in range(len(steps) - 1):
        _arrow(ax, 5, steps[i][1] - 0.35, 5, steps[i+1][1] + 0.35, color=C_GRAY)

    # Resultado final — caja más grande y llamativa
    ax.add_patch(FancyBboxPatch(
        (5 - 4.25, 0.35), 8.5, 1.0,
        boxstyle="round,pad=0.15",
        facecolor="#FEF2F2", edgecolor=C_RED,
        linewidth=2.5, zorder=2
    ))
    ax.text(5, 1.05, "7. RESULTADO: ALTO potencial geotérmico",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=C_RED, zorder=3)
    ax.text(5, 0.65, "Probabilidad = 0,9842  (98,42 %)  ·  Tiempo de inferencia < 3 s",
            ha="center", va="center", fontsize=9, color=C_RED,
            alpha=0.8, zorder=3)

    _arrow(ax, 5, 2.0 - 0.35, 5, 1.35, color=C_RED)

    path = OUT / "figura_4_prediccion.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _style()
    print("Generando figuras de la tesis...\n")
    figura_1_pipeline()
    figura_2_arquitectura()
    figura_3_curvas()
    figura_4_prediccion()
    print(f"\n✓ Las 4 figuras se guardaron en: {OUT}")
