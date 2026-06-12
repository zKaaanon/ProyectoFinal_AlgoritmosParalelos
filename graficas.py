"""
graficas.py — Visualización de resultados Monte Carlo con OpenMP
Proyecto: Algoritmos Paralelos — Prof. Mario Arturo Nieto Butrón
Genera 5 gráficas listas para el informe PDF.

Uso:
    python graficas.py
Las imágenes se guardan en la misma carpeta que este script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Carpeta base = donde vive este script
BASE = Path(__file__).parent

# ── Estilo general ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

PALETTE = {
    1:  "#534AB7",
    2:  "#1D9E75",
    4:  "#BA7517",
    8:  "#993C1D",
}
N_LABELS  = {100_000: "100 K", 1_000_000: "1 M", 10_000_000: "10 M"}
N_MARKERS = {100_000: "o",     1_000_000: "s",   10_000_000: "^"}

# ── Leer datos ────────────────────────────────────────────────────
df      = pd.read_csv(BASE / "benchmarks_resumen.csv")
par     = df[df["version"] == "paralelo"].copy()
seq     = df[df["version"] == "secuencial"].copy()
res_seq = pd.read_csv(BASE / "resultados_seq.csv")
res_omp = pd.read_csv(BASE / "resultados_omp.csv")

# ══════════════════════════════════════════════════════════════════
#  FIGURA 1 — Speedup vs hilos
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot([1,2,4,8], [1,2,4,8], "k--", lw=1.2, alpha=0.5, label="Speedup ideal (S=p)")

for N, grp in par.groupby("N"):
    grp = grp.sort_values("hilos")
    ax.plot(grp["hilos"], grp["speedup"],
            marker=N_MARKERS[N], color=PALETTE.get(N, "#888"),
            lw=2, ms=7, label=f"N = {N_LABELS[N]}")

ax.set_xlabel("Número de hilos (p)")
ax.set_ylabel("Speedup  S(p) = T(1) / T(p)")
ax.set_title("Speedup vs número de hilos")
ax.set_xticks([1, 2, 4, 8])
ax.set_xlim(0.5, 8.5)
ax.set_ylim(0)
ax.legend(framealpha=0.8)
plt.tight_layout()
plt.savefig(BASE / "fig1_speedup.png")
plt.close()
print("✓ fig1_speedup.png")

# ══════════════════════════════════════════════════════════════════
#  FIGURA 2 — Eficiencia paralela (mapa de calor)
# ══════════════════════════════════════════════════════════════════
pivot = par.pivot(index="N", columns="hilos", values="eficiencia")
pivot.index = [N_LABELS[n] for n in pivot.index]

fig, ax = plt.subplots(figsize=(6, 3.5))
im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.5, vmax=1.05, aspect="auto")

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{h} hilo{'s' if h>1 else ''}" for h in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_title("Eficiencia  E(p) = S(p) / p")
ax.grid(False)

for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        color = "white" if val < 0.72 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Eficiencia")
plt.tight_layout()
plt.savefig(BASE / "fig2_eficiencia.png")
plt.close()
print("✓ fig2_eficiencia.png")

# ══════════════════════════════════════════════════════════════════
#  FIGURA 3 — Tiempo de ejecución con barras de error
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

for ax, (N, grp) in zip(axes, par.groupby("N")):
    grp   = grp.sort_values("hilos")
    seq_t = seq[seq["N"] == N]["tiempo_avg_s"].values[0]
    seq_s = seq[seq["N"] == N]["tiempo_std_s"].values[0]

    ax.bar(0, seq_t, yerr=seq_s, width=0.55,
           color="#888780", capsize=5, zorder=3)

    xs     = range(1, len(grp)+1)
    colors = [PALETTE.get(h, "#aaa") for h in grp["hilos"]]
    ax.bar(xs, grp["tiempo_avg_s"], yerr=grp["tiempo_std_s"],
           width=0.55, color=colors, capsize=5, zorder=3)

    ax.set_xticks(range(len(grp)+1))
    ax.set_xticklabels(["Seq"] + [f"{h}h" for h in grp["hilos"]], fontsize=10)
    ax.set_title(f"N = {N_LABELS[N]}")
    ax.set_ylabel("Tiempo promedio (s)" if N == 100_000 else "")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

fig.suptitle("Tiempo de ejecución por configuración (±1σ, 5 repeticiones)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(BASE / "fig3_tiempos.png")
plt.close()
print("✓ fig3_tiempos.png")

# ══════════════════════════════════════════════════════════════════
#  FIGURA 4 — Monte Carlo vs Black-Scholes
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

ax = axes[0]
versiones = ["Secuencial", "Paralelo", "Black-Scholes"]
precios   = [res_seq["precio_mc"].values[0],
             res_omp["precio_mc"].values[0],
             res_seq["precio_bs"].values[0]]
colors_v  = ["#534AB7", "#1D9E75", "#444441"]
bars = ax.bar(versiones, precios, color=colors_v, width=0.5, zorder=3)
ax.set_ylim(7.9, 8.15)
ax.set_ylabel("Precio de la opción call ($)")
ax.set_title("Precio call europea  (S0=100, K=105, T=1, r=5%, sigma=20%)")
for bar, val in zip(bars, precios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"${val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax = axes[1]
labels_err = ["Secuencial\n(N=10M)", "Paralelo\n(N=10M, 8h)"]
errores    = [res_seq["error_rel_pct"].values[0],
              res_omp["error_rel_pct"].values[0]]
bars2 = ax.bar(labels_err, errores, color=["#534AB7", "#1D9E75"], width=0.4, zorder=3)
ax.axhline(1.0, color="#993C1D", lw=1.5, ls="--", alpha=0.7, label="Umbral 1%")
ax.set_ylabel("Error relativo vs Black-Scholes (%)")
ax.set_title("Precision del metodo Monte Carlo")
ax.legend()
for bar, val in zip(bars2, errores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.4f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(BASE / "fig4_financiero.png")
plt.close()
print("✓ fig4_financiero.png")

# ══════════════════════════════════════════════════════════════════
#  FIGURA 5 — Dashboard resumen
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 8))
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# Panel A: Speedup
ax_s = fig.add_subplot(gs[0, :2])
ax_s.plot([1,2,4,8], [1,2,4,8], "k--", lw=1, alpha=0.4, label="Ideal")
for N, grp in par.groupby("N"):
    grp = grp.sort_values("hilos")
    ax_s.plot(grp["hilos"], grp["speedup"],
              marker=N_MARKERS[N], lw=2, ms=6,
              color=PALETTE.get(N, "#888"), label=f"N={N_LABELS[N]}")
ax_s.set_xticks([1,2,4,8])
ax_s.set_xlabel("Hilos")
ax_s.set_ylabel("Speedup")
ax_s.set_title("A  Speedup vs hilos")
ax_s.legend(fontsize=9, framealpha=0.7)

# Panel B: KPIs financieros
ax_k = fig.add_subplot(gs[0, 2])
ax_k.axis("off")
kpis = [
    ("Precio MC (paralelo)", f"${res_omp['precio_mc'].values[0]:.4f}"),
    ("Precio Black-Scholes",  f"${res_omp['precio_bs'].values[0]:.4f}"),
    ("Error relativo",         f"{res_omp['error_rel_pct'].values[0]:.4f}%"),
    ("VaR 5%",                 f"${res_omp['var_5pct'].values[0]:.2f}"),
    ("Retorno esperado",       f"{res_omp['ret_esperado_pct'].values[0]:.2f}%"),
    ("Tiempo (8h, 10M)",       f"{res_omp['tiempo_s'].values[0]:.2f} s"),
]
y0 = 0.95
for label, val in kpis:
    ax_k.text(0.0, y0, label, transform=ax_k.transAxes,
              fontsize=9, color="#5F5E5A", va="top")
    ax_k.text(1.0, y0, val, transform=ax_k.transAxes,
              fontsize=10, fontweight="bold", color="#26215C", va="top", ha="right")
    y0 -= 0.155
ax_k.set_title("B  KPIs financieros")

# Panel C: Tiempos N=10M
ax_t = fig.add_subplot(gs[1, :2])
grp10 = par[par["N"] == 10_000_000].sort_values("hilos")
seq10 = seq[seq["N"] == 10_000_000]["tiempo_avg_s"].values[0]
seq10s = seq[seq["N"] == 10_000_000]["tiempo_std_s"].values[0]
ys = [seq10] + list(grp10["tiempo_avg_s"])
es = [seq10s] + list(grp10["tiempo_std_s"])
cs = ["#888780"] + [PALETTE.get(h) for h in grp10["hilos"]]
ax_t.bar(range(len(ys)), ys, yerr=es, color=cs, width=0.6, capsize=5, zorder=3)
ax_t.set_xticks(range(len(ys)))
ax_t.set_xticklabels(["Seq"] + [f"{h} hilos" for h in grp10["hilos"]])
ax_t.set_ylabel("Tiempo promedio (s)")
ax_t.set_title("C  Tiempos — N = 10 M trayectorias")

# Panel D: Eficiencia N=10M
ax_e = fig.add_subplot(gs[1, 2])
ax_e.bar(range(len(grp10)), grp10["eficiencia"],
         color=[PALETTE.get(h) for h in grp10["hilos"]], width=0.55, zorder=3)
ax_e.axhline(1.0, color="k", lw=1, ls="--", alpha=0.4)
ax_e.set_xticks(range(len(grp10)))
ax_e.set_xticklabels([f"{h}h" for h in grp10["hilos"]])
ax_e.set_ylim(0, 1.15)
ax_e.set_ylabel("Eficiencia E(p)")
ax_e.set_title("D  Eficiencia — N = 10 M")

fig.suptitle("Monte Carlo con OpenMP — Resumen de resultados",
             fontsize=15, fontweight="bold", y=1.01)
plt.savefig(BASE / "fig5_dashboard.png", bbox_inches="tight")
plt.close()
print("✓ fig5_dashboard.png")

print(f"\n Todas las graficas guardadas en: {BASE}")