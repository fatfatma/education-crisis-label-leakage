"""Sekil 1: kurgulara gore makro F1 (nokta + %95 GA), referans modellerle.
Kullanim: python -I make_fig.py <sonuclar.csv> <cikti_klasoru>
"""
import sys, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
r = pd.read_csv(src)
r["k"] = r["kurgu"].str.split(":").str[0]

order = ["A", "B", "B2", "C"]
xlab = {"A": "A\nbileşenler\ngirdide", "B": "B\nham göstergeler\ngirdide",
        "B2": "B2\ngöstergeler\nçıkarıldı", "C": "C\ngelecek yıl\n(t+1)"}
models = [("LojistikRegresyon", "Lojistik Regresyon", "#2a78d6", "o"),
          ("RandomForest", "Random Forest", "#eb6834", "s"),
          ("GradientBoosting", "Gradient Boosting", "#1baf7a", "^")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(3.5, 2.7))
off = {0: -0.22, 1: 0.0, 2: 0.22}

for gi, k in enumerate(order):
    g = r[r["k"] == k]
    maj = g.loc[g["model"] == "Cogunluk_sinifi", "makro_F1"].iloc[0]
    ax.hlines(maj, gi - 0.42, gi + 0.42, colors=MUTED, linestyles=(0, (3, 2)), linewidth=1, zorder=1)
    for mi, (key, name, col, mk) in enumerate(models):
        row = g[g["model"] == key].iloc[0]
        x = gi + off[mi]
        ax.vlines(x, row["F1_GA_alt"], row["F1_GA_ust"], colors=col, linewidth=1.6, zorder=2)
        ax.plot(x, row["makro_F1"], marker=mk, ms=5.5, color=col, mec="white", mew=0.8,
                linestyle="none", zorder=3, label=name if gi == 0 else None)
    if k == "C":
        p = g[g["model"].str.startswith("Kalicilik")].iloc[0]
        xp = gi + 0.42
        ax.vlines(xp, p["F1_GA_alt"], p["F1_GA_ust"], colors=INK, linewidth=1.6, zorder=2)
        ax.plot(xp, p["makro_F1"], marker="D", ms=5, color=INK, mec="white", mew=0.8,
                linestyle="none", zorder=3, label="Kalıcılık (geçen yılın etiketi)")

ax.plot([], [], color=MUTED, linestyle=(0, (3, 2)), linewidth=1, label="Çoğunluk sınıfı")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([xlab[k] for k in order], fontsize=6.8)
ax.set_xlim(-0.6, len(order) - 0.35)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Makro F1 (test, %95 GA)")
ax.yaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.legend(loc="upper right", fontsize=6.3, frameon=False, handlelength=1.6, borderaxespad=0.2)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(out, "fig_leakage.pdf"))
fig.savefig(os.path.join(out, "fig_leakage.png"), dpi=300)
print("ok")
