"""Kopterix Phase 3 figures: H_rare timeseries and before/after H_t vs n_total panel."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE   = Path(__file__).resolve().parents[1]
TABLES = BASE / "phase3" / "tables"
FIGDIR = BASE / "phase3" / "figures"

df = pd.read_csv(TABLES / "phase3_rarefied_entropy.csv", comment="#")
df["ts"] = pd.to_datetime(df["Timestamp_UTC"].str.replace(" UTC", "", regex=False), utc=True)
df = df.sort_values("ts")

CAVEAT_TXT = ("Rarefied entropy of TOP-200 truncated UnigramCounts (B=703 tokens, "
               "50 multinomial draws, seed=42). Instrument diagnostic, not a "
               "platform-level claim.")

# ── Figure 1: H_rare timeseries ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
regular = df[~df["n_total_regime"]]
lowN    = df[df["n_total_regime"]]

ax.plot(df["ts"], df["H_rare"], color="#4477AA", lw=1, alpha=0.6, zorder=1)
ax.errorbar(regular["ts"], regular["H_rare"], yerr=regular["H_rare_std"],
             fmt="o", ms=3, color="#4477AA", ecolor="#4477AA", alpha=0.4,
             elinewidth=0.6, capsize=0, zorder=2, label="Regular (n_total > 300)")
ax.errorbar(lowN["ts"], lowN["H_rare"], yerr=lowN["H_rare_std"],
             fmt="D", ms=7, color="#EE6677", ecolor="#EE6677",
             elinewidth=1.2, capsize=2, zorder=3,
             label="Low-n flagged (n_total <= 300, May)")

apr_may_boundary = pd.Timestamp("2026-05-01", tz="UTC")
ax.axvline(apr_may_boundary, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(apr_may_boundary, ax.get_ylim()[1], "  May ->", va="top", ha="left",
        color="gray", fontsize=9)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
fig.autofmt_xdate()
ax.set_ylabel("H_rare (bits, rarefied at B=703)")
ax.set_xlabel("Timestamp (UTC)")
ax.set_title("Rarefied lexical entropy H_rare over time\n" + CAVEAT_TXT, fontsize=9)
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIGDIR / "phase3_hrare_timeseries.png", dpi=150)
plt.close(fig)
print("Wrote phase3_hrare_timeseries.png")

# ── Figure 2: before/after panel, H_t vs n_total and H_rare vs n_total, by month ──
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

colors = {"2026-04": "#4477AA", "2026-05": "#EE6677"}
markers = {"2026-04": "o", "2026-05": "^"}

ax = axes[0]
for mo, sub in df.groupby("month"):
    ax.scatter(sub["n_total"], sub["H_t"], s=18, alpha=0.6,
               color=colors[mo], marker=markers[mo], label=mo)
ax.set_xlabel("n_total")
ax.set_ylabel("H_t (raw, as logged)")
ax.set_title("BEFORE: raw H_t vs n_total\n(r = 0.92 two-month, Pearson)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

ax = axes[1]
for mo, sub in df.groupby("month"):
    ax.scatter(sub["n_total"], sub["H_rare"], s=18, alpha=0.6,
               color=colors[mo], marker=markers[mo], label=mo)
    lowN_sub = sub[sub["n_total_regime"]]
    if len(lowN_sub):
        ax.scatter(lowN_sub["n_total"], lowN_sub["H_rare"], s=70,
                   facecolors="none", edgecolors="black", linewidths=1.2,
                   label="_nolegend_", zorder=5)
ax.set_xlabel("n_total")
ax.set_ylabel("H_rare (rarefied, B=703 bits)")
ax.set_title("AFTER: H_rare vs n_total\n(r = -0.18 two-month, Pearson; "
              "Spearman not significant)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

fig.suptitle("Rarefaction repair of the H_t / n_total confound\n" + CAVEAT_TXT, fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(FIGDIR / "phase3_hrare_vs_ntotal.png", dpi=150)
plt.close(fig)
print("Wrote phase3_hrare_vs_ntotal.png")
