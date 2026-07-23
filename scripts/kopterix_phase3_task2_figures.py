"""Figures for Phase 3 Task 2. Recomputes lightweight ACF/fits from stashed grids."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

BASE=Path(__file__).resolve().parents[1]
FIG=BASE/"phase3"/"figures"; LOGS=BASE/"phase3"/"logs"; TAB=BASE/"phase3"/"tables"
DT=6.0
mem=pd.read_csv(TAB/"phase3_memory_timescales.csv",comment="#")

def nan_acf(x,maxlag):
    x=np.asarray(x,float); rho=np.full(maxlag+1,np.nan); rho[0]=1.0
    for k in range(1,maxlag+1):
        a,b=x[:-k],x[k:]; m=~np.isnan(a)&~np.isnan(b)
        if m.sum()>=5 and np.std(a[m])>0 and np.std(b[m])>0:
            rho[k]=np.corrcoef(a[m],b[m])[0,1]
    return rho
def detrend(x):
    x=np.asarray(x,float); idx=np.arange(len(x)); m=~np.isnan(x)
    b1,b0=np.polyfit(idx[m],x[m],1); return x-(b0+b1*idx)
def fit_exp(rho,dt,lags=range(1,17)):
    ks=np.array([k for k in lags if not np.isnan(rho[k])]); ys=rho[ks]
    popt,_=curve_fit(lambda k,tau:np.exp(-k*dt/tau),ks,ys,p0=[24],bounds=(0.1,1e5))
    return popt[0]

C=np.load(LOGS/"C_series.npz")
g_phi=C["g_phi"]; g_D=C["g_D"]; g_lex=C["g_lex"]

# ── FIG 1: ACF + fitted decay, one panel per series (detrended phi & D, raw lexical) ──
series=[("phi_t (detrended)",detrend(g_phi),True),
        ("D_t (detrended)",detrend(g_D),True),
        ("lexical cos-step",g_lex,False)]
fig,axes=plt.subplots(1,3,figsize=(15,4.2))
for ax,(name,s,detr) in zip(axes,series):
    rho=nan_acf(s,28); ks=np.arange(29)
    ax.axhline(0,color="grey",lw=0.6)
    ax.plot(ks,rho,"o",ms=4,color="#2c3e50",label="ACF")
    tau=fit_exp(rho,DT)
    kk=np.linspace(1,16,100); ax.plot(kk,np.exp(-kk*DT/tau),"-",color="#e74c3c",
        label=f"exp fit, tau={tau:.1f}h")
    ax.axvline(16,ls=":",color="grey",lw=0.8)
    ax.set_title(f"{name}\n(dt={DT:.0f}h grid; fit lags 1-16)"); ax.set_xlabel("lag k (x6h)")
    ax.set_ylabel("autocorrelation"); ax.legend(fontsize=8); ax.set_ylim(-0.3,1.05)
fig.suptitle("Phase 3 Task 2 — ACF and fitted exponential decay (tau in hours)",y=1.02)
fig.tight_layout(); fig.savefig(FIG/"phase3_task2_acf_fits.png",dpi=130,bbox_inches="tight"); plt.close(fig)

# ── FIG 2: per-layer tau bar chart with CI (log scale; wide upper CIs) ──
lyr=mem[mem["series"].str.startswith("layer_")].copy()
lyr["layer"]=lyr["series"].str.replace("layer_","").str.replace("_cosine_decay","")
order=["surface","mid","residue"]; lyr=lyr.set_index("layer").loc[order]
fig,ax=plt.subplots(figsize=(6.2,4.6))
x=np.arange(3); tau=lyr["tau_hours"].values
lo=tau-lyr["ci95_lo"].values; hi=lyr["ci95_hi"].values-tau
ax.bar(x,tau,color=["#3498db","#9b59b6","#e67e22"],width=0.6,alpha=0.85)
ax.errorbar(x,tau,yerr=[lo,hi],fmt="none",ecolor="black",capsize=5,lw=1.2)
ax.axhline(24,ls="--",color="red",label="24h feed-age window")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(order)
ax.set_ylabel("tau (hours, log scale)")
ax.set_title("Per-layer centroid-memory tau\n(point est ~166-193h; 95% CI right-skewed/wide)")
for xi,t in zip(x,tau): ax.text(xi,t*1.05,f"{t:.0f}h",ha="center",fontsize=9)
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(FIG/"phase3_task2_perlayer_tau.png",dpi=130,bbox_inches="tight"); plt.close(fig)

# ── FIG 3: residual angle distribution vs null overlay ──
B=np.load(LOGS/"B_angle_arrays.npz")
fig,ax=plt.subplots(figsize=(8,4.6))
bins=np.linspace(0.6,2.8,80)
ax.hist(B["null1_all"],bins=bins,density=True,alpha=0.5,color="#95a5a6",label="Null1 (Gaussian, rescaled)")
ax.hist(B["null2_all"],bins=bins,density=True,alpha=0.5,color="#f1c40f",label="Null2 (label perm)")
ax.hist(B["obs_all"],bins=bins,density=True,alpha=0.65,color="#2c3e50",label="Observed residual angles")
ax.axvline(np.pi/2,color="red",ls="--",label="pi/2 = 90deg (HD null)")
ax.axvline(2*np.pi/3,color="green",ls=":",label="120deg (sum-to-zero constraint)")
ax.set_xlabel("pairwise angle (radians)"); ax.set_ylabel("density")
ax.set_title("Residual pairwise angles vs high-dimensional null\n"
             "observed pinned near 120deg (residuals sum to zero), NOT near 90deg")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(FIG/"phase3_task2_angle_null_overlay.png",dpi=130,bbox_inches="tight"); plt.close(fig)

# ── FIG 4: phi_t and D_t time series (detrended) with changepoints ──
D=np.load(LOGS/"D_series.npz")
phi_ts=pd.to_datetime(D["phi_ts"]); D_ts=pd.to_datetime(D["D_ts"])
cps_phi=pd.to_datetime(D["cps_phi"]); cps_D=pd.to_datetime(D["cps_D"])
dt_out=pd.to_datetime(D["dt_out_ts"])
fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True)
axes[0].plot(phi_ts,D["phi_detr"],color="#2c3e50",lw=0.9)
for cp in cps_phi: axes[0].axvline(cp,color="#e74c3c",ls="--",lw=1)
axes[0].set_ylabel("phi_t (detrended)")
axes[0].set_title(f"phi_t detrended with PELT changepoints (n={len(cps_phi)}, BIC penalty)")
axes[1].plot(D_ts,D["D_detr"],color="#16a085",lw=0.9)
for cp in cps_D: axes[1].axvline(cp,color="#e74c3c",ls="--",lw=1)
for o in dt_out: axes[1].axvline(o,color="orange",ls=":",lw=0.7,alpha=0.6)
axes[1].set_ylabel("D_t (detrended)")
axes[1].set_title(f"D_t detrended with PELT changepoints (red dashed, n={len(cps_D)}) "
                  f"and robust-z outliers (orange dotted)")
axes[1].set_xlabel("date (UTC)")
fig.tight_layout(); fig.savefig(FIG/"phase3_task2_changepoints_timeseries.png",dpi=130,bbox_inches="tight"); plt.close(fig)
print("wrote 4 figures to",FIG)
for p in sorted(FIG.glob("phase3_task2_*.png")): print(" ",p.name)
