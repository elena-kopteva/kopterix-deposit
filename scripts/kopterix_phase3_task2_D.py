"""
Kopterix Phase 3 Task 2 -- Analysis D (changepoints) + per-layer tau CI fix.

Run standalone after A/B/C (which already wrote their tables). This script:
  (1) Re-estimates per-layer memory tau with a moving-block bootstrap that PRESERVES
      original adjacency (the earlier block bootstrap scrambled adjacency, making the
      CI meaningless). Overwrites only the 3 per-layer rows of
      phase3_memory_timescales.csv.
  (2) Appends the algebraic finding (MeanEmbedding == mean of the 3 layer centroids =>
      residuals sum to zero => angles pinned near 120deg) to the B table note.
  (3) Runs Analysis D: PELT/rbf changepoints on detrended phi_t and D_t.
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import ruptures as rpt

BASE   = Path(__file__).resolve().parents[1]
TABLES = BASE / "phase3" / "tables"
LOGS   = BASE / "phase3" / "logs"
WS = pd.Timestamp("2026-04-01", tz="UTC"); WE = pd.Timestamp("2026-06-01", tz="UTC")
SEED = 42; LAYERS = ["surface","mid","residue"]
log=[]
def L(m): print(m); log.append(str(m))

# ── load ──
state = pd.read_csv(BASE/"kopterix_state.csv")
state["ts"]=pd.to_datetime(state["Timestamp"],utc=True)
state=state.sort_values("ts").reset_index(drop=True)
def parse_layers(s):
    if pd.isna(s): return None
    try:
        d=json.loads(s); out={}
        for l in LAYERS:
            v=d.get(l)
            if v is None: return None
            a=np.asarray(v,float)
            if a.size!=384 or not np.all(np.isfinite(a)) or np.allclose(a,0): return None
            out[l]=a
        return out
    except: return None
state["lc"]=state["LayerCentroids"].apply(parse_layers)
inw=(state["ts"]>=WS)&(state["ts"]<WE)
slc=state[inw & state["lc"].notna()].copy().sort_values("ts").reset_index(drop=True)
gaps=slc["ts"].diff().dt.total_seconds().values/3600.0
median_gap=float(np.nanmedian(gaps[1:]))
n=len(slc); BIG=12.0
L(f"per-layer rows: {n}, median gap {median_gap:.3f}h")

# ok_pair[i,k] = every consecutive gap in (i, i+k] < BIG
def build_ok():
    ok=np.zeros((n,17),bool)
    for i in range(n):
        for k in range(1,17):
            if i+k>=n: break
            seg=gaps[i+1:i+k+1]
            ok[i,k]= (not np.any(np.isnan(seg))) and (not np.any(seg>BIG))
    return ok
OK=build_ok()

# ── (1) per-layer tau via similarity matrices ──
def Smat(layer):
    V=np.array([slc.iloc[i]["lc"][layer] for i in range(n)])
    Vn=V/np.linalg.norm(V,axis=1,keepdims=True)
    return Vn@Vn.T
def sim_curve(S, idx_blocks=None):
    """mean cosine sim per lag k=1..16 over original-adjacent valid pairs.
    idx_blocks: list of (start,length) contiguous blocks (for bootstrap); None=full."""
    num=np.zeros(17); den=np.zeros(17)
    if idx_blocks is None:
        for i in range(n):
            for k in range(1,17):
                if i+k<n and OK[i,k]:
                    num[k]+=S[i,i+k]; den[k]+=1
    else:
        for (s0,Lb) in idx_blocks:
            for i in range(s0, s0+Lb):
                for k in range(1,17):
                    j=i+k
                    if j< s0+Lb and j<n and OK[i,k]:
                        num[k]+=S[i,j]; den[k]+=1
    ks=np.array([k for k in range(1,17) if den[k]>0])
    ys=np.array([num[k]/den[k] for k in ks])
    return ks,ys
def fit_tau(ks,ys,dt):
    def model(k,A,tau,C): return A*np.exp(-k*dt/tau)+C
    popt,_=curve_fit(model,ks,ys,p0=[ys[0]-ys[-1],24.0,ys[-1]],
                     bounds=([0,0.1,-1],[2,1e5,1]),maxfev=20000)
    resid=ys-model(ks,*popt); ss=np.sum(resid**2); st=np.sum((ys-ys.mean())**2)
    return float(popt[1]), (1-ss/st if st>0 else np.nan)
# Corrected per-layer exponential-fit bootstrap: adjacency-preserving, NON-CIRCULAR
# moving blocks of 24 rows, 5000 resamples, valid starts 0..n-Lb INCLUSIVE (no wrap),
# lagged pairs only within a sampled original block. Block length 24 supplies within-
# block pairs through lag 23, so every replicate can fit the full lags 1-16 the point
# estimate uses. (The earlier block=16, 500-resample bootstrap reached only lag 15.)
PERLAYER_SEEDS={"surface":42,"mid":43,"residue":44}
def boot_ci(S,dt,n_boot=5000,Lb=24,seed=SEED):
    # Per-layer exponential-fit CI computed with the AUTHORITATIVE repair implementation
    # (kopterix_phase3_task2_perlayer_ci.py :: boot_ci_exp). Vectorized within-block
    # pooling: for each lag k precompute prefix sums over the ORIGINAL index i of the
    # adjacency-valid similarity S[i,i+k] and the pair count; a sampled block [s0,s0+Lb)
    # then contributes, at lag k, the pooled sum/count over i in [s0, s0+Lb-k) via one
    # prefix-sum difference. This is the SAME set of pairs (within-block only, original
    # adjacency mask, no wrap-around) as the earlier explicit nested loop, but it follows
    # the repair script's exact summation order, so the pooled per-lag values -- and hence
    # the percentile CIs -- reproduce the authoritative intervals to the bit. Seed usage,
    # block sampling (starts uniform on 0..n-Lb inclusive, size=ceil(n/Lb)) and the
    # discard-on-empty-lag rule are unchanged from the previous implementation.
    csum_s=np.zeros((17,n+1)); csum_d=np.zeros((17,n+1))
    for k in range(1,17):
        sval=np.zeros(n); dval=np.zeros(n)
        for i in range(n-k):
            if OK[i,k]: sval[i]=S[i,i+k]; dval[i]=1.0
        csum_s[k,1:]=np.cumsum(sval); csum_d[k,1:]=np.cumsum(dval)
    rng=np.random.default_rng(seed); nb=int(np.ceil(n/Lb)); max_start=n-Lb
    ksall=np.arange(1,17); taus=[]
    for _ in range(n_boot):
        starts=rng.integers(0,max_start+1,size=nb)   # 0..n-Lb inclusive, non-circular
        ys=np.empty(16); ok_all=True
        for k in range(1,17):
            hi=starts+(Lb-k); lo=starts       # within-block: i in [s0, s0+Lb-k)
            num=float(np.sum(csum_s[k,hi]-csum_s[k,lo])); den=float(np.sum(csum_d[k,hi]-csum_d[k,lo]))
            if den>0: ys[k-1]=num/den
            else: ok_all=False; break         # this lag has no within-block pair -> discard
        if not ok_all: continue               # require all 16 fitted lags present
        try:
            t,_=fit_tau(ksall,ys,dt)
            if np.isfinite(t): taus.append(t)
        except: pass
    taus=np.array(taus)
    return (np.nanpercentile(taus,2.5),np.nanpercentile(taus,97.5)) if len(taus)>10 else (np.nan,np.nan)

per_layer={}
for layer in LAYERS:
    S=Smat(layer)
    ks,ys=sim_curve(S); tau,r2=fit_tau(ks,ys,median_gap)
    ci=boot_ci(S,median_gap,seed=PERLAYER_SEEDS[layer])
    per_layer[layer]=(tau,r2,ci,ys[0])
    L(f"  {layer}: tau={tau:.2f}h R2={r2:.3f} CI[{ci[0]:.2f},{ci[1]:.2f}] vs24h={tau/24:.2f} "
      f"{'EXCEEDS' if tau>24 else 'below'}")
L(f"  tau_residue({per_layer['residue'][0]:.1f}) > tau_surface({per_layer['surface'][0]:.1f})? "
  f"{per_layer['residue'][0]>per_layer['surface'][0]}")

# patch the 3 per-layer rows in the memory table (preserve header + other rows)
mpath=TABLES/"phase3_memory_timescales.csv"
raw=mpath.read_text().splitlines()
hdr=[l for l in raw if l.startswith("#")]
body=[l for l in raw if not l.startswith("#")]
mdf=pd.read_csv(mpath, comment="#")
for layer in LAYERS:
    tau,r2,ci,rho1=per_layer[layer]
    m=mdf["series"]==f"layer_{layer}_cosine_decay"
    mdf.loc[m,"dt_hours"]=median_gap
    mdf.loc[m,"rho1"]=rho1
    mdf.loc[m,"tau_hours"]=tau
    mdf.loc[m,"ci95_lo"]=ci[0]; mdf.loc[m,"ci95_hi"]=ci[1]
    mdf.loc[m,"exp_R2"]=r2
    mdf.loc[m,"tau_vs_24h"]=tau/24.0
    mdf.loc[m,"exceeds_24h"]=bool(tau>24)
with open(mpath,"w") as f:
    f.write("\n".join(hdr)+"\n")
    f.write("# NOTE (per-layer CI corrected, exponential fits): the three raw per-layer "
            "exponential 95% CIs (series layer_surface/mid/residue_cosine_decay) now come "
            "from an ADJACENCY-PRESERVING, NON-CIRCULAR moving-block bootstrap with block "
            "length 24 rows, 5000 resamples, exponential fitting over lags 1-16, lagged "
            "pairs formed only WITHIN a sampled original block (original adjacency preserved, "
            "no bridging gaps>12h; no pairs across block boundaries and no wrap-around), and "
            "layer-specific seeds 42/43/44 for surface/mid/residue. This supersedes the "
            "earlier block=16, 500-resample per-layer bootstrap (which reached only lag 15).\n")
    mdf.to_csv(f,index=False)
L(f"patched per-layer rows in {mpath}")

# ── (2) append sum-to-zero finding to B table note ──
bpath=TABLES/"phase3_residual_angle_null.csv"
btxt=bpath.read_text()
if "sum to zero" not in btxt:
    lines=btxt.splitlines()
    ins=("# MECHANISM: MeanEmbedding(t) is EXACTLY the equal-weight mean of the three "
         "layer centroids (||MeanEmbedding - (c_S+c_M+c_R)/3|| = 0 to machine precision), "
         "so r_S+r_M+r_R = 0 by construction. Three equal-norm vectors summing to zero "
         "have pairwise angle arccos(-1/2)=120deg=2.094 rad; observed all-pairs mean is "
         "2.094 rad. The near-orthogonality framing is therefore an algebraic artifact of "
         "the residual definition, not a geometric finding, and the angles are DISPLACED "
         "(to ~120deg), not tight, relative to the high-dimensional ~90deg null.")
    # insert after the first two comment lines
    out=[]; done=False
    for i,l in enumerate(lines):
        out.append(l)
        if l.startswith("# CONCLUSION") and not done:
            out.append(ins); done=True
    if not done: out.insert(1,ins)
    bpath.write_text("\n".join(out)+"\n")
    L("appended sum-to-zero mechanism note to B table")

# ── (3) Analysis D: PELT changepoints ──
real=pd.read_csv(BASE/"analysis_intermediate"/"real_clean.csv")
real["ts"]=pd.to_datetime(real["Timestamp_UTC"].str.replace(" UTC","",regex=False),utc=True)
real=real.sort_values("ts").reset_index(drop=True)

def detrend(ts,vals):
    df=pd.DataFrame({"ts":ts,"v":vals}).dropna().sort_values("ts").reset_index(drop=True)
    idx=np.arange(len(df)); b1,b0=np.polyfit(idx,df["v"].values,1)
    df["v_detr"]=df["v"].values-(b0+b1*idx); return df

def pelt(df,name):
    sig=df["v_detr"].values.reshape(-1,1); m=len(sig)
    algo=rpt.Pelt(model="rbf",min_size=3,jump=1).fit(sig)
    pens=np.geomspace(0.2,200,60); seen={}
    for pen in pens:
        bk=algo.predict(pen=pen); K=len(bk)-1
        if K not in seen:
            cost=algo.cost.sum_of_costs(bk)
            seen[K]={"pen":pen,"bic":m*np.log(cost/m+1e-12)+(K+1)*np.log(m),"bk":bk}
    bestK=min(seen,key=lambda k:seen[k]["bic"]); pstar=seen[bestK]["pen"]
    L(f"[{name}] n={m} BIC K={bestK} pen~{pstar:.3g}")
    sens={}
    for mult,ml in [(1/np.sqrt(3),"div_sqrt3"),(1.0,"pen"),(np.sqrt(3),"mul_sqrt3")]:
        bk=algo.predict(pen=pstar*mult)
        cps=[df.iloc[b-1]["ts"] for b in bk[:-1]]; sens[ml]=(pstar*mult,len(cps),cps)
        L(f"    {ml} pen={pstar*mult:.3g}: {len(cps)} cps -> {[str(c)[:16] for c in cps]}")
    cps_main=[df.iloc[b-1]["ts"] for b in seen[bestK]["bk"][:-1]]
    return pstar,bestK,cps_main,sens

dfp=detrend(real["ts"],real["phi_t"].values)
dfd=detrend(real["ts"],real["D_t"].values)
pp,Kp,cps_phi,sp=pelt(dfp,"phi_t_detrended")
pd_,Kd,cps_D,sd=pelt(dfd,"D_t_detrended")

# D_t robust-z outlier clusters
dv=real[["ts","D_t"]].dropna().sort_values("ts").reset_index(drop=True)
med=dv["D_t"].median(); mad=stats.median_abs_deviation(dv["D_t"],scale="normal")
dv["rz"]=(dv["D_t"]-med)/mad; dv["out"]=dv["rz"].abs()>3.5
oi=np.where(dv["out"].values)[0]; clusters=[]
if len(oi):
    cur=[oi[0]]
    for j in oi[1:]:
        if j==cur[-1]+1: cur.append(j)
        else: clusters.append(cur); cur=[j]
    clusters.append(cur)
n_in_consec=sum(len(c) for c in clusters if len(c)>=2)
spans=[(str(dv.iloc[c[0]]["ts"]),str(dv.iloc[c[-1]]["ts"]),len(c)) for c in clusters]
L(f"\nD_t robust-z |z|>3.5: {int(dv['out'].sum())} outliers; "
  f"{sum(1 for c in clusters if len(c)>=2)} consecutive runs covering {n_in_consec}")
def near(cp):
    cp=pd.Timestamp(cp)
    for c in clusters:
        t0=pd.Timestamp(dv.iloc[c[0]]["ts"]); t1=pd.Timestamp(dv.iloc[c[-1]]["ts"])
        if t0-pd.Timedelta("12h")<=cp<=t1+pd.Timedelta("12h"): return f"{t0}..{t1}(len{len(c)})"
    return ""

cp_rows=[]
for name,cps,p,sens,K in [("phi_t_detrended",cps_phi,pp,sp,Kp),("D_t_detrended",cps_D,pd_,sd,Kd)]:
    for cp in cps:
        cp_rows.append({"series":name,"changepoint_utc":str(cp),"penalty_BIC":p,
            "n_cp_at_BIC":K,"n_cp_div_sqrt3":sens["div_sqrt3"][1],"n_cp_pen":sens["pen"][1],
            "n_cp_mul_sqrt3":sens["mul_sqrt3"][1],"near_Dt_outlier_cluster":near(cp)})
cp_df=pd.DataFrame(cp_rows)
with open(TABLES/"phase3_changepoints.csv","w") as f:
    f.write("# PELT changepoints (ruptures, model='rbf', min_size=3, jump=1). Penalty "
            "selected by BIC-style sweep over segment count K; sensitivity reported over a "
            "3x penalty range (pen/sqrt3, pen, pen*sqrt3). Run on LINEARLY DETRENDED phi_t "
            "and D_t (observed series, NaN dropped, time-ordered). changepoint_utc = "
            "timestamp of the last run before each break. near_Dt_outlier_cluster flags "
            "coincidence (within 12h) with a robust-z (|z|>3.5, median/MAD) D_t outlier cluster.\n")
    cp_df.to_csv(f,index=False)
L(f"wrote phase3_changepoints.csv ({len(cp_df)} cps)")

# markdown log cross-ref
obs=[]
for mf in ["kopterix_log_2026-04.md","kopterix_log_2026-05.md"]:
    p=BASE/mf
    if not p.exists(): continue
    for blk in re.split(r"\n## obs :: ", p.read_text(encoding="utf-8"))[1:]:
        mh=re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s*UTC",blk)
        if not mh: continue
        ts=pd.Timestamp(mh.group(1),tz="UTC")
        mood=re.search(r"\*\*surface mood:\*\*\s*(.+)",blk)
        flag=re.search(r"\*\*anomaly flag:\*\*\s*(\w+)",blk)
        obs.append({"ts":ts,"mood":mood.group(1).strip() if mood else "",
                    "flag":flag.group(1).strip() if flag else ""})
olog=pd.DataFrame(obs).sort_values("ts").reset_index(drop=True)
L(f"parsed {len(olog)} obs-log entries")
xr=[]
for name,cps in [("phi_t_detrended",cps_phi),("D_t_detrended",cps_D)]:
    for cp in cps:
        cp=pd.Timestamp(cp)
        w=olog[(olog["ts"]>=cp-pd.Timedelta("12h"))&(olog["ts"]<=cp+pd.Timedelta("12h"))]
        if len(w)==0:
            xr.append({"series":name,"changepoint_utc":str(cp),"log_ts":"","delta_h":np.nan,
                       "surface_mood":"(no log entry within +/-12h)","anomaly_flag":""})
        for _,ww in w.iterrows():
            xr.append({"series":name,"changepoint_utc":str(cp),"log_ts":str(ww["ts"]),
                       "delta_h":(ww["ts"]-cp).total_seconds()/3600,
                       "surface_mood":ww["mood"],"anomaly_flag":ww["flag"]})
xr_df=pd.DataFrame(xr)
with open(TABLES/"phase3_changepoint_crossref.csv","w") as f:
    f.write("# Alignment (NOT interpretation) of detected changepoints with observation-log "
            "entries within +/-12h; 'surface mood' and 'anomaly flag' quoted verbatim.\n")
    f.write(f"# D_t robust-z outlier clusters (|z|>3.5): {spans}\n")
    xr_df.to_csv(f,index=False)
L(f"wrote phase3_changepoint_crossref.csv ({len(xr_df)} rows)")

np.savez(LOGS/"D_series.npz",
         phi_ts=dfp["ts"].view("int64").values, phi_detr=dfp["v_detr"].values,
         D_ts=dfd["ts"].view("int64").values, D_detr=dfd["v_detr"].values,
         cps_phi=np.array([pd.Timestamp(c).value for c in cps_phi]),
         cps_D=np.array([pd.Timestamp(c).value for c in cps_D]),
         dt_out_ts=dv[dv["out"]]["ts"].view("int64").values)
(LOGS/"task2_D_log.txt").write_text("\n".join(log))
L("=== D + per-layer CI fix DONE ===")
