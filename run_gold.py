"""Gold-pipeline verification runner for the FDP agent-correctness eval.

Executes each prompt's gold pipeline with the *documented* TokSearch Record API
(rec['k'] = v in-place; map funcs return nothing) and records reference outputs.

Path/API fixes folded in from the verification pass (see prompt_set.md):
  - Record has no update(); item assignment is the documented idiom.
  - MdsSignal is in `toksearch`, not `toksearch_d3d`.
  - connect_d3drdb is `toksearch_d3d.sql` (auto-sets TDSVER).
  - wmhd is not a PTDATA pointname -> EFIT MDS node \\wmhd (J).
  - `summary` IDS is unsupported -> P08 rescoped to core_profiles peak n_e.
  - diamag flux: magnetics.diamagnetic_flux.data (shape (1,N)).

Usage:
  cd toksearch_d3d && BEARER_TOKEN=$(cat ~/.fdp/<valid-token>) \
    pixi run fdp run python <repo>/papers/fdp_ai/eval/run_gold.py [--n N] [--only P01,P05]
"""
import os, sys, json, time, traceback
import numpy as np
from toksearch import Pipeline, MdsSignal
from toksearch_d3d import PtDataSignal, ImasSignal

N = 50
if "--n" in sys.argv:
    N = int(sys.argv[sys.argv.index("--n") + 1])
ONLY = None
if "--only" in sys.argv:
    ONLY = set(sys.argv[sys.argv.index("--only") + 1].split(","))

# Eval set: 50 plasma shots from 2024-06 (d3drdb), all carrying beta_normal.
# Sourced via source_2024_shots.py; locked here for reproducibility.
EVAL_ALL = [
    198873, 198877, 198878, 198879, 198880, 198881, 198882, 198883, 198884, 198885,
    198886, 198887, 198888, 198889, 198890, 198891, 198892, 198893, 198894, 198895,
    198896, 198897, 198898, 198899, 198908, 198909, 198910, 198911, 198912, 198913,
    198914, 198915, 198916, 198917, 198918, 198919, 198920, 198921, 198922, 198923,
    198925, 198926, 198927, 198928, 198929, 198930, 198931, 198932, 198933, 198934,
]
EVAL   = EVAL_ALL[:min(N, 50)]
EVAL_BN = EVAL[:min(N, 21)]   # P05 sub-range
EVAL10 = EVAL[:min(N, 10)]    # P18 sub-range
ONE = [198877]   # NBI-heated plasma shot (carries nbi + thomson + boundary)

def jsonable(o):
    if isinstance(o, dict):  return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer):  return int(o)
    return o

def _bn_total_pnbi(units):
    return np.nansum(np.stack([np.asarray(u, float) for u in units], axis=0), axis=0)

def _peak_nbi_mw(r):
    """Peak total NBI power in MW; 0.0 if the shot has no usable NBI (ohmic)."""
    if "pnbi" not in r.keys():
        return 0.0
    try:
        return float(np.nanmax(_bn_total_pnbi(r["pnbi"]["data"])) / 1e6)
    except Exception:
        return 0.0

# ---- A. PTDATA single signal / units / aggregation ---------------------
def P01(_):
    pipe = Pipeline(list(EVAL)); pipe.fetch("ip", PtDataSignal("ip"))
    def f(r): r["peak_ip_MA"] = float(np.nanmax(np.abs(r["ip"]["data"])) / 1e6)
    pipe.map(f); pipe.keep(["shot", "peak_ip_MA"])
    return {r["shot"]: r["peak_ip_MA"] for r in pipe.compute_multiprocessing(num_workers=16) if "peak_ip_MA" in r.keys()}

def P02(_):  # stored energy via EFIT MDS \wmhd (Joules)
    pipe = Pipeline(list(EVAL)); pipe.fetch("wmhd", MdsSignal(r"\wmhd", "efit01"))
    def f(r): r["peak_wmhd_MJ"] = float(np.nanmax(r["wmhd"]["data"]) / 1e6)
    pipe.map(f); pipe.keep(["shot", "peak_wmhd_MJ"])
    return {r["shot"]: r["peak_wmhd_MJ"] for r in pipe.compute_multiprocessing(num_workers=16) if "peak_wmhd_MJ" in r.keys()}

def P03(_):
    pipe = Pipeline(list(EVAL)); pipe.fetch("ip", PtDataSignal("ip"))
    def f(r):
        t = r["ip"]["times"]; i = int(np.argmin(np.abs(t - 2000.0)))
        r["ip_2000_MA"] = float(r["ip"]["data"][i] / 1e6)
    pipe.map(f); pipe.keep(["shot", "ip_2000_MA"])
    return {r["shot"]: r["ip_2000_MA"] for r in pipe.compute_multiprocessing(num_workers=16) if "ip_2000_MA" in r.keys()}

def P04(_):
    pipe = Pipeline(list(EVAL)); pipe.fetch("ip", PtDataSignal("ip"))
    def f(r):
        d = np.abs(r["ip"]["data"]) / 1e6; t = r["ip"]["times"]
        r["dur_ms"] = float(np.nansum(np.gradient(t)[d > 1.0]))
    pipe.map(f); pipe.keep(["shot", "dur_ms"])
    return {r["shot"]: r["dur_ms"] for r in pipe.compute_multiprocessing(num_workers=16) if "dur_ms" in r.keys()}

# ---- B. IMAS path discovery -------------------------------------------
def P05(_):
    pipe = Pipeline(list(EVAL_BN))
    pipe.fetch("bn", ImasSignal("equilibrium.time_slice.global_quantities.beta_normal"))
    def f(r): r["max_bn"] = float(np.nanmax(r["bn"]["data"]))
    pipe.map(f); pipe.where(lambda r: r.get("max_bn", 0.0) > 2.0); pipe.keep(["shot", "max_bn"])
    return {r["shot"]: r["max_bn"] for r in pipe.compute_multiprocessing(num_workers=16) if "max_bn" in r.keys()}

def P06(_):
    pipe = Pipeline(list(EVAL)); pipe.fetch("pnbi", ImasSignal("nbi.unit.power_launched.data"))
    def f(r): r["peak_pnbi_MW"] = _peak_nbi_mw(r)
    pipe.map(f); pipe.keep(["shot", "peak_pnbi_MW"])
    return {r["shot"]: r["peak_pnbi_MW"] for r in pipe.compute_multiprocessing(num_workers=16) if "peak_pnbi_MW" in r.keys()}

def P07(_):
    pipe = Pipeline(list(EVAL))
    pipe.fetch("q95", ImasSignal("equilibrium.time_slice.global_quantities.q_95"))
    def f(r): r["min_q95"] = float(np.nanmin(r["q95"]["data"]))
    pipe.map(f); pipe.keep(["shot", "min_q95"])
    return {r["shot"]: r["min_q95"] for r in pipe.compute_multiprocessing(num_workers=16) if "min_q95" in r.keys()}

def P08(_):  # rescoped: peak electron density (core_profiles); no line-avg IDS exists
    pipe = Pipeline(list(EVAL))
    pipe.fetch("ne", ImasSignal("core_profiles.profiles_1d.electrons.density_thermal"))
    def f(r): r["peak_ne"] = float(np.nanmax(r["ne"]["data"]))
    pipe.map(f); pipe.keep(["shot", "peak_ne"])
    return {r["shot"]: r["peak_ne"] for r in pipe.compute_multiprocessing(num_workers=16) if "peak_ne" in r.keys()}

def P09(_):  # diamagnetic flux (magnetics IDS); data shape (1, N)
    pipe = Pipeline(list(EVAL))
    pipe.fetch("dia", ImasSignal("magnetics.diamagnetic_flux.data"))
    def f(r): r["peak_diaflux"] = float(np.nanmax(np.abs(np.asarray(r["dia"]["data"]))))
    pipe.map(f); pipe.keep(["shot", "peak_diaflux"])
    return {r["shot"]: r["peak_diaflux"] for r in pipe.compute_multiprocessing(num_workers=16) if "peak_diaflux" in r.keys()}

# ---- C. SQL shot sourcing ---------------------------------------------
def _plasma_shots(lo, hi):
    from toksearch_d3d.sql import connect_d3drdb
    with connect_d3drdb() as conn:
        pipe = Pipeline.from_sql(conn,
            "SELECT s.shot FROM shots s JOIN shots_type t ON s.shot=t.shot "
            "WHERE t.shot_type='plasma' AND s.entered BETWEEN %s AND %s", lo, hi)
        return pipe, [rec["shot"] for rec in pipe.compute_multiprocessing(num_workers=16)]

def P10(_):
    _, shots = _plasma_shots("2024-06-01", "2024-06-08")
    return sorted(shots)

def P11(_):
    _, shots = _plasma_shots("2024-06-01", "2024-07-01")
    return {"n_plasma_june2024": len(shots)}

def P12(_):
    from toksearch_d3d.sql import connect_d3drdb
    with connect_d3drdb() as conn:
        pipe = Pipeline.from_sql(conn,
            "SELECT s.shot FROM shots s JOIN shots_type t ON s.shot=t.shot "
            "WHERE t.shot_type='plasma' AND s.entered BETWEEN %s AND %s",
            "2024-06-01", "2024-06-08")
        pipe.fetch("ip", PtDataSignal("ip"))
        def f(r): r["peak_ip_MA"] = float(np.nanmax(np.abs(r["ip"]["data"])) / 1e6)
        pipe.map(f); pipe.where(lambda r: r.get("peak_ip_MA", 0.0) > 1.2)
        pipe.keep(["shot", "peak_ip_MA"])
        return {r["shot"]: r["peak_ip_MA"] for r in pipe.compute_multiprocessing(num_workers=16) if "peak_ip_MA" in r.keys()}

# ---- D. MDSplus direct -------------------------------------------------
def P13(_):
    pipe = Pipeline(list(EVAL)); pipe.fetch("ipmhd", MdsSignal(r"\ipmhd", "efit01"))
    def f(r): r["peak_ipmhd_MA"] = float(np.nanmax(np.abs(r["ipmhd"]["data"])) / 1e6)
    pipe.map(f); pipe.keep(["shot", "peak_ipmhd_MA"])
    return {r["shot"]: r["peak_ipmhd_MA"] for r in pipe.compute_multiprocessing(num_workers=16) if "peak_ipmhd_MA" in r.keys()}

# ---- E. multi-signal / conditional ------------------------------------
def _bn_pnbi(shots):
    pipe = Pipeline(list(shots))
    pipe.fetch("bn", ImasSignal("equilibrium.time_slice.global_quantities.beta_normal"))
    pipe.fetch("pnbi", ImasSignal("nbi.unit.power_launched.data"))
    def f(r):
        if "bn" in r.keys():   r["max_bn"] = float(np.nanmax(r["bn"]["data"]))
        r["peak_pnbi_MW"] = _peak_nbi_mw(r)
    pipe.map(f); pipe.keep(["shot", "max_bn", "peak_pnbi_MW"])
    return list(pipe.compute_multiprocessing(num_workers=16))

def P14(_):
    return {r["shot"]: {"max_bn": r.get("max_bn", None), "peak_pnbi_MW": r.get("peak_pnbi_MW", None)}
            for r in _bn_pnbi(EVAL) if "max_bn" in r.keys()}

def P15(_):
    return {r["shot"]: {"max_bn": r.get("max_bn", None), "peak_pnbi_MW": r.get("peak_pnbi_MW", None)}
            for r in _bn_pnbi(EVAL) if r.get("max_bn", 0.0) > 3.0}

def P16(_):
    out = {}
    for r in _bn_pnbi(EVAL):
        bn, p = r.get("max_bn", None), r.get("peak_pnbi_MW", None)
        if bn is not None and p not in (None, 0.0):
            out[r["shot"]] = float(bn / p)
    return out

# ---- F. datasets / alignment ------------------------------------------
def P17(_):
    pipe = Pipeline(list(EVAL))
    pipe.fetch_dataset("ds", {"ip": PtDataSignal("ip"),
                              "wmhd": MdsSignal(r"\wmhd", "efit01")})
    pipe.align("ds", align_with=10.0, method="pad")
    def f(r):
        ds = r["ds"]; a = np.asarray(ds["ip"].values); b = np.asarray(ds["wmhd"].values)
        m = np.isfinite(a) & np.isfinite(b)
        r["corr"] = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float("nan")
    pipe.map(f); pipe.keep(["shot", "corr"])
    return {r["shot"]: r["corr"] for r in pipe.compute_multiprocessing(num_workers=16) if "corr" in r.keys()}

def P18(_):
    pipe = Pipeline(list(EVAL10))
    pipe.fetch_dataset("ds", {"ip": PtDataSignal("ip"),
                              "wmhd": MdsSignal(r"\wmhd", "efit01")})
    pipe.align("ds", align_with="ip", method="pad")
    def f(r): r["n_samples"] = int(np.asarray(r["ds"]["times"].values).size)
    pipe.map(f); pipe.keep(["shot", "n_samples"])
    return {r["shot"]: r["n_samples"] for r in pipe.compute_multiprocessing(num_workers=16) if "n_samples" in r.keys()}

# ---- G. channels / ragged ---------------------------------------------
def P19(_):
    pipe = Pipeline(list(ONE))
    pipe.fetch("te", ImasSignal("thomson_scattering.channel.t_e.data",
                                split_by="channel", dims={"times": "auto"}))
    def f(r): r["n_chan"] = int(len(r["te"]))
    pipe.map(f); pipe.keep(["shot", "n_chan"])
    return {r["shot"]: r["n_chan"] for r in pipe.compute_multiprocessing(num_workers=16) if "n_chan" in r.keys()}

def P20(_):
    pipe = Pipeline(list(ONE))
    pipe.fetch("bnd", ImasSignal("equilibrium.time_slice.boundary.outline.r"))
    def f(r):
        d = r["bnd"]["data"]; r["n_boundary_pts"] = int(len(d[-1]))
    pipe.map(f); pipe.keep(["shot", "n_boundary_pts"])
    return {r["shot"]: r["n_boundary_pts"] for r in pipe.compute_multiprocessing(num_workers=16) if "n_boundary_pts" in r.keys()}

def P21(_):
    pipe = Pipeline(list(ONE)); pipe.fetch("pnbi", ImasSignal("nbi.unit.power_launched.data"))
    def f(r): r["n_active"] = int(sum(np.nanmax(np.asarray(u, float)) > 0 for u in r["pnbi"]["data"]))
    pipe.map(f); pipe.keep(["shot", "n_active"])
    return {r["shot"]: r["n_active"] for r in pipe.compute_multiprocessing(num_workers=16) if "n_active" in r.keys()}

# ---- H. fault handling / aggregation ----------------------------------
def P22(_):
    pipe = Pipeline(list(EVAL)); pipe.fetch("ip", PtDataSignal("ip"))
    def f(r):
        d = r.get("ip", None)
        if d is not None and np.isfinite(d["data"]).any():
            r["peak_ip_MA"] = float(np.nanmax(np.abs(d["data"])) / 1e6)
    pipe.map(f); pipe.where(lambda r: r.get("peak_ip_MA", None) is not None)
    pipe.keep(["shot", "peak_ip_MA"])
    recs = pipe.compute_multiprocessing(num_workers=16)
    return {"n_valid": len(recs),
            "mean_peak_ip_MA": float(np.mean([r["peak_ip_MA"] for r in recs])) if recs else None}

def P23(_):
    pipe = Pipeline(list(EVAL))
    pipe.fetch("bn", ImasSignal("equilibrium.time_slice.global_quantities.beta_normal"))
    def f(r): r["max_bn"] = float(np.nanmax(r["bn"]["data"]))
    pipe.map(f); pipe.keep(["shot", "max_bn"])
    vals = [r["max_bn"] for r in pipe.compute_multiprocessing(num_workers=16) if "max_bn" in r.keys()]
    return {"median_max_bn": float(np.nanmedian(vals)) if vals else None, "n": len(vals)}

# Dropped: P07 (q95 COCOS-negative + edge artifact), P09 (diamag flux placeholder).
REG = {f"P{i:02d}": globals()[f"P{i:02d}"] for i in range(1, 24) if i not in (7, 9)}

def main():
    results = {}
    for pid, fn in REG.items():
        if ONLY and pid not in ONLY:
            continue
        t0 = time.time()
        try:
            out = jsonable(fn(None))
            n = len(out) if isinstance(out, (dict, list)) else 1
            results[pid] = {"ok": True, "n": n, "result": out, "secs": round(time.time() - t0, 1)}
            print(f"{pid}: OK  n={n}  {results[pid]['secs']}s")
        except Exception as e:
            results[pid] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                            "tb": traceback.format_exc()[-700:], "secs": round(time.time() - t0, 1)}
            print(f"{pid}: FAIL  {type(e).__name__}: {str(e)[:150]}")
    outpath = os.path.join(os.path.dirname(__file__), "gold_references.json")
    with open(outpath, "w") as f:
        json.dump({"eval_n": N, "results": results}, f, indent=2)
    print(f"\nwrote {outpath}")
    print(f"summary: {sum(1 for v in results.values() if v['ok'])}/{len(results)} prompts ran")

if __name__ == "__main__":
    main()
