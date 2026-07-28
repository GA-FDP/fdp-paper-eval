"""Source a 2024 plasma-shot eval set and verify signal coverage + beta_N range."""
import numpy as np, pandas as pd
from toksearch import Pipeline
from toksearch_d3d import PtDataSignal, ImasSignal
from toksearch_d3d.sql import connect_d3drdb

# first ~60 plasma shots from 2024-06-01
with connect_d3drdb() as conn:
    df = pd.read_sql(
        "SELECT TOP 60 s.shot FROM shots s JOIN shots_type t ON s.shot=t.shot "
        "WHERE t.shot_type='plasma' AND s.entered >= '2024-06-01' "
        "ORDER BY s.shot", conn)
cand = [int(s) for s in df["shot"]]
print(f"candidate plasma shots: {len(cand)}  {cand[:5]}..{cand[-3:]}")

pipe = Pipeline(list(cand))
pipe.fetch("ip", PtDataSignal("ip"))
pipe.fetch("bn", ImasSignal("equilibrium.time_slice.global_quantities.beta_normal"))
def f(r):
    if "ip" in r.keys(): r["peak_ip_MA"] = float(np.nanmax(np.abs(r["ip"]["data"]))/1e6)
    if "bn" in r.keys():
        b = r["bn"]["data"]
        if np.isfinite(b).any(): r["max_bn"] = float(np.nanmax(b))
pipe.map(f); pipe.keep(["shot","peak_ip_MA","max_bn"])
recs = pipe.compute_multiprocessing(num_workers=16)

has_ip = [r["shot"] for r in recs if "peak_ip_MA" in r.keys()]
has_bn = [r["shot"] for r in recs if "max_bn" in r.keys()]
bns = [r["max_bn"] for r in recs if "max_bn" in r.keys()]
ips = [r["peak_ip_MA"] for r in recs if "peak_ip_MA" in r.keys()]
print(f"have ip: {len(has_ip)}  have beta_normal: {len(has_bn)}")
if ips: print(f"peak Ip MA: {min(ips):.2f}..{max(ips):.2f}; n>1MA={sum(v>1 for v in ips)}")
if bns: print(f"max beta_N: {min(bns):.2f}..{max(bns):.2f}; n>2={sum(b>2 for b in bns)}; n>3={sum(b>3 for b in bns)}")

# choose 50 shots that carry beta_normal (so equilibrium prompts populate)
chosen = sorted(set(has_bn))[:50]
print(f"\nCHOSEN ({len(chosen)}): {chosen}")
