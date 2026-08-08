import json, glob, os

OUT = "/zfsauton/scratch/yixiz/waymax_rs/es_baseline/experiments/out"
ONLINE = {"tf00000":"es_dense_sac_safe_online_tf00000_20260805_144414",
          "tf00001":"es_dense_sac_safe_online_tf00001_20260805_145535",
          "tf00002":"es_dense_sac_safe_online_tf00002_20260805_150706"}
OFFLINE = {"tf00000":"es_dense_sac_safe_tf00000_20260728_183523",
           "tf00001":"es_dense_sac_safe_tf00001_20260728_184226",
           "tf00002":"es_dense_sac_safe_tf00002_20260728_185335"}
FAIL = {"tf00000":{71,92,113,134,187,252,315,377},
        "tf00001":{45,62,147,158,200,243,289,360,463},
        "tf00002":{0,13,20,24,70,94,97}}
CTRL = {"tf00000":{0,1,2,3},"tf00001":{0,1,2,3},"tf00002":{1,2,3,4}}

def load_results(run_dir):
    res={}
    for f in glob.glob(os.path.join(OUT,run_dir,"**","*.scenario_*.json"),recursive=True):
        if os.path.basename(f)=="summary.json": continue
        d=json.load(open(f)); res[int(d["scenario_idx"])]=d
    return res
def load_args(run_dir):
    for f in glob.glob(os.path.join(OUT,run_dir,"**","diagnostics_*.json"),recursive=True):
        return json.load(open(f))["args"]

def corrected_online(shard):
    a=load_args(ONLINE[shard]); res=load_results(ONLINE[shard])
    req=[int(x) for x in str(a["scenario_indices"]).split(",")]; nw=int(a.get("num_worlds") or len(req))
    l2t={}
    for c in range(0,len(req),nw):
        b=req[c:c+nw]; s=sorted(b)
        for j,lbl in enumerate(b): l2t[lbl]=s[j]
    return {l2t.get(lbl,lbl):bool(d.get("success")) for lbl,d in res.items()}

print(f"{'shard':<8}{'scene':>6} {'kind':>5} {'off':>4} {'on':>4}   note")
tf,tof=0,0; cf,cof=0,0; conv=reg=0; ncf=ncc=0
for shard in ("tf00000","tf00001","tf00002"):
    off={k:bool(v.get("success")) for k,v in load_results(OFFLINE[shard]).items()}
    on=corrected_online(shard)
    for sc in sorted(set(off)&set(on)):
        kind="F" if sc in FAIL[shard] else ("C" if sc in CTRL[shard] else "?")
        o,n=off[sc],on[sc]
        note=""
        if n and not o: note="online WINS"; conv+=1
        elif o and not n: note="online LOSES"; reg+=1
        print(f"{shard:<8}{sc:>6} {kind:>5} {int(o):>4} {int(n):>4}   {note}")
        if kind=="F": ncf+=1; tf+=int(n); tof+=int(o)
        elif kind=="C": ncc+=1; cf+=int(n); cof+=int(o)
print("="*60)
print(f"FAILURE scenes (paired n={ncf}): offline {tof}  ->  online {tf}")
print(f"CONTROL scenes (paired n={ncc}): offline {cof}  ->  online {cf}")
print(f"online WINS (converted): {conv}    online LOSES (regressed): {reg}")
