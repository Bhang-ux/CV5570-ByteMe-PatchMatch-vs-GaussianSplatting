"""Per-point error distributions vs ETH3D laser scans (complements the official evaluator's fractions).
Accuracy distances: reconstruction -> laser (each recon point to nearest GT point).
Only points within MAX_D of the laser are kept (proxy for ETH3D's observed-space mask); the share
dropped is reported. Gives E50 / U95 for the tolerance verdict.
usage: python eth3d_dist.py <gt_dir> <out_json> name=recon.ply [name=recon.ply ...]"""
import json, re, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from eval_harness import read_ply, voxel_down, tolerance_verdict

MAX_D, N_SAMPLE, GT_VOX = 0.5, 300_000, 0.005
BANDS = {"ACI117 section <=300mm (+10/-6)": (-6, 10), "IS456 section (+12/-6)": (-6, 12),
         "ACI117 section 300-900mm (+13/-10)": (-10, 13), "ACI117 opening (+/-13)": (-13, 13),
         "ACI117 plumb (+/-25)": (-25, 25)}

def xyz(p):
    r = read_ply(p); r = r[0] if isinstance(r, tuple) else r
    return np.asarray(r)[:, :3].astype(np.float64)

def load_gt(gt_dir):
    txt = open(f"{gt_dir}/scan_alignment.mlp").read()
    pts = []
    for fn, mat in re.findall(r'filename="([^"]+)".*?<MLMatrix44>(.*?)</MLMatrix44>', txt, re.S):
        T = np.array(mat.split(), float).reshape(4, 4)
        x = xyz(f"{gt_dir}/{fn}")
        x = voxel_down(x, GT_VOX); x = x[0] if isinstance(x, tuple) else x
        pts.append(x @ T[:3, :3].T + T[:3, 3])
    g = np.concatenate(pts)
    g = voxel_down(g, GT_VOX); return g[0] if isinstance(g, tuple) else g

def main():
    gt_dir, out = sys.argv[1], sys.argv[2]
    tree = cKDTree(load_gt(gt_dir))
    res = {}
    for arg in sys.argv[3:]:
        name, path = arg.split("=", 1)
        x = xyz(path)
        if len(x) > N_SAMPLE:
            x = x[np.random.default_rng(0).choice(len(x), N_SAMPLE, replace=False)]
        d, _ = tree.query(x, distance_upper_bound=MAX_D, workers=-1)
        near = np.isfinite(d); e = d[near]
        r = {"n_eval": int(near.sum()), "share_beyond_50cm": float(1 - near.mean()),
             "E50_mm": float(np.median(e) * 1000), "U95_mm": float(np.percentile(e, 95) * 1000),
             "verdicts": {k: tolerance_verdict(e, b)["verdict"] for k, b in BANDS.items()}}
        res[name] = r
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "verdicts"})
    json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
