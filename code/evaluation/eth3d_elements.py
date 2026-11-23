"""Element-level error vs ETH3D laser: what a tolerance check actually measures is the position of a planar
element (a wall, a slab face), not of single points. The laser scan is cut into CELL-sized cubes; every
cube whose laser points form a clean plane becomes one 'element'. For each element the reconstruction's
offset = median signed distance of its points (within GATE of the plane) to the laser plane.
U95 over elements -> verdict. usage: python eth3d_elements.py <gt_dir> <out_json> name=recon.ply ..."""
import json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from eth3d_dist import load_gt, xyz, BANDS
from eval_harness import tolerance_verdict

CELL, GATE, MIN_GT, MIN_REC, PLANAR_MM = 1.0, 0.25, 2000, 50, 5.0

def cells(p):
    k = np.floor(p / CELL).astype(np.int64)
    return k[:, 0] * 1_000_003 ** 2 + k[:, 1] * 1_000_003 + k[:, 2]

def main():
    gt_dir, out = sys.argv[1], sys.argv[2]
    g = load_gt(gt_dir); gk = cells(g)
    order = np.argsort(gk); gk, g = gk[order], g[order]
    u, st, cnt = np.unique(gk, return_index=True, return_counts=True)
    planes = {}
    for key, s0, c in zip(u, st, cnt):
        if c < MIN_GT:
            continue
        P = g[s0:s0 + c]; m = P.mean(0)
        _, sv, vt = np.linalg.svd(P - m, full_matrices=False)
        if sv[2] / np.sqrt(c) * 1000 < PLANAR_MM:          # rms out-of-plane < 5 mm: a clean planar element
            planes[key] = (m, vt[2])
    print(f"{len(planes)} planar 1 m elements in the laser scan")
    res = {}
    for arg in sys.argv[3:]:
        name, path = arg.split("=", 1)
        r = xyz(path); rk = cells(r)
        # group recon points by cell
        o = np.argsort(rk); rk_s, r_s = rk[o], r[o]
        ru, rs, rc = np.unique(rk_s, return_index=True, return_counts=True)
        idx = dict(zip(ru.tolist(), zip(rs.tolist(), rc.tolist())))
        e = []
        for key, (m, n) in planes.items():
            if key not in idx:
                continue
            s0, c = idx[key]
            d = (r_s[s0:s0 + c] - m) @ n
            d = d[np.abs(d) < GATE]
            if len(d) >= MIN_REC:
                e.append(np.median(d))
        e = np.abs(np.array(e))
        res[name] = {"elements_measured": int(len(e)), "elements_total": len(planes),
                     "E50_mm": float(np.median(e) * 1000), "U95_mm": float(np.percentile(e, 95) * 1000),
                     "verdicts": {k: tolerance_verdict(e, b)["verdict"] for k, b in BANDS.items()}}
        print(name, {k: v for k, v in res[name].items() if k != "verdicts"}, res[name]["verdicts"])
    json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
