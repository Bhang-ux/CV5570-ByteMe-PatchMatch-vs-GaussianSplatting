"""Self-test: every harness function must recover KNOWN answers before we trust it on real data.
Uses the phase-1 dense cloud (pass its path as the first argument; default data/dense.ply) as the test object."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from eval_harness import *

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
rng = np.random.default_rng(42)
res = {}
xyz, rgb = read_ply(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/data/dense.ply")
xyz, _ = voxel_down(xyz, 0.01)
print("dense cloud (voxel 1 cm units):", len(xyz))

# 1. Umeyama recovers a known similarity exactly
ang = np.radians([10, -25, 40]); cx, cy, cz = np.cos(ang); sx, sy, sz = np.sin(ang)
R0 = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]) @ np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @ np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
s0, t0 = 1.523, np.array([3., -1., 0.5])
sub = xyz[rng.choice(len(xyz), 5000, replace=False)]
s, R, t = umeyama(sub, apply_sim3(sub, s0, R0, t0))
res["umeyama_scale_err"] = abs(s - s0); res["umeyama_rot_err_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(R.T @ R0) - 1) / 2, -1, 1))))
print(res); assert res["umeyama_scale_err"] < 1e-6 and res["umeyama_rot_err_deg"] < 1e-3

# 2. ICP recovers a small perturbation (2 deg, 5 cm) with 3 mm noise
small = xyz[rng.choice(len(xyz), 60000, replace=False)]
a = np.radians(2.0); Rp = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
moved = apply_sim3(small, 1.0, Rp, np.array([0.05, -0.03, 0.02])) + rng.normal(0, 0.003, small.shape)
t_ = time.time(); s, R, t, rmse, inl = align_icp(moved, small, max_dist=0.2, iters=60)
back = apply_sim3(moved, s, R, t)
res["icp_residual_mean_mm"] = float(np.linalg.norm(back - small, axis=1).mean() * 1000); res["icp_time_s"] = time.time() - t_
assert res["icp_residual_mean_mm"] < 8, res

# 3. cloud_metrics: Gaussian noise sigma -> known fraction within tolerance (|N3(0,s)| < t)
sig = 0.01
noisy = small + rng.normal(0, sig, small.shape)
m = cloud_metrics(noisy, small)
res["metrics_noise1cm_acc@2cm"] = m["acc@2cm"]; res["metrics_noise1cm_chamfer_cm"] = m["chamfer_m"] * 100
assert 0.6 < m["acc@2cm"] < 0.95

# 4. tolerance verdict behaves: 2 mm errors adequate for a 16 mm band, 20 mm errors inadequate
res["verdict_2mm"] = tolerance_verdict(rng.normal(0, 0.002, 1000), (-6, 10))["verdict"]
res["verdict_20mm"] = tolerance_verdict(rng.normal(0, 0.020, 1000), (-6, 10))["verdict"]
assert res["verdict_2mm"] in ("ADEQUATE", "MARGINAL") and res["verdict_20mm"] == "INADEQUATE"

# 5. element deviation: a synthetic 'column face' displaced by 25 mm is detected as ~25 mm
gy, gz = np.meshgrid(np.linspace(0, 0.4, 40), np.linspace(0, 3, 200))
face = np.stack([np.zeros(gy.size), gy.ravel(), gz.ravel()], 1); nrm = np.tile([1., 0, 0], (len(face), 1))
scan = face + np.array([0.025, 0, 0]) + rng.normal(0, 0.004, face.shape)
ed = element_deviation(scan, {"C1_face": (face, nrm)})
res["element_offset_detected_mm"] = ed["C1_face"]["offset_median_mm"]
assert abs(ed["C1_face"]["offset_median_mm"] - 25) < 2

# 6. DBSCAN finds one bulge
resid = np.zeros(len(face)); bump = (np.abs(face[:, 2] - 1.5) < 0.3); resid[bump] = 0.03
cl = dbscan_residual_clusters(face, resid, 0.015, eps=0.05, min_pts=10)
res["clusters_found"] = len(cl); assert len(cl) == 1

# 7. ETH3D parser
txt = "Tolerances: 0.01 0.02 0.05\nCompletenesses: 0.31 0.52 0.80\nAccuracies: 0.70 0.84 0.95\nF1-scores: 0.43 0.64 0.87\n"
res["eth3d_parse_F1@2cm"] = parse_eth3d(txt)[0.02]["f1"]; assert res["eth3d_parse_F1@2cm"] == 0.64

print(json.dumps(res, indent=1, default=float))
json.dump(res, open(os.path.join(os.path.dirname(__file__), "selftest_results.json"), "w"), indent=1, default=float)
print("ALL HARNESS SELF-TESTS PASSED")
