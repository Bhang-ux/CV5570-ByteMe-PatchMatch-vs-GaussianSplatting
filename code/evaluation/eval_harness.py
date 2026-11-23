"""CV5570 Byte Me: evaluation harness (numpy + scipy only).

Turns any reconstruction (.ply from PatchMatch / 3DGS / 2DGS) into the paper's numbers:
  1. read_ply / write_ply            : binary or ascii PLY, xyz + rgb
  2. umeyama / align_icp             : similarity alignment (scale+R+t) and point-to-plane ICP
  3. scale_from_planes               : metric scale from two parallel planes of known spacing (slab-soffit)
  4. cloud_metrics                   : accuracy / completeness / F1 at tolerances + Chamfer (ETH3D-style)
  5. parse_eth3d                     : reads ETH3DMultiViewEvaluation stdout into numbers
  6. tolerance_verdict               : can method X adjudicate a check with tolerance T? (the paper's punchline)
  7. element_deviation               : per-element signed deviation vs a reference surface (BIM) + clustering
CLI:  python eval_harness.py compare --ref ref.ply --rec a.ply b.ply --names A B --out results/
"""
import argparse, json, os, re, sys
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- IO
_PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
              "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}

def read_ply(path):
    """Returns (xyz Nx3 float64, rgb Nx3 float in [0,1] or None). Vertex element only."""
    with open(path, "rb") as f:
        header, props, n, fmt = [], [], 0, "ascii"
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            header.append(line)
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                cur = line.split()[1]
                if cur == "vertex":
                    n = int(line.split()[2])
            elif line.startswith("property") and "list" not in line and cur == "vertex":
                _, t, name = line.split()
                props.append((name, _PLY_TYPES[t]))
            elif line == "end_header":
                break
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n)
            arr = {name: data[:, i] for i, (name, _) in enumerate(props)}
        else:
            end = "<" if fmt == "binary_little_endian" else ">"
            dt = np.dtype([(nm, end + t) for nm, t in props])
            raw = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
            arr = {nm: raw[nm] for nm, _ in props}
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
    rgb = None
    if all(k in arr for k in ("red", "green", "blue")):
        rgb = np.stack([arr["red"], arr["green"], arr["blue"]], 1).astype(np.float64)
        if rgb.max() > 1.0:
            rgb /= 255.0
    return xyz, rgb

def write_ply(path, xyz, rgb=None):
    xyz = np.asarray(xyz, np.float32); n = len(xyz)
    if rgb is None:
        rgb = np.full((n, 3), 0.7)
    rgb8 = np.clip(np.asarray(rgb) * 255, 0, 255).astype(np.uint8)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    a = np.empty(n, dt); a["x"], a["y"], a["z"] = xyz.T; a["red"], a["green"], a["blue"] = rgb8.T
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(a.tobytes())

def voxel_down(xyz, voxel, rgb=None):
    """One point per voxel (the first one). Keeps metric comparisons density-fair."""
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return (xyz[idx], rgb[idx]) if rgb is not None else (xyz[idx], None)

# ---------------------------------------------------------------- alignment
def umeyama(src, dst, with_scale=True):
    """Least-squares similarity dst ~ s R src + t from corresponding points (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t

def apply_sim3(xyz, s, R, t):
    return s * xyz @ R.T + t

def estimate_normals(xyz, k=16):
    tree = cKDTree(xyz)
    _, nn = tree.query(xyz, k=k)
    nb = xyz[nn] - xyz[nn].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    w, v = np.linalg.eigh(cov)
    return v[:, :, 0]                       # eigenvector of the smallest eigenvalue

def align_icp(src, dst, max_dist, iters=50, with_scale=False, dst_normals=None, tol=1e-7):
    """Point-to-plane ICP (falls back to point-to-point Umeyama when with_scale=True).
    Returns (s, R, t, rmse, inlier_fraction). Needs a reasonable initial pose (do coarse alignment first)."""
    tree = cKDTree(dst)
    if dst_normals is None and not with_scale:
        dst_normals = estimate_normals(dst)
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    prev = np.inf
    for _ in range(iters):
        cur = apply_sim3(src, s, R, t)
        d, j = tree.query(cur, distance_upper_bound=max_dist)
        m = np.isfinite(d)
        if m.sum() < 10:
            break
        p, q = cur[m], dst[j[m]]
        if with_scale:
            ds, dR, dt = umeyama(p, q, True)
        else:
            nrm = dst_normals[j[m]]
            A = np.hstack([np.cross(p, nrm), nrm]); b = ((q - p) * nrm).sum(1)
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            a, bb, c = x[:3]
            dR = np.array([[1, -c, bb], [c, 1, -a], [-bb, a, 1]])
            U, _, Vt = np.linalg.svd(dR); dR = U @ Vt
            ds, dt = 1.0, x[3:]
        s, R, t = ds * s, dR @ R, ds * dR @ t + dt
        rmse = np.sqrt((d[m] ** 2).mean())
        if abs(prev - rmse) < tol:
            break
        prev = rmse
    cur = apply_sim3(src, s, R, t)
    d, _ = tree.query(cur, distance_upper_bound=max_dist)
    m = np.isfinite(d)
    return s, R, t, float(np.sqrt((d[m] ** 2).mean())) if m.any() else np.inf, float(m.mean())

# ---------------------------------------------------------------- metric scale
def fit_plane_ransac(xyz, thresh, iters=2000, seed=0, normal_hint=None, max_angle_deg=15):
    rng = np.random.default_rng(seed); best = (None, -1)
    for _ in range(iters):
        p = xyz[rng.choice(len(xyz), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0]); nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if normal_hint is not None and abs(n @ normal_hint) < np.cos(np.radians(max_angle_deg)):
            continue
        inl = np.abs((xyz - p[0]) @ n) < thresh
        if inl.sum() > best[1]:
            best = ((n, p[0]), inl.sum())
    n, p0 = best[0]
    inl = np.abs((xyz - p0) @ n) < thresh
    c = xyz[inl].mean(0); _, _, Vt = np.linalg.svd(xyz[inl] - c, full_matrices=False); n = Vt[2]
    return n, c, inl

def scale_from_planes(xyz, known_spacing_m, thresh, up_hint=None):
    """Two largest roughly parallel planes (e.g. floor slab and soffit) -> metres per model unit."""
    n1, c1, inl1 = fit_plane_ransac(xyz, thresh, normal_hint=up_hint)
    rest = xyz[~inl1]
    n2, c2, _ = fit_plane_ransac(rest, thresh, normal_hint=n1, max_angle_deg=10)
    gap = abs((c2 - c1) @ n1)
    return known_spacing_m / gap, {"gap_model_units": float(gap), "normal": n1.tolist(),
                                   "parallel_deg": float(np.degrees(np.arccos(min(1, abs(n1 @ n2)))))}

# ---------------------------------------------------------------- metrics
TOLS_M = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)

def cloud_metrics(rec, ref, tolerances=TOLS_M, voxel=None, max_eval_dist=None):
    """ETH3D-style: accuracy = % of rec points within tol of ref; completeness = % of ref points within tol of rec.
    Also mean/median/95th-percentile distances and Chamfer (mean of both directions)."""
    if voxel:
        rec, _ = voxel_down(rec, voxel); ref, _ = voxel_down(ref, voxel)
    d_acc, _ = cKDTree(ref).query(rec)
    d_com, _ = cKDTree(rec).query(ref)
    if max_eval_dist:                      # ignore reconstruction outside the GT's extent (ETH3D does this with occupancy)
        d_acc = d_acc[d_acc < max_eval_dist]
    out = {"n_rec": int(len(rec)), "n_ref": int(len(ref)),
           "acc_mean_m": float(d_acc.mean()), "acc_median_m": float(np.median(d_acc)), "acc_p95_m": float(np.percentile(d_acc, 95)),
           "com_mean_m": float(d_com.mean()), "com_median_m": float(np.median(d_com)), "com_p95_m": float(np.percentile(d_com, 95)),
           "chamfer_m": float(0.5 * (d_acc.mean() + d_com.mean()))}
    for t in tolerances:
        a, c = (d_acc < t).mean(), (d_com < t).mean()
        f = 2 * a * c / (a + c) if a + c > 0 else 0.0
        out[f"acc@{t*100:g}cm"], out[f"com@{t*100:g}cm"], out[f"F1@{t*100:g}cm"] = float(a), float(c), float(f)
    return out

def parse_eth3d(text):
    """Parses ETH3DMultiViewEvaluation stdout. Returns {tolerance_m: {accuracy, completeness, f1}}."""
    grab = lambda key: [float(x) for x in re.search(rf"{key}:\s*([\d.\s]+)", text).group(1).split()] \
        if re.search(rf"{key}:", text) else []
    tol, acc, com, f1 = grab("Tolerances"), grab("Accuracies"), grab("Completenesses"), grab("F1-scores")
    return {t: {"accuracy": a, "completeness": c, "f1": f} for t, a, c, f in zip(tol, acc, com, f1)}

# ---------------------------------------------------------------- tolerance verdict
# Permissible deviations for cast-in-place RC. VERIFY against the code books before the paper is final.
TOLERANCES_MM = {
    "ACI117_cross_section_<=300mm": (-6, 10),    # ACI 117-10, cross-sectional dimension <= 12 in: +3/8 in, -1/4 in
    "ACI117_cross_section_300-900mm": (-10, 13), # > 12 in to 3 ft: +1/2 in, -3/8 in
    "ACI117_plumb_column_<=25m": (-25, 25),      # vertical alignment, total, height <= 83 ft: 1 in
    "IS456_cross_section_columns_beams": (-6, 12),   # IS 456:2000 cl. 17 tolerances (+12 / -6 mm) - verify clause
    "IS456_slab_thickness": (-6, 6),
}

def tolerance_verdict(errors_m, band_mm, confidence=0.95, ratio_rule=0.25):
    """Can a method with these signed measurement errors adjudicate a check with tolerance band (lo, hi) mm?
    Metrology rule of thumb: measurement uncertainty (U95) <= 1/4 of tolerance width (the 'test uncertainty ratio' of 4:1).
    Returns dict with U95, ratio and verdict: ADEQUATE / MARGINAL (<=1/2) / INADEQUATE."""
    e = np.abs(np.asarray(errors_m, float)) * 1000
    u95 = float(np.percentile(e, confidence * 100))
    width = band_mm[1] - band_mm[0]
    r = u95 / width
    verdict = "ADEQUATE" if r <= ratio_rule else ("MARGINAL" if r <= 0.5 else "INADEQUATE")
    return {"U95_mm": u95, "tolerance_width_mm": width, "ratio": r, "verdict": verdict}

def verdict_table(method_errors, tolerances=TOLERANCES_MM):
    """method_errors: {method: array of signed errors in metres} -> rows for the paper's verdict table."""
    rows = []
    for m, err in method_errors.items():
        for name, band in tolerances.items():
            rows.append({"method": m, "check": name, **tolerance_verdict(err, band)})
    return rows

# ---------------------------------------------------------------- element deviation (scan vs BIM)
def signed_distance_to_surface(pts, ref_pts, ref_normals):
    tree = cKDTree(ref_pts)
    d, j = tree.query(pts)
    return ((pts - ref_pts[j]) * ref_normals[j]).sum(1), d

def element_deviation(rec, elements, max_dist=0.15):
    """elements: {element_id: (sample_points Nx3, normals Nx3)} sampled from the BIM surfaces (same metric frame).
    For each element: recon points within max_dist -> robust signed offset (median), spread, coverage."""
    out = {}
    tree = cKDTree(rec)
    for eid, (P, N) in elements.items():
        lo, hi = P.min(0) - max_dist, P.max(0) + max_dist
        cand = rec[np.all((rec > lo) & (rec < hi), 1)]
        if len(cand) < 20:
            out[eid] = {"status": "not observed", "n": int(len(cand))}; continue
        sd, ud = signed_distance_to_surface(cand, P, N)
        near = ud < max_dist
        if near.sum() < 20:
            out[eid] = {"status": "not observed", "n": int(near.sum())}; continue
        d, _ = tree.query(P)
        out[eid] = {"status": "observed", "n": int(near.sum()),
                    "offset_median_mm": float(np.median(sd[near]) * 1000),
                    "offset_mad_mm": float(np.median(np.abs(sd[near] - np.median(sd[near]))) * 1000),
                    "coverage@5cm": float((d < 0.05).mean())}
    return out

def dbscan_residual_clusters(xyz, residual_m, thresh_m, eps, min_pts=30):
    """Clusters points whose |residual| > thresh into deviation blobs (simple grid-free DBSCAN)."""
    idx = np.where(np.abs(residual_m) > thresh_m)[0]
    if len(idx) == 0:
        return []
    P = xyz[idx]; tree = cKDTree(P); lab = -np.ones(len(P), int); c = 0
    for i in range(len(P)):
        if lab[i] != -1:
            continue
        nb = tree.query_ball_point(P[i], eps)
        if len(nb) < min_pts:
            continue
        lab[i] = c; stack = list(nb)
        while stack:
            k = stack.pop()
            if lab[k] == -1:
                lab[k] = c
                nk = tree.query_ball_point(P[k], eps)
                if len(nk) >= min_pts:
                    stack.extend(nk)
        c += 1
    out = []
    for k in range(c):
        m = lab == k
        out.append({"cluster": k, "n": int(m.sum()), "centroid": P[m].mean(0).tolist(),
                    "mean_residual_mm": float(residual_m[idx][m].mean() * 1000),
                    "extent_m": (P[m].max(0) - P[m].min(0)).tolist()})
    return sorted(out, key=lambda r: -r["n"])

# ---------------------------------------------------------------- CLI
def _compare(a):
    ref, _ = read_ply(a.ref)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for path, name in zip(a.rec, a.names or [os.path.basename(p) for p in a.rec]):
        rec, _ = read_ply(path)
        if a.icp:
            s, R, t, rmse, inl = align_icp(rec, ref, max_dist=a.icp)
            rec = apply_sim3(rec, s, R, t)
        m = cloud_metrics(rec, ref, voxel=a.voxel, max_eval_dist=a.max_eval_dist)
        m["method"] = name; rows.append(m)
        print(f"{name:20s} chamfer {m['chamfer_m']*100:6.2f} cm | F1@2cm {m['F1@2cm']:.3f} | F1@5cm {m['F1@5cm']:.3f}")
    json.dump(rows, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    import csv
    with open(os.path.join(a.out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method"] + [k for k in rows[0] if k != "method"]); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compare"); c.add_argument("--ref", required=True); c.add_argument("--rec", nargs="+", required=True)
    c.add_argument("--names", nargs="*"); c.add_argument("--out", default="eval_out"); c.add_argument("--voxel", type=float, default=0.01)
    c.add_argument("--icp", type=float, default=0.0, help="max correspondence dist for ICP refinement (0 = off)")
    c.add_argument("--max_eval_dist", type=float, default=None)
    a = ap.parse_args()
    {"compare": _compare}[a.cmd](a)
