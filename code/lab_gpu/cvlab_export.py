"""Export trained splats to point clouds in the original COLMAP frame (open3d-free)."""
# ---- export utilities --------------------------------------------------------------
# gsplat trains in a NORMALISED world frame. Everything exported here is mapped back to
# the original COLMAP frame with parser.transform, so all teams' outputs line up.
import os, sys, glob, json, re
from pathlib import Path
import numpy as np, torch
REPO = str(Path(__file__).resolve().parent / "gsplat_repo")
if f"{REPO}/examples" not in sys.path:
    sys.path.insert(0, f"{REPO}/examples")
open(f"{REPO}/examples/datasets/__init__.py", "a").close()
for _m in [m for m in sys.modules if m == "datasets" or m.startswith("datasets.")]:
    del sys.modules[_m]                      # drop HuggingFace "datasets" if anything imported it
from datasets.colmap import Parser

C0 = 0.28209479177387814  # SH band-0 constant

def latest_ckpt(run_dir):
    cks = glob.glob(f"{run_dir}/ckpts/*.pt")
    assert cks, f"no checkpoints in {run_dir}/ckpts - did training finish?"
    step = lambda p: int(re.search(r"ckpt_(\d+)", os.path.basename(p)).group(1))
    return max(cks, key=step)

def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, np.float32)
    rgb = np.clip(np.asarray(rgb, np.float32) * 255.0, 0, 255).astype(np.uint8)
    n = len(xyz)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(n, dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode()); f.write(arr.tobytes())
    print(f"  wrote {os.path.basename(path)}: {n:,} pts, {os.path.getsize(path)/1e6:.1f} MB")

def to_colmap_frame(pts, T):
    Ti = np.linalg.inv(T)
    return pts @ Ti[:3, :3].T + Ti[:3, 3]

def subsample(xyz, rgb, max_pts, seed=0):
    if len(xyz) <= max_pts:
        return xyz, rgb
    idx = np.random.default_rng(seed).choice(len(xyz), max_pts, replace=False)
    return xyz[idx], rgb[idx]

def export_centers(run_dir, parser, out_path, min_opacity=0.5, max_pts=800_000):
    """Gaussian centres with opacity > min_opacity. Cheap, but centres of volumetric
    3DGS blobs do NOT lie on the surface - that is one of the things we measure."""
    s = torch.load(latest_ckpt(run_dir), map_location="cpu", weights_only=False)["splats"]
    means = s["means"].float().numpy()
    op = torch.sigmoid(s["opacities"]).float().numpy()
    rgb = (0.5 + C0 * s["sh0"][:, 0, :]).clamp(0, 1).float().numpy()
    keep = op > min_opacity
    xyz, col = subsample(means[keep], rgb[keep], max_pts)
    write_ply(out_path, to_colmap_frame(xyz, parser.transform), col)
    return {"gaussians_total": int(len(op)), "gaussians_opaque": int(keep.sum())}


def fuse_depths(depths, alphas, rgbs, Ks, c2ws, alpha_min=0.5, rel_tol=0.01, min_consistent=2,
                n_neighbors=8, stride=2, voxel=None, max_pts=2_000_000, seed=0):
    """Multi-view consistent depth fusion (same idea as COLMAP stereo_fusion), numpy only.
    depths/alphas: list of HxW float arrays (camera z-depth); rgbs: list of HxWx3 in [0,1];
    Ks: list of 3x3; c2ws: list of 4x4 OpenCV camera-to-world.
    A pixel becomes a 3D point if its depth agrees (|dz|/z < rel_tol) with the rendered depth of at least
    `min_consistent` of its `n_neighbors` nearest cameras. Returns (xyz Nx3, rgb Nx3)."""
    n = len(depths)
    centers = np.array([c[:3, 3] for c in c2ws])
    out_xyz, out_rgb = [], []
    for i in range(n):
        D, A = depths[i], alphas[i]
        H, W = D.shape
        v, u = np.mgrid[0:H:stride, 0:W:stride]
        d = D[v, u]
        ok = (A[v, u] > alpha_min) & (d > 0) & np.isfinite(d)
        u, v, d = u[ok].astype(np.float64), v[ok].astype(np.float64), d[ok].astype(np.float64)
        if len(d) == 0:
            continue
        K = Ks[i]
        xc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], 1)
        R, t = c2ws[i][:3, :3], c2ws[i][:3, 3]
        Xw = xc @ R.T + t
        dist = np.linalg.norm(centers - centers[i], axis=1); dist[i] = np.inf
        nbrs = np.argsort(dist)[:n_neighbors]
        count = np.zeros(len(d), np.int32)
        for j in nbrs:
            Rj, tj, Kj = c2ws[j][:3, :3], c2ws[j][:3, 3], Ks[j]
            Xc = (Xw - tj) @ Rj
            z = Xc[:, 2]
            front = z > 1e-6
            uj = np.full(len(z), -1.0); vj = np.full(len(z), -1.0)
            uj[front] = Kj[0, 0] * Xc[front, 0] / z[front] + Kj[0, 2]
            vj[front] = Kj[1, 1] * Xc[front, 1] / z[front] + Kj[1, 2]
            ui, vi = np.round(uj).astype(np.int64), np.round(vj).astype(np.int64)
            Hj, Wj = depths[j].shape
            inside = front & (ui >= 0) & (ui < Wj) & (vi >= 0) & (vi < Hj)
            dj = np.zeros(len(z)); aj = np.zeros(len(z))
            dj[inside] = depths[j][vi[inside], ui[inside]]
            aj[inside] = alphas[j][vi[inside], ui[inside]]
            good = inside & (aj > alpha_min) & (dj > 0) & (np.abs(dj - z) < rel_tol * z)
            count += good
        keep = count >= min_consistent
        out_xyz.append(Xw[keep])
        out_rgb.append(rgbs[i][v.astype(int)[keep], u.astype(int)[keep]])
    if not out_xyz:
        return np.zeros((0, 3)), np.zeros((0, 3))
    xyz, rgb = np.concatenate(out_xyz), np.concatenate(out_rgb)
    if voxel:
        keys = np.floor(xyz / voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        xyz, rgb = xyz[idx], rgb[idx]
    if len(xyz) > max_pts:
        idx = np.random.default_rng(seed).choice(len(xyz), max_pts, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb

@torch.no_grad()
def export_surface(run_dir, parser, out_path, model="3dgs", alpha_min=0.5, rel_tol=0.01, min_consistent=2,
                   n_neighbors=8, stride=2, voxel_frac=1/2048, max_pts=1_500_000):
    """Render depth from every training view and keep only depths that agree across neighbouring views
    (multi-view consistent fusion, the same principle as COLMAP's stereo_fusion used for PatchMatch).
    3DGS uses expected depth; 2DGS uses median depth as in the 2DGS paper. Identical for both models."""
    from gsplat.rendering import rasterization, rasterization_2dgs
    dev = "cuda"
    s = torch.load(latest_ckpt(run_dir), map_location=dev, weights_only=False)["splats"]
    means, quats = s["means"], s["quats"]
    scales, op = torch.exp(s["scales"]), torch.sigmoid(s["opacities"])
    colors = torch.cat([s["sh0"], s["shN"]], 1)
    sh_deg = int(round(np.sqrt(colors.shape[1]) - 1))
    depths, alphas, rgbs, Ks, c2ws = [], [], [], [], []
    n_views = len(parser.camtoworlds)
    for i in range(n_views):
        cid = parser.camera_ids[i]
        W, H = parser.imsize_dict[cid]
        Kn = parser.Ks_dict[cid]
        c2w = torch.from_numpy(parser.camtoworlds[i]).float().to(dev)
        vm = torch.linalg.inv(c2w)[None]
        K = torch.from_numpy(Kn).float().to(dev)[None]
        if model == "3dgs":
            rc, ra, _ = rasterization(means, quats, scales, op, colors, vm, K, W, H,
                                      sh_degree=sh_deg, render_mode="RGB+ED")
            depth = rc[0, ..., 3]
        else:
            out = rasterization_2dgs(means, quats, scales, op, colors, vm, K, W, H,
                                     sh_degree=sh_deg, render_mode="RGB+ED")
            rc, ra = out[0], out[1]
            depth = out[5][0, ..., 0]
        depths.append(depth.float().cpu().numpy()); alphas.append(ra[0, ..., 0].float().cpu().numpy())
        rgbs.append(rc[0, ..., :3].clamp(0, 1).float().cpu().numpy())
        Ks.append(np.asarray(Kn, np.float64)); c2ws.append(np.asarray(parser.camtoworlds[i], np.float64))
    print(f"    rendered {n_views} views, fusing ...")
    xyz, col = fuse_depths(depths, alphas, rgbs, Ks, c2ws, alpha_min=alpha_min, rel_tol=rel_tol,
                           min_consistent=min_consistent, n_neighbors=n_neighbors, stride=stride,
                           voxel=float(parser.scene_scale) * voxel_frac, max_pts=max_pts)
    write_ply(out_path, to_colmap_frame(xyz, parser.transform), col)
    return {"surface_points": int(len(xyz))}

def run_stats(run_dir):
    out = {}
    for kind in ("val", "train"):
        fs = sorted(glob.glob(f"{run_dir}/stats/{kind}_step*.json"))
        if fs:
            out[kind] = json.load(open(fs[-1]))
    return out

