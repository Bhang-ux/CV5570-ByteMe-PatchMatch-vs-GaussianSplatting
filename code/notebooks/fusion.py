import numpy as np

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
