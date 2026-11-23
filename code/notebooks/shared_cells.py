"""Cell sources shared between notebooks. Imported by build_notebooks.py."""

GPU_CHECK = r'''# ---- 1. GPU check -------------------------------------------------------------
import subprocess, sys, os, glob, json, time, shutil
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU available:", torch.cuda.is_available())
print(subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv"],
                     capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "No GPU. Settings (right panel) -> Accelerator -> GPU T4 x2"
N_GPUS = torch.cuda.device_count()
print(f"{N_GPUS} GPU(s) visible")
'''

INSTALL_GSPLAT = r'''# ---- install gsplat 1.5.3 (pinned: examples must match the library version) --------
# Takes 5-15 min. If it prints "gsplat OK" at the end, you are good.
import subprocess, sys, os, torch
GSPLAT_VER = "1.5.3"
REPO = "/kaggle/working/gsplat_repo"

def pip(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
    return r.returncode == 0

# Try a prebuilt wheel first (no CUDA compile). Falls back to PyPI + JIT compile.
tv = torch.__version__.split("+")[0].split(".")
cv = torch.version.cuda.split(".")
wheel_index = f"https://docs.gsplat.studio/whl/pt{tv[0]}{tv[1]}cu{cv[0]}{cv[1]}"
pip("ninja", "jaxtyping", "rich")
if pip(f"gsplat=={GSPLAT_VER}", "--index-url", wheel_index, "--no-deps"):
    print("Installed prebuilt wheel from", wheel_index)
else:
    print("No prebuilt wheel for this torch/CUDA combo -> PyPI build (CUDA compiles on first use, ~10 min)")
    assert pip(f"gsplat=={GSPLAT_VER}"), "gsplat install failed - paste the error above to Claude"

if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", f"v{GSPLAT_VER}",
                    "https://github.com/nerfstudio-project/gsplat.git", REPO], check=True)
# gsplat's examples/datasets/ has no __init__.py, so Kaggle's preinstalled HuggingFace "datasets"
# package shadows it ("No module named datasets.colmap"). Making it a real package fixes that.
open(f"{REPO}/examples/datasets/__init__.py", "a").close()
# Example-script dependencies, installed one by one. (requirements.txt pins numpy<2, which has no
# wheel for Kaggle's Python 3.13 and fails with "Cannot import mesonpy".)
pip("tyro>=0.8.8", "viser", "imageio[ffmpeg]", "scikit-learn", "tqdm", "torchmetrics[image]", "opencv-python",
    "Pillow", "tensorboard", "tensorly", "pyyaml", "matplotlib", "splines", "plyfile", "scipy")
pip("git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787")
pip("git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e")
# that pycolmap fork predates numpy 2: np.uint64(-1) raises OverflowError -> patch the one line
import importlib, importlib.util
importlib.invalidate_caches()
_pc = importlib.util.find_spec("pycolmap").submodule_search_locations[0] + "/scene_manager.py"
_src = open(_pc).read().replace("np.uint64(-1)", "np.uint64(np.iinfo(np.uint64).max)")
open(_pc, "w").write(_src)
# fused-ssim compiles a CUDA kernel against the installed torch -> no build isolation
pip("--no-build-isolation", "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5")
chk = subprocess.run([sys.executable, "-c", "import tyro, nerfview, viser, pycolmap, fused_ssim, torchmetrics, sklearn, cv2, yaml, imageio; print('deps OK')"],
                     capture_output=True, text=True)
print(chk.stdout, chk.stderr[-2000:])
assert "deps OK" in chk.stdout, "a dependency failed to install - paste the output above to Claude"

# Warm-up: forces the CUDA kernels to compile NOW rather than mid-training.
warm = r"""
import torch, gsplat
from gsplat import rasterization
N = 256
d = 'cuda'
means = torch.randn(N, 3, device=d); quats = torch.randn(N, 4, device=d)
scales = torch.rand(N, 3, device=d) * 0.1; op = torch.rand(N, device=d); col = torch.rand(N, 3, device=d)
vm = torch.eye(4, device=d)[None].clone(); vm[0, 2, 3] = 5.0
K = torch.tensor([[100., 0, 50], [0, 100., 50], [0, 0, 1]], device=d)[None]
out, alpha, _ = rasterization(means, quats, scales, op, col, vm, K, 100, 100)
print('gsplat OK', gsplat.__version__, tuple(out.shape))
"""
r = subprocess.run([sys.executable, "-c", warm], capture_output=True, text=True)
print(r.stdout, r.stderr[-3000:] if r.returncode else "")
assert r.returncode == 0, "gsplat warm-up failed - paste the error above to Claude"
'''

EXPORT_UTILS = r'''# ---- export utilities --------------------------------------------------------------
# gsplat trains in a NORMALISED world frame. Everything exported here is mapped back to
# the original COLMAP frame with parser.transform, so all teams' outputs line up.
import os, sys, glob, json, re
import numpy as np, torch
REPO = "/kaggle/working/gsplat_repo"
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
print("export utilities loaded")
'''
