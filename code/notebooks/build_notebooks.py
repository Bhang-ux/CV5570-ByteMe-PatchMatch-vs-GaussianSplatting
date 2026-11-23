import nbformat as nbf
from shared_cells import GPU_CHECK, INSTALL_GSPLAT, EXPORT_UTILS

def md(s): return nbf.v4.new_markdown_cell(s.strip())
def code(s): return nbf.v4.new_code_cell(s.strip())

def save(cells, path):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}
    nbf.write(nb, path)
    print("wrote", path, len(cells), "cells")

# ---------------------------------------------------------------------------------------
LOCATE_SITE = r'''# ---- 2. locate the team dataset (cv5570-site1) -----------------------------------
import glob, os, shutil, subprocess
hits = glob.glob("/kaggle/input/**/colmap_workspace/sparse/cameras.bin", recursive=True)
assert hits, ("Dataset not attached. Right panel -> Add Input -> search 'cv5570-site1' -> Add. "
              "Then re-run this cell.")
SRC = os.path.dirname(os.path.dirname(hits[0]))           # .../colmap_workspace
DATA = "/kaggle/working/data"
if not os.path.exists(DATA):
    shutil.copytree(SRC, DATA)                           # writable copy
n_img = sum(len(f) for _, _, f in os.walk(f"{DATA}/images"))
print("dataset:", SRC, "->", DATA, "|", n_img, "images")
print(sorted(os.listdir(DATA)))
'''

LAUNCH_GS = r'''# ---- 4. train 3DGS and 2DGS (both GPUs in parallel if you have T4 x2) ---------------
import subprocess, os, torch
REPO = "/kaggle/working/gsplat_repo"; DATA = "/kaggle/working/data"; RES = "/kaggle/working/results"
os.makedirs(RES, exist_ok=True)
JOBS = {
    # standard 3DGS
    "3dgs": f"python simple_trainer.py default --data_dir {DATA} --data_factor 1 "
            f"--result_dir {RES}/3dgs --disable_viewer --save_ply",
    # 2DGS WITH normal + distortion regularisers - these are what make 2DGS geometrically accurate
    # (gsplat leaves them OFF by default)
    "2dgs": f"python simple_trainer_2dgs.py --data_dir {DATA} --data_factor 1 "
            f"--result_dir {RES}/2dgs --disable_viewer --normal_loss --dist_loss",
}
def wrap(name, cmd, gpu):
    return (f"cd {REPO}/examples && CUDA_VISIBLE_DEVICES={gpu} {cmd} "
            f"> {RES}/{name}.log 2>&1; echo EXIT=$? >> {RES}/{name}.log")
if torch.cuda.device_count() >= 2:
    for g, (name, cmd) in enumerate(JOBS.items()):
        subprocess.Popen(wrap(name, cmd, g), shell=True)
    print("launched 3DGS on GPU0 and 2DGS on GPU1 (parallel, ~45-75 min)")
else:
    subprocess.Popen(" ; ".join(wrap(n, c, 0) for n, c in JOBS.items()), shell=True)
    print("single GPU: launched 3DGS then 2DGS sequentially (~1.5-2.5 h)")
'''

WAIT_GS = r'''# ---- 5. wait for training (safe in 'Save & Run All' mode) ---------------------------
import time, re, os
RES = "/kaggle/working/results"
def status(name):
    log = f"{RES}/{name}.log"
    if not os.path.exists(log):
        return "pending", None
    txt = open(log, errors="ignore").read()
    if "EXIT=" in txt:
        code = re.findall(r"EXIT=(\d+)", txt)[-1]
        return ("done" if code == "0" else f"FAILED ({code})"), txt
    steps = re.findall(r"(\d+)/30000", txt)
    return f"running, step {steps[-1] if steps else '?'}/30000", txt
t0 = time.time()
while True:
    st = {n: status(n) for n in ("3dgs", "2dgs")}
    print(time.strftime("%H:%M"), " | ".join(f"{n}: {s[0]}" for n, s in st.items()), flush=True)
    if all(s[0] != "pending" and not s[0].startswith("running") for s in st.values()):
        break
    time.sleep(120)
for n, (s, txt) in st.items():
    if s != "done":
        print(f"\n===== {n} log tail =====\n", txt[-4000:] if txt else "(no log)")
print(f"waited {(time.time()-t0)/60:.0f} min")
'''

EXPORT_SITE = r'''# ---- 7. export geometry in the COLMAP frame ----------------------------------------
import json, os
DATA = "/kaggle/working/data"; RES = "/kaggle/working/results"; OUT = "/kaggle/working/deliverables"
os.makedirs(OUT, exist_ok=True)
parser = Parser(data_dir=DATA, factor=1, normalize=True, test_every=8)
print("normalisation transform (COLMAP -> gsplat frame):\n", parser.transform)
summary = {"scene": "site1", "transform": parser.transform.tolist(),
           "scene_scale_normalised": float(parser.scene_scale)}
for name, model in (("3dgs", "3dgs"), ("2dgs", "2dgs")):
    run = f"{RES}/{name}"
    if not os.path.isdir(f"{run}/ckpts"):
        print(f"skip {name}: no checkpoints"); continue
    print(f"[{name}]")
    rec = {"stats": run_stats(run)}
    rec.update(export_centers(run, parser, f"{OUT}/site1_{name}_centers.ply"))
    try:
        rec.update(export_surface(run, parser, f"{OUT}/site1_{name}_surface.ply", model=model))
    except Exception as e:
        rec["surface_error"] = repr(e); print("  surface export failed:", e)
    summary[name] = rec
json.dump(summary, open(f"{OUT}/summary_site1.json", "w"), indent=1, default=float)
print(json.dumps({k: v.get("stats", {}).get("val") for k, v in summary.items() if isinstance(v, dict)}, indent=1))
'''

PACKAGE = r'''# ---- 8. package deliverables -------------------------------------------------------
import shutil, glob, os, subprocess
RES = "/kaggle/working/results"; OUT = "/kaggle/working/deliverables"
for name in ("3dgs", "2dgs"):
    renders = sorted(glob.glob(f"{RES}/{name}/renders/val_*.png"))[:4]
    for i, p in enumerate(renders):
        shutil.copy(p, f"{OUT}/site1_{name}_render_{i}.png")
    if os.path.exists(f"{RES}/{name}.log"):
        txt = open(f"{RES}/{name}.log", errors="ignore").read()
        open(f"{OUT}/site1_{name}_log_tail.txt", "w").write(txt[-6000:])
shutil.make_archive("/kaggle/working/deliverables_site1", "zip", OUT)
for f in sorted(os.listdir(OUT)):
    print(f"{f:40s} {os.path.getsize(f'{OUT}/{f}')/1e6:7.1f} MB")
print("\nZIP:", f"{os.path.getsize('/kaggle/working/deliverables_site1.zip')/1e6:.1f} MB",
      "-> /kaggle/working/deliverables_site1.zip")
splats = glob.glob(f"{RES}/3dgs/ply/*.ply")
print("Full 3DGS splat for the live demo (open in https://superspl.at/editor):", splats)
'''

nb1 = [
    md(r'''
# CV5570 · Shubhang · Gaussian Splatting on the site data

Trains **3DGS** and **2DGS** on the shared `cv5570-site1` dataset and exports their geometry in the
original COLMAP frame, so Claude can score them against PatchMatch and the BIM.

**Setup (once):**
1. Right panel → **Accelerator: GPU T4 ×2** (P100 also works, just slower) · **Internet: On**
2. Right panel → **Add Input** → your dataset `cv5570-site1` → Add
3. Top-right → **Save Version** → **Save & Run All (Commit)**. This runs headless for ~1.5–2.5 h —
   you can close the laptop. No more lost runs from disconnects.

**When it finishes:** open the version → **Output** tab → download `deliverables_site1.zip`
(~30–60 MB) and attach it in the Claude chat with "site GS results". Also download the big
splat `.ply` under `results/3dgs/ply/` for the presentation demo.

If a cell fails, copy the red error text into Claude.
'''),
    code(GPU_CHECK),
    code(LOCATE_SITE),
    code(INSTALL_GSPLAT),
    code(LAUNCH_GS),
    code(WAIT_GS),
    code(EXPORT_UTILS),
    code(EXPORT_SITE),
    code(PACKAGE),
]
save(nb1, "./CV5570_Shubhang_GaussianSplatting.ipynb")

# ---------------------------------------------------------------------------------------
COLMAP_INSTALL = r'''# ---- install COLMAP with CUDA (the apt build has NO CUDA - that is what blocked us in Sept)
import subprocess, os, shutil
def sh(cmd, check=True, tail=1500):
    print("$", cmd[:200]); r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode and check:
        print(r.stdout[-tail:], r.stderr[-tail:]); raise RuntimeError("command failed")
    return r
def has_cuda(binary):
    r = subprocess.run(f"{binary} -h", shell=True, capture_output=True, text=True)
    return "with CUDA" in (r.stdout + r.stderr)

COLMAP = None
# Option A - conda-forge CUDA build via micromamba (~5 min)
try:
    sh("cd /kaggle/working && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba")
    sh("/kaggle/working/bin/micromamba create -y -q -p /kaggle/working/mm -c conda-forge 'colmap=*=*cuda*'")
    if has_cuda("/kaggle/working/mm/bin/colmap"):
        COLMAP = "/kaggle/working/mm/bin/colmap"
except Exception as e:
    print("Option A failed:", e)

# Option B - build COLMAP 3.9.1 from source with CUDA (~40 min)
if COLMAP is None:
    print("Falling back to source build ...")
    cc = subprocess.run("nvidia-smi --query-gpu=compute_cap --format=csv,noheader", shell=True,
                        capture_output=True, text=True).stdout.split()[0].replace(".", "")
    sh("apt-get update -qq && apt-get install -y -qq git cmake ninja-build build-essential "
       "libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev "
       "libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev "
       "libgmock-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev")
    if not os.path.exists("/kaggle/working/colmap_src"):
        sh("git clone -q --depth 1 --branch 3.9.1 https://github.com/colmap/colmap.git /kaggle/working/colmap_src")
    sh(f"cd /kaggle/working/colmap_src && mkdir -p build && cd build && cmake .. -GNinja "
       f"-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES={cc} -DGUI_ENABLED=OFF > cmake.log 2>&1 "
       f"&& ninja -j4 > ninja.log 2>&1", tail=4000)
    COLMAP = sh("find /kaggle/working/colmap_src/build -name colmap -type f -perm -u+x | head -1").stdout.strip()

assert COLMAP and has_cuda(COLMAP), "No CUDA COLMAP - paste the output above to Claude"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
print("CUDA COLMAP ready:", COLMAP)
print(subprocess.run(f"{COLMAP} -h", shell=True, capture_output=True, text=True).stdout.splitlines()[0])
'''

PATCHMATCH_SITE = r'''# ---- 4. PatchMatch MVS + fusion on the site data -----------------------------------
import time, os, json
WS = "/kaggle/working/data"            # the shared dataset IS a COLMAP dense workspace
OUT = "/kaggle/working/deliverables"; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
sh(f"{COLMAP} patch_match_stereo --workspace_path {WS} --workspace_format COLMAP "
   f"--PatchMatchStereo.geom_consistency true --PatchMatchStereo.max_image_size 1024 "
   f"> {OUT}/patchmatch.log 2>&1", tail=4000)
t_pm = time.time() - t0
sh(f"{COLMAP} stereo_fusion --workspace_path {WS} --workspace_format COLMAP "
   f"--input_type geometric --output_path {WS}/fused.ply > {OUT}/fusion.log 2>&1", tail=4000)
t_all = time.time() - t0
print(f"PatchMatch {t_pm/60:.1f} min, total {t_all/60:.1f} min")
json.dump({"patchmatch_min": t_pm/60, "total_min": t_all/60}, open(f"{OUT}/timing_patchmatch.json", "w"))
'''

SLIM_PM = r'''# ---- 5. slim export (binary PLY xyz+rgb, same format as the GS exports) ------------
import numpy as np, os, json, shutil
WS = "/kaggle/working/data"; OUT = "/kaggle/working/deliverables"
from plyfile import PlyData
v = PlyData.read(f"{WS}/fused.ply")["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
n_total = len(xyz); print(f"fused: {n_total:,} points")
MAX = 1_500_000
if n_total > MAX:
    idx = np.random.default_rng(0).choice(n_total, MAX, replace=False); xyz, rgb = xyz[idx], rgb[idx]
dt = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("red","u1"),("green","u1"),("blue","u1")])
a = np.empty(len(xyz), dt); a["x"], a["y"], a["z"] = xyz.T; a["red"], a["green"], a["blue"] = rgb.T
with open(f"{OUT}/site1_patchmatch_fused.ply", "wb") as f:
    f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(a)}\nproperty float x\nproperty float y\n"
             "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
    f.write(a.tobytes())
json.dump({"fused_points_total": int(n_total)}, open(f"{OUT}/stats_patchmatch.json", "w"))
shutil.make_archive("/kaggle/working/deliverables_patchmatch", "zip", OUT)
for fn in sorted(os.listdir(OUT)):
    print(f"{fn:40s} {os.path.getsize(f'{OUT}/{fn}')/1e6:7.1f} MB")
print("ZIP -> /kaggle/working/deliverables_patchmatch.zip")
'''

nb2 = [
    md(r'''
# CV5570 · Raj · CUDA COLMAP + PatchMatch MVS baseline (site data)

The classical arm of the comparison. Replaces the CPU stereo workaround with COLMAP's real
PatchMatch MVS, on exactly the same images and camera poses the GS runs use.

**Setup:** Accelerator **GPU T4 ×2** · Internet **On** · Add Input → `cv5570-site1` →
**Save Version → Save & Run All**. Runtime ~20 min (conda route) or ~60 min (source build).

**When done:** Output tab → download `deliverables_patchmatch.zip` → attach in the Claude chat with
"patchmatch results".
'''),
    code(GPU_CHECK),
    code(LOCATE_SITE),
    code(COLMAP_INSTALL),
    code(PATCHMATCH_SITE),
    code("import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'plyfile'])"),
    code(SLIM_PM),
]
save(nb2, "./CV5570_Raj_PatchMatch.ipynb")

# ---------------------------------------------------------------------------------------
ETH3D_SETUP = r'''# ---- ETH3D evaluation tool + data (survey-grade laser ground truth) ------------------
import os, subprocess, glob
SCENES = ["facade", "courtyard"]        # building scenes. Drop 'courtyard' if short on time.
E = "/kaggle/working/eth3d"; os.makedirs(E, exist_ok=True)
sh("apt-get update -qq && apt-get install -y -qq p7zip-full cmake libboost-filesystem-dev "
   "libboost-system-dev libeigen3-dev libpcl-dev", tail=3000)
if not os.path.exists(f"{E}/mve/build/ETH3DMultiViewEvaluation"):
    sh(f"git clone -q --depth 1 https://github.com/ETH3D/multi-view-evaluation.git {E}/mve")
    sh(f"sed -i 's/ -std=c++11//' {E}/mve/CMakeLists.txt")   # PCL >= 1.12 needs C++14
    sh(f"cd {E}/mve && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release > cmake.log 2>&1 "
       f"&& make -j4 > make.log 2>&1", tail=4000)
EVAL = f"{E}/mve/build/ETH3DMultiViewEvaluation"
assert os.path.exists(EVAL), "evaluator build failed"
for sc in SCENES:
    for part in ("dslr_undistorted", "scan_eval"):
        f = f"{E}/{sc}_{part}.7z"
        if not glob.glob(f"{E}/{sc}/*{part.split('_')[-1]}*") and not os.path.exists(f + ".done"):
            sh(f"cd {E} && wget -q https://www.eth3d.net/data/{sc}_{part}.7z && 7z x -y {sc}_{part}.7z > /dev/null "
               f"&& rm {sc}_{part}.7z && touch {f}.done")
    print(sc, sorted(os.listdir(f"{E}/{sc}")))
'''

ETH3D_PREP = r'''# ---- prepare each scene: downscale to <=1600 px with COLMAP (keeps poses = GT frame) ----
WS = {}
for sc in SCENES:
    calib = glob.glob(f"{E}/{sc}/**/cameras.txt", recursive=True)[0]
    model_dir = os.path.dirname(calib)
    img_root = f"{E}/{sc}/images"
    ws = f"/kaggle/working/ws_{sc}"
    if not os.path.exists(f"{ws}/sparse/cameras.bin"):
        sh(f"{COLMAP} image_undistorter --image_path {img_root} --input_path {model_dir} "
           f"--output_path {ws} --output_type COLMAP --max_image_size 1600")
    WS[sc] = ws
    n = sum(len(f) for _, _, f in os.walk(f"{ws}/images"))
    print(sc, "->", ws, "|", n, "images")
'''

ETH3D_PM = r'''# ---- PatchMatch on each ETH3D scene ---------------------------------------------------
import time, json
TIMING = {}
for sc, ws in WS.items():
    t0 = time.time()
    if not os.path.exists(f"{ws}/fused.ply"):
        sh(f"{COLMAP} patch_match_stereo --workspace_path {ws} --workspace_format COLMAP "
           f"--PatchMatchStereo.geom_consistency true --PatchMatchStereo.max_image_size 1600 > {ws}/pm.log 2>&1", tail=3000)
        sh(f"{COLMAP} stereo_fusion --workspace_path {ws} --workspace_format COLMAP "
           f"--input_type geometric --output_path {ws}/fused.ply > {ws}/fusion.log 2>&1", tail=3000)
    TIMING[f"{sc}_patchmatch_min"] = (time.time() - t0) / 60
    print(sc, f"PatchMatch done in {TIMING[f'{sc}_patchmatch_min']:.1f} min")
'''

ETH3D_GS = r'''# ---- 3DGS + 2DGS on each ETH3D scene (2 GPUs in parallel) ----------------------------
import subprocess, time, re, torch
REPO = "/kaggle/working/gsplat_repo"; RES = "/kaggle/working/results_eth3d"; os.makedirs(RES, exist_ok=True)
def jobs_for(sc, ws):
    return {f"{sc}_3dgs": f"python simple_trainer.py default --data_dir {ws} --data_factor 1 --result_dir {RES}/{sc}_3dgs --disable_viewer",
            f"{sc}_2dgs": f"python simple_trainer_2dgs.py --data_dir {ws} --data_factor 1 --result_dir {RES}/{sc}_2dgs --disable_viewer --normal_loss --dist_loss"}
ALL = {}
for sc, ws in WS.items():
    ALL.update(jobs_for(sc, ws))
def run(name, cmd, gpu):
    return f"cd {REPO}/examples && CUDA_VISIBLE_DEVICES={gpu} {cmd} > {RES}/{name}.log 2>&1; echo EXIT=$? >> {RES}/{name}.log"
ng = max(1, torch.cuda.device_count())
queues = [[] for _ in range(ng)]
for i, (n, c) in enumerate(ALL.items()):
    queues[i % ng].append(run(n, c, i % ng))
for q in queues:
    subprocess.Popen(" ; ".join(q), shell=True)
print(f"launched {len(ALL)} runs across {ng} GPU(s)")
while True:
    done = []
    for n in ALL:
        log = f"{RES}/{n}.log"
        txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
        done.append("EXIT=" in txt)
    print(time.strftime("%H:%M"), f"{sum(done)}/{len(ALL)} finished", flush=True)
    if all(done): break
    time.sleep(180)
for n in ALL:
    txt = open(f"{RES}/{n}.log", errors="ignore").read()
    if "EXIT=0" not in txt:
        print(f"===== {n} FAILED =====\n", txt[-3000:])
'''

ETH3D_EXPORT_EVAL = r'''# ---- export every reconstruction into the GT frame and score it with the ETH3D tool ----
import json, re, csv
OUT = "/kaggle/working/deliverables"; os.makedirs(OUT, exist_ok=True)
TOL = "0.01,0.02,0.05,0.1,0.2,0.5"
rows = []
def evaluate(sc, method, ply):
    mlp = glob.glob(f"{E}/{sc}/**/scan_alignment.mlp", recursive=True)[0]
    r = subprocess.run(f"{EVAL} --reconstruction_ply_path {ply} --ground_truth_mlp_path {mlp} --tolerances {TOL}",
                       shell=True, capture_output=True, text=True)
    open(f"{OUT}/eval_{sc}_{method}.txt", "w").write(r.stdout + "\n" + r.stderr)
    print(f"--- {sc} / {method} ---\n" + r.stdout[-1200:])
    rows.append({"scene": sc, "method": method, "raw": r.stdout[-2000:]})

for sc, ws in WS.items():
    evaluate(sc, "patchmatch", f"{ws}/fused.ply")
    parser = Parser(data_dir=ws, factor=1, normalize=True, test_every=8)
    for model in ("3dgs", "2dgs"):
        run = f"{RES}/{sc}_{model}"
        if not os.path.isdir(f"{run}/ckpts"):
            continue
        cen = f"{OUT}/{sc}_{model}_centers.ply"; sur = f"{OUT}/{sc}_{model}_surface.ply"
        export_centers(run, parser, cen, max_pts=3_000_000)
        try:
            export_surface(run, parser, sur, model=model, max_pts=5_000_000)
            evaluate(sc, f"{model}_surface", sur)
        except Exception as e:
            print("surface export failed:", e)
        evaluate(sc, f"{model}_centers", cen)
        json.dump(run_stats(run), open(f"{OUT}/stats_{sc}_{model}.json", "w"), indent=1)
json.dump({"timing": TIMING, "results": rows}, open(f"{OUT}/eth3d_results.json", "w"), indent=1)
'''

ETH3D_PACKAGE = r'''# ---- package: small files only (the eval text files carry the numbers) -------------
import shutil
small = "/kaggle/working/deliverables_eth3d_small"; os.makedirs(small, exist_ok=True)
for f in glob.glob(f"{OUT}/*.txt") + glob.glob(f"{OUT}/*.json"):
    shutil.copy(f, small)
shutil.make_archive("/kaggle/working/deliverables_eth3d", "zip", small)
print(os.listdir(small))
print("ZIP -> /kaggle/working/deliverables_eth3d.zip  (the .ply files stay in Output if Claude asks for them)")
'''

nb3 = [
    md(r'''
# CV5570 · Raj · ETH3D benchmark — PatchMatch vs 3DGS vs 2DGS against laser ground truth

This is the most rigorous experiment in the project: building scenes (**facade**, **courtyard**) with
survey-grade laser scans. All three methods use the *provided* ETH3D camera poses, so their output
is already in the ground-truth frame, and the official ETH3D evaluator scores accuracy,
completeness and F1 at 1, 2, 5, 10, 20, 50 cm.

**Setup:** Accelerator **GPU T4 ×2** · Internet **On** · **Save Version → Save & Run All**.
Runtime ~4–6 h for both scenes (set `SCENES = ["facade"]` in the ETH3D cell to halve it).

**When done:** Output → download `deliverables_eth3d.zip` (small) → attach in the Claude chat with
"eth3d results".
'''),
    code(GPU_CHECK),
    code(COLMAP_INSTALL),
    code(INSTALL_GSPLAT),
    code(ETH3D_SETUP),
    code(ETH3D_PREP),
    code(ETH3D_PM),
    code(ETH3D_GS),
    code(EXPORT_UTILS),
    code(ETH3D_EXPORT_EVAL),
    code(ETH3D_PACKAGE),
]
save(nb3, "./CV5570_Raj_ETH3D_Benchmark.ipynb")

# ---------------------------------------------------------------------------------------
# NB4: Shubhang's paid-cloud notebook: more ETH3D scenes + ablations (the "cost vs accuracy" study)
ABL_SETUP = r'''# ---- ablation plan ------------------------------------------------------------------
# Every run is scored against laser ground truth, so each one is a data point in the paper.
SCENES = ["facade", "courtyard", "electro", "terrace", "delivery_area"]   # building/urban scenes
ABL_SCENE = "facade"                                                      # ablations run here
VIEW_FRACTIONS = [0.5, 0.25]          # how many photos does a site engineer really need?
SHORT_STEPS = 7000                    # does a 4x cheaper training run lose accuracy?
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
'''

ABL_PREP = r'''# ---- build full + view-subsampled workspaces (poses always = ETH3D GT frame) --------
import glob, os
WS = {}
def make_ws(sc, frac):
    calib = glob.glob(f"{E}/{sc}/**/cameras.txt", recursive=True)[0]
    model = os.path.dirname(calib)
    tag = f"{sc}" if frac == 1 else f"{sc}_v{int(frac*100)}"
    ws = f"{W}/ws_{tag}"
    if os.path.exists(f"{ws}/sparse/cameras.bin"):
        return tag, ws
    if frac < 1:
        names = sorted(l.split()[-1] for l in open(f"{model}/images.txt")
                       if l.strip() and not l.startswith("#") and l.split()[-1].lower().endswith((".jpg", ".png")))
        keep_every = int(round(1 / frac))
        drop = [n for i, n in enumerate(names) if i % keep_every]
        sub = f"{W}/model_{tag}"; os.makedirs(sub, exist_ok=True)
        open(f"{sub}/drop.txt", "w").write("\n".join(drop))
        sh(f"{COLMAP} image_deleter --input_path {model} --output_path {sub} --image_names_path {sub}/drop.txt")
        model = sub
    sh(f"{COLMAP} image_undistorter --image_path {E}/{sc}/images --input_path {model} "
       f"--output_path {ws} --output_type COLMAP --max_image_size 1600")
    return tag, ws
for sc in SCENES:
    t, w = make_ws(sc, 1); WS[t] = w
for fr in VIEW_FRACTIONS:
    t, w = make_ws(ABL_SCENE, fr); WS[t] = w
for t, w in WS.items():
    print(f"{t:22s}", sum(len(f) for _, _, f in os.walk(f"{w}/images")), "images")
'''

ABL_RUNS = r'''# ---- queue every run across all GPUs ------------------------------------------------
import subprocess, time, torch, json
REPO = f"{W}/gsplat_repo"; RES = f"{W}/results_abl"; os.makedirs(RES, exist_ok=True)
G3 = "python simple_trainer.py default --data_dir {ws} --data_factor 1 --result_dir {out} --disable_viewer"
G2 = "python simple_trainer_2dgs.py --data_dir {ws} --data_factor 1 --result_dir {out} --disable_viewer"
RUNS = {}
for tag, ws in WS.items():
    RUNS[f"{tag}__3dgs"] = G3.format(ws=ws, out=f"{RES}/{tag}__3dgs")
    RUNS[f"{tag}__2dgs"] = G2.format(ws=ws, out=f"{RES}/{tag}__2dgs") + " --normal_loss --dist_loss"
ws = WS[ABL_SCENE]
RUNS[f"{ABL_SCENE}__2dgs_noreg"] = G2.format(ws=ws, out=f"{RES}/{ABL_SCENE}__2dgs_noreg")
RUNS[f"{ABL_SCENE}__3dgs_7k"] = G3.format(ws=ws, out=f"{RES}/{ABL_SCENE}__3dgs_7k") + f" --max_steps {SHORT_STEPS}"
RUNS[f"{ABL_SCENE}__2dgs_7k"] = (G2.format(ws=ws, out=f"{RES}/{ABL_SCENE}__2dgs_7k")
                                 + f" --normal_loss --dist_loss --max_steps {SHORT_STEPS}")
RUNS = {k: v for k, v in RUNS.items() if not os.path.isdir(f"{RES}/{k}/ckpts")}   # resumable
ng = max(1, torch.cuda.device_count())
q = [[] for _ in range(ng)]
for i, (n, c) in enumerate(RUNS.items()):
    q[i % ng].append(f"cd {REPO}/examples && CUDA_VISIBLE_DEVICES={i % ng} /usr/bin/time -f WALL=%e {c} "
                     f"> {RES}/{n}.log 2>&1; echo EXIT=$? >> {RES}/{n}.log")
for qq in q:
    if qq: subprocess.Popen(" ; ".join(qq), shell=True)
print(f"{len(RUNS)} GS runs queued on {ng} GPU(s)")

# PatchMatch for every workspace runs on the CPU side of the same GPUs meanwhile? No - it needs the
# GPU too, so it runs after the GS queue in this same cell. Total wall-time ~ (runs/GPUs) x 40 min.
while True:
    logs = [open(f"{RES}/{n}.log", errors="ignore").read() if os.path.exists(f"{RES}/{n}.log") else "" for n in RUNS]
    done = sum("EXIT=" in t for t in logs)
    print(time.strftime("%H:%M"), f"GS {done}/{len(RUNS)}", flush=True)
    if done == len(RUNS): break
    time.sleep(300)
TIMING = {}
for tag, ws in WS.items():
    if not os.path.exists(f"{ws}/fused.ply"):
        t0 = time.time()
        sh(f"{COLMAP} patch_match_stereo --workspace_path {ws} --workspace_format COLMAP "
           f"--PatchMatchStereo.geom_consistency true --PatchMatchStereo.max_image_size 1600 > {ws}/pm.log 2>&1", tail=3000)
        sh(f"{COLMAP} stereo_fusion --workspace_path {ws} --workspace_format COLMAP --input_type geometric "
           f"--output_path {ws}/fused.ply > {ws}/fusion.log 2>&1", tail=3000)
        TIMING[f"{tag}__patchmatch_s"] = time.time() - t0
        print(tag, "PatchMatch", round(TIMING[f'{tag}__patchmatch_s']/60, 1), "min")
for n in os.listdir(RES):
    if n.endswith(".log"):
        t = open(f"{RES}/{n}", errors="ignore").read()
        m = re.findall(r"WALL=([\d.]+)", t)
        if m: TIMING[n[:-4] + "_s"] = float(m[-1])
        if "EXIT=0" not in t: print("FAILED:", n, t[-1500:])
json.dump(TIMING, open(f"{RES}/timing.json", "w"), indent=1)
'''

ABL_EVAL = r'''# ---- export + ETH3D evaluation of every run ------------------------------------------
import csv, re, json
OUT = f"{W}/deliverables_abl"; os.makedirs(OUT, exist_ok=True)
TOL = "0.01,0.02,0.05,0.1,0.2,0.5"
def scene_of(tag): return tag.split("_v")[0] if "_v" in tag else tag
def evaluate(tag, method, ply):
    sc = scene_of(tag)
    mlp = glob.glob(f"{E}/{sc}/**/scan_alignment.mlp", recursive=True)[0]
    r = subprocess.run(f"{EVAL} --reconstruction_ply_path {ply} --ground_truth_mlp_path {mlp} --tolerances {TOL}",
                       shell=True, capture_output=True, text=True)
    open(f"{OUT}/eval__{tag}__{method}.txt", "w").write(r.stdout + "\n" + r.stderr)
    print(f"{tag:20s} {method:16s}", " | ".join(l for l in r.stdout.splitlines() if l.startswith(("Accuracies", "Completenesses", "F1")))[:300])
parsers = {}
for tag, ws in WS.items():
    if os.path.exists(f"{ws}/fused.ply"):
        evaluate(tag, "patchmatch", f"{ws}/fused.ply")
for run in sorted(os.listdir(RES)):
    if not os.path.isdir(f"{RES}/{run}/ckpts"): continue
    tag, method = run.split("__")
    if tag not in parsers:
        parsers[tag] = Parser(data_dir=WS[tag], factor=1, normalize=True, test_every=8)
    model = "2dgs" if method.startswith("2dgs") else "3dgs"
    cen, sur = f"{OUT}/{tag}__{method}__centers.ply", f"{OUT}/{tag}__{method}__surface.ply"
    export_centers(f"{RES}/{run}", parsers[tag], cen, max_pts=3_000_000)
    evaluate(tag, method + "_centers", cen)
    try:
        export_surface(f"{RES}/{run}", parsers[tag], sur, model=model, max_pts=5_000_000)
        evaluate(tag, method + "_surface", sur)
    except Exception as e:
        print("surface failed", run, e)
    json.dump(run_stats(f"{RES}/{run}"), open(f"{OUT}/stats__{run}.json", "w"), indent=1)
    for p in (cen, sur):            # keep disk small: ply files are not needed after scoring
        if os.path.exists(p) and "facade__" not in p: os.remove(p)
shutil.copy(f"{RES}/timing.json", OUT)
small = f"{W}/deliverables_abl_small"; os.makedirs(small, exist_ok=True)
for f in glob.glob(f"{OUT}/*.txt") + glob.glob(f"{OUT}/*.json"): shutil.copy(f, small)
shutil.make_archive(f"{W}/deliverables_ablations", "zip", small)
print("ZIP ->", f"{W}/deliverables_ablations.zip")
'''

nb4 = [
    md(r'''
# CV5570 · Shubhang · Paid-GPU run: 5 ETH3D scenes + ablations

This is where the cloud credits go. Every run is scored against survey-grade laser scans, giving the
paper its main results table and three "how cheap can you go" curves a site engineer cares about:
- **photos**: 100% vs 50% vs 25% of the views
- **training budget**: 7k vs 30k steps
- **regularisation**: 2DGS with vs without normal/distortion losses

**Machine:** 1–2× NVIDIA L4 / A10G / RTX 4090 / A100 (≥ 24 GB), Ubuntu 22.04, PyTorch 2.x + CUDA 12
image, root/sudo, ≥ 150 GB disk. ~ 30 GS runs → **~10–14 GPU-hours** on one L4 (half with two GPUs).
At typical L4/A10G prices that is about $10–25, so the $100 covers it with room for reruns.

**Run it headless** so a dropped connection doesn't kill it:
```
pip install jupyter nbconvert
nohup jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  CV5570_Shubhang_Cloud_ETH3D_Ablations.ipynb --output done.ipynb > run.log 2>&1 &
tail -f run.log
```
It is resumable: rerunning skips finished runs.

**When done:** download `deliverables_ablations.zip` (small) → attach in the Claude chat with "ablations".
**Stop the VM** afterwards — idle GPUs burn credits.
'''),
    code('W = "/workspace/cv5570"\nimport os; os.makedirs(W, exist_ok=True); os.chdir(W)\nprint("work dir", W)'),
    code(GPU_CHECK),
    code(COLMAP_INSTALL),
    code(INSTALL_GSPLAT),
    code(ABL_SETUP),
    code(ETH3D_SETUP.replace('SCENES = ["facade", "courtyard"]        # building scenes. Drop \'courtyard\' if short on time.\n', '')),
    code(ABL_PREP),
    code(EXPORT_UTILS),
    code(ABL_RUNS),
    code(ABL_EVAL),
]
def cloudify(cells):
    for c in cells:
        if c.cell_type == "code":
            c.source = c.source.replace("/kaggle/working", "/workspace/cv5570").replace('{W}/', '{W}/')
    return cells
save(cloudify(nb4), "./CV5570_Shubhang_Cloud_ETH3D_Ablations.ipynb")
