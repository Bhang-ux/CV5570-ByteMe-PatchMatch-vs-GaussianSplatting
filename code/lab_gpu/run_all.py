"""CV5570 Byte Me - full experiment run on the lab PC (Windows, RTX 4060 8 GB). Resumable: finished steps are skipped.
Progress: status.txt (one line per event) and logs/*.log. Results for Claude: out/."""
import glob, json, os, shutil, subprocess, sys, time, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
REPO = ROOT / "gsplat_repo"
LOGS, OUT, DATA = ROOT / "logs", ROOT / "out", ROOT / "data"
for d in (LOGS, OUT, DATA):
    d.mkdir(exist_ok=True)
ENVV = dict(os.environ, PYTHONPATH=str(ROOT / "pylib"), PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

def status(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(ROOT / "status.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def done(tag):
    return (ROOT / "markers" / tag).exists()

def mark(tag):
    (ROOT / "markers").mkdir(exist_ok=True)
    (ROOT / "markers" / tag).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))

def run(cmd, log, cwd=None):
    t0 = time.time()
    with open(LOGS / log, "a", encoding="utf-8", errors="replace") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n"); f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=ENVV, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"command failed ({r.returncode}), see logs/{log}")
    return time.time() - t0

def download(url, dst):
    if Path(dst).exists():
        return
    status(f"downloading {url}")
    tmp = str(dst) + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, dst)

def colmap_exe():
    hits = glob.glob(str(ROOT / "colmap" / "**" / "colmap.exe"), recursive=True)
    return hits[0] if hits else None

# ------------------------------------------------------------------ setup
def setup():
    if not done("colmap"):
        z = ROOT / "colmap-x64-windows-cuda.zip"
        download("https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip", z)
        zipfile.ZipFile(z).extractall(ROOT / "colmap")
        mark("colmap")
    if not REPO.exists():
        run(["git", "clone", "-q", "--depth", "1", "--branch", "v1.5.3",
             "https://github.com/nerfstudio-project/gsplat.git", REPO], "setup.log")
    (REPO / "examples" / "datasets" / "__init__.py").touch()
    if not done("site_data"):
        zipfile.ZipFile(ROOT / "kaggle_data.zip").extractall(DATA / "site1_zip")
        src = next((DATA / "site1_zip").rglob("cameras.bin")).parent.parent
        shutil.copytree(src, DATA / "site1", dirs_exist_ok=True)
        mark("site_data")
    status(f"setup ok | colmap: {colmap_exe()}")

# ------------------------------------------------------------------ methods
def patchmatch(ws, tag, max_size):
    out = OUT / f"{tag}_patchmatch_fused.ply"
    if out.exists():
        return
    C = colmap_exe()
    status(f"[{tag}] PatchMatch MVS ...")
    t = run([C, "patch_match_stereo", "--workspace_path", ws, "--workspace_format", "COLMAP",
             "--PatchMatchStereo.geom_consistency", "true", "--PatchMatchStereo.max_image_size", str(max_size)],
            f"{tag}_patchmatch.log")
    t += run([C, "stereo_fusion", "--workspace_path", ws, "--workspace_format", "COLMAP",
              "--input_type", "geometric", "--output_path", out], f"{tag}_patchmatch.log")
    (OUT / f"{tag}_patchmatch_time.json").write_text(json.dumps({"seconds": t}))
    status(f"[{tag}] PatchMatch done in {t/60:.1f} min")

def train(ws, tag, model, extra=()):
    res = ROOT / "results" / f"{tag}_{model}"
    if (res / "ckpts").exists() and list((res / "ckpts").glob("ckpt_*")):
        return res
    status(f"[{tag}] training {model} {' '.join(extra)} ...")
    if model.startswith("3dgs"):
        cmd = [PY, "simple_trainer.py", "default", "--disable_video", "--save_ply"]
    else:
        cmd = [PY, "simple_trainer_2dgs.py"]
        if "noreg" not in model:
            cmd += ["--normal_loss", "--dist_loss"]
    cmd += ["--data_dir", ws, "--data_factor", "1", "--result_dir", res, "--disable_viewer", "--packed", *extra]
    t = run(cmd, f"{tag}_{model}.log", cwd=REPO / "examples")
    (res / "train_time.json").write_text(json.dumps({"seconds": t}))
    status(f"[{tag}] {model} trained in {t/60:.1f} min")
    return res

def export(ws, tag, model, res):
    """Runs in a separate process so the GPU memory is fully released before the next training run."""
    if (OUT / f"{tag}_{model}_surface.ply").exists():
        return
    status(f"[{tag}] exporting {model} ...")
    run([PY, __file__, "--export", ws, tag, model, res], f"{tag}_{model}_export.log")

def export_inproc(ws, tag, model, res):
    ws, res = Path(ws), Path(res)
    sur = OUT / f"{tag}_{model}_surface.ply"
    sys.path.insert(0, str(ROOT))
    import cvlab_export as X
    parser = X.Parser(data_dir=str(ws), factor=1, normalize=True, test_every=8)
    rec = {"stats": X.run_stats(str(res))}
    rec.update(X.export_centers(str(res), parser, str(OUT / f"{tag}_{model}_centers.ply"), max_pts=3_000_000))
    rec.update(X.export_surface(str(res), parser, str(sur), model="2dgs" if model.startswith("2dgs") else "3dgs",
                                max_pts=5_000_000))
    (OUT / f"{tag}_{model}_summary.json").write_text(json.dumps(rec, indent=1, default=float))
    status(f"[{tag}] {model} exported: {rec.get('surface_points')} surface pts")

def gs_all(ws, tag, models=("3dgs", "2dgs"), extra=()):
    for m in models:
        try:
            res = train(ws, tag, m, extra)
            export(ws, tag, m, res)
        except Exception as e:
            status(f"[{tag}] {m} FAILED: {e}")

# ------------------------------------------------------------------ ETH3D
def eth3d_scene(sc, max_size=1200, fraction=1.0):
    import py7zr
    E = DATA / "eth3d"; E.mkdir(exist_ok=True)
    for part in ("dslr_undistorted", "scan_eval"):
        if not done(f"{sc}_{part}"):
            arc = E / f"{sc}_{part}.7z"
            download(f"https://www.eth3d.net/data/{sc}_{part}.7z", arc)
            status(f"extracting {arc.name}")
            with py7zr.SevenZipFile(arc, "r") as z:
                z.extractall(E)
            mark(f"{sc}_{part}")
    tag = sc if fraction == 1 else f"{sc}_v{int(fraction*100)}"
    ws = DATA / f"ws_{tag}"
    if not (ws / "sparse" / "cameras.bin").exists():
        model = next((E / sc).rglob("cameras.txt")).parent
        if fraction < 1:
            names = sorted(l.split()[-1] for l in open(model / "images.txt")
                           if l.strip() and not l.startswith("#") and l.split()[-1].lower().endswith((".jpg", ".png")))
            keep = int(round(1 / fraction))
            sub = DATA / f"model_{tag}"; sub.mkdir(exist_ok=True)
            (sub / "drop.txt").write_text("\n".join(n for i, n in enumerate(names) if i % keep))
            run([colmap_exe(), "image_deleter", "--input_path", model, "--output_path", sub,
                 "--image_names_path", sub / "drop.txt"], f"{tag}_prep.log")
            model = sub
        run([colmap_exe(), "image_undistorter", "--image_path", E / sc / "images", "--input_path", model,
             "--output_path", ws, "--output_type", "COLMAP", "--max_image_size", str(max_size)], f"{tag}_prep.log")
    # ground truth for Claude's evaluation
    gt = OUT / "gt" / sc
    if not gt.exists():
        shutil.copytree(next((E / sc).rglob("scan_alignment.mlp")).parent, gt)
    return tag, ws

# ------------------------------------------------------------------ plan
def plan():
    status("===== run started =====")
    setup()
    site = DATA / "site1"
    # 1. our site: PatchMatch + 3DGS + 2DGS (same poses)
    patchmatch(site, "site1", 1024)
    gs_all(site, "site1")
    # 2. ETH3D laser ground truth: building scenes
    for sc in ("facade", "courtyard", "electro", "terrace"):
        try:
            tag, ws = eth3d_scene(sc)
            patchmatch(ws, tag, 1200)
            gs_all(ws, tag)
        except Exception as e:
            status(f"[{sc}] FAILED: {e}")
        if sc == "facade":
            # 3. ablations on facade: fewer photos, shorter training, no regularisers
            try:
                for fr in (0.5, 0.25):
                    tag, ws = eth3d_scene("facade", fraction=fr)
                    patchmatch(ws, tag, 1200)
                    gs_all(ws, tag)
                _, ws = eth3d_scene("facade")
                gs_all(ws, "facade", models=("2dgs_noreg",))
                gs_all(ws, "facade", models=("3dgs_7k", "2dgs_7k"), extra=("--max_steps", "7000"))
            except Exception as e:
                status(f"[ablations] FAILED: {e}")
    status("===== ALL DONE =====")


def wait_for_claude():
    """Claude controls the run remotely: restart.flag -> re-launch (picking up any fixed code), stop.flag -> exit."""
    status("waiting for Claude (restart.flag / stop.flag)")
    while True:
        if (ROOT / "stop.flag").exists():
            os.remove(ROOT / "stop.flag"); status("stopped by Claude"); return False
        if (ROOT / "restart.flag").exists():
            os.remove(ROOT / "restart.flag"); status("restart requested by Claude"); return True
        time.sleep(30)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        export_inproc(*sys.argv[2:6]); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--plan":
        try:
            plan()
        except Exception as e:
            import traceback
            status(f"RUN CRASHED: {e!r}")
            with open(LOGS / "crash.log", "a", encoding="utf-8") as f:
                traceback.print_exc(file=f)
        sys.exit(0)
    # supervisor: each pass runs the plan in a fresh process, so code fixed by Claude is picked up on restart
    while True:
        subprocess.run([PY, __file__, "--plan"], env=ENVV)
        if not wait_for_claude():
            break
