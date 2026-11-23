"""Builds the paper/deck results from results/eth3d/eval_<scene>_<method>[_<export>].txt (ETH3D evaluator output)
and dist_<scene>.json (eth3d_dist.py). Writes eth3d_summary.csv, results_tables.md, f1_vs_tolerance.png."""
import glob, json, os, re, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(__file__), "..", "..", "results", "eth3d")
TOL = [1, 2, 5, 10, 20, 50]
COL = {"PatchMatch": "#2a78d6", "2DGS": "#eb6834", "3DGS": "#1baf7a"}
SCENES = ["facade", "courtyard", "terrace"]

def parse(f):
    t = open(f).read()
    g = lambda k: [float(x) for x in re.search(k + r":\s*([\d.e\-\s]+)", t).group(1).split()][:6]
    return {"acc": g("Accuracies"), "comp": g("Completenesses"), "f1": g("F1-scores")}

runs = {}
for f in sorted(glob.glob(f"{D}/eval_*.txt")):
    key = os.path.basename(f)[5:-4]           # e.g. facade_3dgs_surface, facade_patchmatch, facade_v50_2dgs_centers
    try:
        runs[key] = parse(f)
    except Exception:
        pass

with open(f"{D}/eth3d_summary.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run"] + [f"{m}_{t}cm" for m in ("acc", "comp", "F1") for t in TOL])
    for k, r in runs.items():
        w.writerow([k] + [f"{x:.3f}" for x in r["acc"] + r["comp"] + r["f1"]])

def best(scene, model):
    """Best geometry export of a splat model on a scene (by F1 at 5 cm), with its export name."""
    c = {k: r for k, r in runs.items() if re.fullmatch(f"{scene}_{model}_(surface|surfstrict|centers)", k)}
    if not c:
        return None, None
    k = max(c, key=lambda k: c[k]["f1"][2])
    return k.rsplit("_", 1)[1], c[k]

md = ["## Main ETH3D table (F1 / accuracy / completeness at 2, 5, 10 cm)\n",
      "| Scene | Method (export) | F1 2cm | F1 5cm | F1 10cm | Acc 5cm | Comp 5cm |", "|---|---|---|---|---|---|---|"]
scenes = [s for s in SCENES if f"{s}_patchmatch" in runs]
fig, axes = plt.subplots(1, len(scenes), figsize=(4.2 * len(scenes), 3.6), sharey=True, squeeze=False)
for ax, s in zip(axes[0], scenes):
    rows = [("PatchMatch", "fused", runs[f"{s}_patchmatch"])]
    for m in ("2dgs", "3dgs"):
        e, r = best(s, m)
        if r:
            rows.append((m.upper(), e, r))
    for name, e, r in rows:
        md.append(f"| {s} | {name} ({e}) | {r['f1'][1]:.2f} | {r['f1'][2]:.2f} | {r['f1'][3]:.2f} | {r['acc'][2]:.2f} | {r['comp'][2]:.2f} |")
        ax.plot(TOL, r["f1"], color=COL[name], lw=2, marker="o", ms=5, label=name)
    ax.axvspan(1, 2.5, color="#999999", alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_xticks(TOL); ax.set_xticklabels(TOL)
    ax.set_title(s, fontsize=11); ax.set_xlabel("tolerance (cm)")
    ax.grid(alpha=0.25, lw=0.6); ax.set_ylim(0, 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0][0].set_ylabel("F1 vs laser scan")
axes[0][0].text(1.05, 0.93, "construction\ntolerance zone", fontsize=7, color="#555555", va="top")
axes[0][-1].legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(f"{D}/f1_vs_tolerance.png", dpi=200)

# all exports (for the export-method discussion)
md += ["\n## Every export (F1 at 2 / 5 / 10 cm)\n", "| Run | F1 2cm | F1 5cm | F1 10cm |", "|---|---|---|---|"]
md += [f"| {k} | {r['f1'][1]:.3f} | {r['f1'][2]:.3f} | {r['f1'][3]:.3f} |" for k, r in runs.items()]

# tolerance verdict from per-point distances
md += ["\n## Tolerance verdict (per-point error vs laser; U95 must be <= 1/4 of band width)\n",
       "| Scene | Method | E50 (mm) | U95 (mm) | ACI 117 section <=300 (16 mm band) | IS 456 (18 mm) | ACI plumb (50 mm) |", "|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(f"{D}/dist_*.json")):
    s = os.path.basename(f)[5:-5]
    for m, r in json.load(open(f)).items():
        v = r["verdicts"]
        md.append(f"| {s} | {m} | {r['E50_mm']:.0f} | {r['U95_mm']:.0f} | {v['ACI117 section <=300mm (+10/-6)']} | "
                  f"{v['IS456 section (+12/-6)']} | {v['ACI117 plumb (+/-25)']} |")
md += ["\n## Element-level verdict (1 m planar laser elements; offset = median signed distance)\n",
       "| Scene | Method | elements measured | E50 (mm) | U95 (mm) | ACI 117 section <=300 | ACI plumb |", "|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(f"{D}/elem_*.json")):
    s_ = os.path.basename(f)[5:-5]
    for m, r in json.load(open(f)).items():
        v = r["verdicts"]
        md.append(f"| {s_} | {m} | {r['elements_measured']}/{r['elements_total']} | {r['E50_mm']:.0f} | {r['U95_mm']:.0f} | "
                  f"{v['ACI117 section <=300mm (+10/-6)']} | {v['ACI117 plumb (+/-25)']} |")
open(f"{D}/results_tables.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
