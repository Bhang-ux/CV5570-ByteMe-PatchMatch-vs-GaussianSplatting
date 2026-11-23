"""Turn the GPU deliverable zips into the paper's tables and figures.

Usage:  python make_results.py --zips <any of the deliverables_*.zip> --out ../../results/tables_figures
Reads whatever is present and skips the rest:
  deliverables_eth3d.zip / deliverables_ablations.zip  -> ETH3D F1 table, F1-vs-tolerance chart, U95 + verdict table, ablations
  deliverables_site1.zip + deliverables_patchmatch.zip -> site cross-method agreement (splats vs PatchMatch, same frame)
U95 from ETH3D = tolerance at which accuracy reaches 95 % (= 95th percentile of reconstruction-to-laser distance),
log-interpolated between the evaluated tolerances.
"""
import argparse, glob, io, json, os, re, sys, zipfile, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_harness import parse_eth3d, tolerance_verdict, TOLERANCES_MM, read_ply, cloud_metrics, voxel_down

NICE = {"patchmatch": "PatchMatch MVS", "3dgs_surface": "3DGS (fused surface)", "2dgs_surface": "2DGS (fused surface)",
        "3dgs_centers": "3DGS (Gaussian centres)", "2dgs_centers": "2DGS (Gaussian centres)",
        "2dgs_noreg_surface": "2DGS no regularisers", "3dgs_7k_surface": "3DGS 7k steps", "2dgs_7k_surface": "2DGS 7k steps"}

def u95_from_curve(tols_m, acc):
    t, a = np.asarray(tols_m, float), np.asarray(acc, float)
    if a.max() < 0.95:
        return float("inf")
    if a[0] >= 0.95:
        return t[0] * 1000
    i = int(np.argmax(a >= 0.95))
    lt = np.interp(0.95, [a[i - 1], a[i]], np.log([t[i - 1], t[i]]))
    return float(np.exp(lt) * 1000)

def pct_from_curve(tols_m, acc, q):
    """Distance below which a fraction q of reconstructed points lie (q=.5 -> median point error)."""
    t, a = np.asarray(tols_m, float), np.asarray(acc, float)
    if a.max() < q:
        return float("inf")
    if a[0] >= q:
        return t[0] * 1000
    i = int(np.argmax(a >= q))
    return float(np.exp(np.interp(q, [a[i - 1], a[i]], np.log([t[i - 1], t[i]]))) * 1000)

def collect_eth3d(zips):
    rows = []
    pat_k = re.compile(r"eval_(?P<scene>[a-z_0-9]+?)_(?P<method>patchmatch|[23]dgs_(?:surface|centers))\.txt$")
    pat_c = re.compile(r"eval__(?P<scene>[^_].*?)__(?P<method>.+)\.txt$")
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                b = os.path.basename(n)
                m = pat_c.match(b) or pat_k.match(b)
                if not m:
                    continue
                res = parse_eth3d(zf.read(n).decode(errors="ignore"))
                if not res:
                    continue
                tols = sorted(res)
                rows.append({"source": os.path.basename(z), "scene": m["scene"], "method": m["method"],
                             **{f"F1@{t*100:g}cm": res[t]["f1"] for t in tols},
                             **{f"acc@{t*100:g}cm": res[t]["accuracy"] for t in tols},
                             **{f"com@{t*100:g}cm": res[t]["completeness"] for t in tols},
                             "U95_mm": round(u95_from_curve(tols, [res[t]["accuracy"] for t in tols]), 1),
                             "E50_mm": round(pct_from_curve(tols, [res[t]["accuracy"] for t in tols], .5), 1)})
    return rows

def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def md_table(rows, cols, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)

def main_tables(rows, out):
    full = [r for r in rows if "_v" not in r["scene"] and not any(k in r["method"] for k in ("noreg", "7k"))]
    methods = sorted({r["method"] for r in full}, key=lambda m: list(NICE).index(m) if m in NICE else 99)
    mean_rows = []
    for m in methods:
        rs = [r for r in full if r["method"] == m]
        agg = {"method": NICE.get(m, m), "scenes": len(rs)}
        for k in ("F1@1cm", "F1@2cm", "F1@5cm", "F1@10cm", "acc@2cm", "com@2cm"):
            v = [r[k] for r in rs if k in r]
            agg[k] = round(float(np.mean(v)), 3) if v else ""
        u = [r["U95_mm"] for r in rs]
        agg["U95_mm"] = round(float(np.median(u)), 1) if u else ""
        e = [r["E50_mm"] for r in rs]
        agg["E50_mm"] = round(float(np.median(e)), 1) if e else ""
        mean_rows.append(agg)
    cols = ["method", "scenes", "F1@1cm", "F1@2cm", "F1@5cm", "F1@10cm", "acc@2cm", "com@2cm", "E50_mm", "U95_mm"]
    md = ["## Table 1. ETH3D, mean over full scenes\n", md_table(mean_rows, cols, cols)]
    verdict_rows = []
    for r in mean_rows:
        if r["U95_mm"] == "":
            continue
        row = {"method": r["method"], "U95_mm": r["U95_mm"]}
        for check, band in TOLERANCES_MM.items():
            width = band[1] - band[0]; ratio = r["U95_mm"] / width
            row[check] = "Adequate" if ratio <= .25 else ("Marginal" if ratio <= .5 else "Inadequate")
        verdict_rows.append(row)
    if verdict_rows:
        vc = ["method", "U95_mm"] + list(TOLERANCES_MM)
        md += ["\n## Table 2. Tolerance verdict, per point (U95 = 95th pct. of reconstruction-to-laser distance; conservative, includes floaters. E50 = median point error)\n", md_table(verdict_rows, vc, vc)]
    abl = [r for r in rows if "_v" in r["scene"] or any(k in r["method"] for k in ("noreg", "7k"))]
    if abl:
        ac = ["scene", "method", "F1@2cm", "F1@5cm", "E50_mm", "U95_mm"]
        md += ["\n## Table 3. Ablations\n", md_table(abl, ac, ac)]
    open(os.path.join(out, "results_tables.md"), "w").write("\n".join(md) + "\n")
    write_csv(os.path.join(out, "eth3d_all_runs.csv"), rows)
    write_csv(os.path.join(out, "eth3d_means.csv"), mean_rows)
    write_csv(os.path.join(out, "verdict.csv"), verdict_rows)
    return mean_rows

def f1_chart(rows, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    full = [r for r in rows if "_v" not in r["scene"] and "centers" not in r["method"]
            and not any(k in r["method"] for k in ("noreg", "7k"))]
    if not full:
        return
    tol_keys = sorted({k for r in full for k in r if k.startswith("F1@")}, key=lambda k: float(k[3:-2]))
    tols = [float(k[3:-2]) for k in tol_keys]
    colors = {"patchmatch": "#3A6EA5", "3dgs_surface": "#D9822B", "2dgs_surface": "#8C5A2B"}
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for m in sorted({r["method"] for r in full}):
        rs = [r for r in full if r["method"] == m]
        y = [np.mean([r[k] for r in rs]) for k in tol_keys]
        ax.plot(tols, y, marker="o", lw=2, color=colors.get(m), label=f"{NICE.get(m, m)} (n={len(rs)})")
    ax.axvspan(0, 1.8, color="#E3A15A", alpha=.15)
    ax.text(0.95, 0.03, "IS 456 section\nband (18 mm)", fontsize=8, ha="center", color="#8C5A2B")
    ax.set_xscale("log"); ax.set_xticks(tols); ax.set_xticklabels([f"{t:g}" for t in tols])
    ax.set_xlabel("Tolerance (cm)"); ax.set_ylabel("F1 vs laser scan"); ax.set_ylim(0, 1)
    ax.grid(alpha=.3); ax.legend(frameon=False); ax.set_title("ETH3D: F1 against laser ground truth")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_f1_vs_tolerance.png")); plt.close(fig)

def site_agreement(zips, out, workdir):
    plys = {}
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                b = os.path.basename(n)
                if b.startswith("site1_") and b.endswith(".ply"):
                    p = os.path.join(workdir, b); open(p, "wb").write(zf.read(n)); plys[b[6:-4]] = p
    if "patchmatch_fused" not in plys:
        return None
    ref, _ = read_ply(plys["patchmatch_fused"])
    rows = []
    for k, p in sorted(plys.items()):
        if k == "patchmatch_fused":
            continue
        rec, _ = read_ply(p)
        m = cloud_metrics(rec, ref, tolerances=(0.01, 0.02, 0.05, 0.1), voxel=0.002)
        rows.append({"method": k, **{kk: round(v, 4) if isinstance(v, float) else v for kk, v in m.items()}})
    write_csv(os.path.join(out, "site_agreement_vs_patchmatch_model_units.csv"), rows)
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--zips", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, "_unzipped"); os.makedirs(work, exist_ok=True)
    rows = collect_eth3d(a.zips)
    print(f"ETH3D evaluations found: {len(rows)}")
    if rows:
        means = main_tables(rows, a.out); f1_chart(rows, a.out)
        for r in means: print(r)
    site = site_agreement(a.zips, a.out, work)
    if site:
        for r in site: print("site:", r["method"], "chamfer", r["chamfer_m"])
    print("wrote", sorted(os.listdir(a.out)))
