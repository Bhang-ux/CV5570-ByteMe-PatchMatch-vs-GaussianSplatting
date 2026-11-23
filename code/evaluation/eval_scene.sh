#!/bin/bash
# usage: eval_scene.sh <scene> <run> [<run>...]   run = e.g. 3dgs_surface, patchmatch_fused
# needs the staged lab outputs in $OUTDIR (default: ./out)
S=$1; shift; G=${S%%_v*}
OUT=${OUTDIR:-./out}
EV=${ETH3D_EVAL:-./ETH3DMultiViewEvaluation}   # compiled ETH3D multi-view evaluation binary
H=$(dirname $0); R=$H/../../results/eth3d
args=()
for r in "$@"; do
  n=$r; [ "$r" = patchmatch_fused ] && n=patchmatch
  $EV --reconstruction_ply_path $OUT/${S}_$r.ply --ground_truth_mlp_path $OUT/gt/$G/scan_alignment.mlp \
      --tolerances 0.01,0.02,0.05,0.1,0.2,0.5 > $R/eval_${S}_$n.txt 2>&1
  echo "$S $n F1: $(grep F1 $R/eval_${S}_$n.txt)"
  args+=("$n=$OUT/${S}_$r.ply")
done
python3 $H/eth3d_dist.py $OUT/gt/$G $R/dist_${S}_new.json "${args[@]}"
python3 $H/eth3d_elements.py $OUT/gt/$G $R/elem_${S}_new.json "${args[@]}"
# merge into dist_/elem_ json
python3 - $R $S <<'P'
import json, os, sys
R, S = sys.argv[1:]
for k in ("dist", "elem"):
    f, n = f"{R}/{k}_{S}.json", f"{R}/{k}_{S}_new.json"
    d = json.load(open(f)) if os.path.exists(f) else {}
    d.update(json.load(open(n))); json.dump(d, open(f, "w"), indent=1); os.remove(n)
P
