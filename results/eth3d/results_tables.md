## Main ETH3D table (F1 / accuracy / completeness at 2, 5, 10 cm)

| Scene | Method (export) | F1 2cm | F1 5cm | F1 10cm | Acc 5cm | Comp 5cm |
|---|---|---|---|---|---|---|
| facade | PatchMatch (fused) | 0.18 | 0.57 | 0.75 | 0.79 | 0.45 |
| facade | 2DGS (centers) | 0.04 | 0.27 | 0.53 | 0.61 | 0.17 |
| facade | 3DGS (centers) | 0.02 | 0.12 | 0.33 | 0.82 | 0.06 |
| courtyard | PatchMatch (fused) | 0.20 | 0.64 | 0.81 | 0.90 | 0.50 |
| courtyard | 2DGS (centers) | 0.09 | 0.37 | 0.58 | 0.64 | 0.26 |
| courtyard | 3DGS (surface) | 0.03 | 0.11 | 0.25 | 0.56 | 0.06 |
| terrace | PatchMatch (fused) | 0.50 | 0.81 | 0.90 | 0.96 | 0.70 |
| terrace | 2DGS (centers) | 0.09 | 0.29 | 0.46 | 0.53 | 0.20 |
| terrace | 3DGS (surface) | 0.02 | 0.09 | 0.22 | 0.21 | 0.05 |

## Every export (F1 at 2 / 5 / 10 cm)

| Run | F1 2cm | F1 5cm | F1 10cm |
|---|---|---|---|
| courtyard_2dgs_centers | 0.087 | 0.374 | 0.578 |
| courtyard_2dgs_surface | 0.103 | 0.238 | 0.366 |
| courtyard_2dgs_surfstrict | 0.083 | 0.214 | 0.340 |
| courtyard_3dgs_centers | 0.011 | 0.092 | 0.275 |
| courtyard_3dgs_surface | 0.028 | 0.109 | 0.247 |
| courtyard_3dgs_surfstrict | 0.008 | 0.029 | 0.071 |
| courtyard_patchmatch | 0.201 | 0.639 | 0.814 |
| facade_2dgs_centers | 0.044 | 0.267 | 0.535 |
| facade_2dgs_surface | 0.081 | 0.214 | 0.368 |
| facade_3dgs_centers | 0.016 | 0.116 | 0.327 |
| facade_3dgs_surface | 0.019 | 0.075 | 0.150 |
| facade_patchmatch | 0.179 | 0.573 | 0.750 |
| facade_v50_3dgs_surface | 0.006 | 0.038 | 0.097 |
| facade_v50_patchmatch | 0.103 | 0.423 | 0.605 |
| terrace_2dgs_centers | 0.087 | 0.285 | 0.463 |
| terrace_2dgs_surface | 0.030 | 0.098 | 0.216 |
| terrace_2dgs_surfstrict | 0.029 | 0.099 | 0.217 |
| terrace_3dgs_centers | 0.010 | 0.075 | 0.233 |
| terrace_3dgs_surface | 0.019 | 0.087 | 0.220 |
| terrace_3dgs_surfstrict | 0.018 | 0.061 | 0.137 |
| terrace_patchmatch | 0.503 | 0.814 | 0.895 |

## Tolerance verdict (per-point error vs laser; U95 must be <= 1/4 of band width)

| Scene | Method | E50 (mm) | U95 (mm) | ACI 117 section <=300 (16 mm band) | IS 456 (18 mm) | ACI plumb (50 mm) |
|---|---|---|---|---|---|---|
| courtyard | patchmatch | 14 | 171 | INADEQUATE | INADEQUATE | INADEQUATE |
| courtyard | 3dgs_surface | 99 | 401 | INADEQUATE | INADEQUATE | INADEQUATE |
| courtyard | 3dgs_surfstrict | 71 | 373 | INADEQUATE | INADEQUATE | INADEQUATE |
| courtyard | 3dgs_centers | 66 | 415 | INADEQUATE | INADEQUATE | INADEQUATE |
| courtyard | 2dgs_surface | 88 | 380 | INADEQUATE | INADEQUATE | INADEQUATE |
| courtyard | 2dgs_surfstrict | 62 | 336 | INADEQUATE | INADEQUATE | INADEQUATE |
| courtyard | 2dgs_centers | 54 | 387 | INADEQUATE | INADEQUATE | INADEQUATE |
| facade | PatchMatch | 24 | 198 | INADEQUATE | INADEQUATE | INADEQUATE |
| facade | 3DGS_surface | 145 | 448 | INADEQUATE | INADEQUATE | INADEQUATE |
| facade | 3DGS_centres | 54 | 382 | INADEQUATE | INADEQUATE | INADEQUATE |
| facade | 2DGS_surface | 102 | 347 | INADEQUATE | INADEQUATE | INADEQUATE |
| facade | 2DGS_centres | 64 | 350 | INADEQUATE | INADEQUATE | INADEQUATE |
| facade_v50 | patchmatch | 24 | 194 | INADEQUATE | INADEQUATE | INADEQUATE |
| facade_v50 | 3dgs_surface | 144 | 450 | INADEQUATE | INADEQUATE | INADEQUATE |
| facade_v50 | 3dgs_centers | 79 | 408 | INADEQUATE | INADEQUATE | INADEQUATE |
| terrace | patchmatch | 7 | 136 | INADEQUATE | INADEQUATE | INADEQUATE |
| terrace | 3dgs_surface | 148 | 437 | INADEQUATE | INADEQUATE | INADEQUATE |
| terrace | 3dgs_surfstrict | 33 | 233 | INADEQUATE | INADEQUATE | INADEQUATE |
| terrace | 3dgs_centers | 103 | 430 | INADEQUATE | INADEQUATE | INADEQUATE |
| terrace | 2dgs_surface | 109 | 410 | INADEQUATE | INADEQUATE | INADEQUATE |
| terrace | 2dgs_surfstrict | 89 | 383 | INADEQUATE | INADEQUATE | INADEQUATE |
| terrace | 2dgs_centers | 55 | 384 | INADEQUATE | INADEQUATE | INADEQUATE |

## Element-level verdict (1 m planar laser elements; offset = median signed distance)

| Scene | Method | elements measured | E50 (mm) | U95 (mm) | ACI 117 section <=300 | ACI plumb |
|---|---|---|---|---|---|---|
| courtyard | patchmatch | 443/730 | 11 | 30 | INADEQUATE | INADEQUATE |
| courtyard | 3dgs_surface | 128/730 | 84 | 193 | INADEQUATE | INADEQUATE |
| courtyard | 3dgs_surfstrict | 26/730 | 98 | 164 | INADEQUATE | INADEQUATE |
| courtyard | 3dgs_centers | 14/730 | 9 | 117 | INADEQUATE | INADEQUATE |
| courtyard | 2dgs_surface | 272/730 | 93 | 209 | INADEQUATE | INADEQUATE |
| courtyard | 2dgs_surfstrict | 179/730 | 78 | 197 | INADEQUATE | INADEQUATE |
| courtyard | 2dgs_centers | 234/730 | 12 | 83 | INADEQUATE | INADEQUATE |
| facade | PatchMatch | 558/1078 | 13 | 47 | INADEQUATE | INADEQUATE |
| facade | 3DGS_surface | 120/1078 | 95 | 212 | INADEQUATE | INADEQUATE |
| facade | 3DGS_centres | 37/1078 | 3 | 129 | INADEQUATE | INADEQUATE |
| facade | 2DGS_surface | 412/1078 | 117 | 215 | INADEQUATE | INADEQUATE |
| facade | 2DGS_centres | 212/1078 | 24 | 118 | INADEQUATE | INADEQUATE |
| facade_v50 | patchmatch | 303/1078 | 12 | 53 | INADEQUATE | INADEQUATE |
| terrace | patchmatch | 125/160 | 5 | 18 | INADEQUATE | MARGINAL |
| terrace | 3dgs_surface | 27/160 | 58 | 124 | INADEQUATE | INADEQUATE |
| terrace | 3dgs_surfstrict | 15/160 | 19 | 96 | INADEQUATE | INADEQUATE |
| terrace | 3dgs_centers | 5/160 | 10 | 120 | INADEQUATE | INADEQUATE |
| terrace | 2dgs_surface | 54/160 | 59 | 176 | INADEQUATE | INADEQUATE |
| terrace | 2dgs_surfstrict | 47/160 | 63 | 164 | INADEQUATE | INADEQUATE |
| terrace | 2dgs_centers | 59/160 | 9 | 47 | INADEQUATE | INADEQUATE |
