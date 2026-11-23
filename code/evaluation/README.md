# Evaluation harness (Shubhang's evaluation arm), built 27 Oct

`eval_harness.py`: everything the results section needs. It only uses numpy + scipy, so it runs anywhere.
- read/write PLY, voxel downsampling (fair density)
- Umeyama similarity alignment + point-to-plane ICP (registration to BIM/GT)
- metric scale from two parallel planes (slab-soffit), the same method as phase 1
- ETH3D-style accuracy / completeness / F1 at 1, 2, 5, 10, 20, 50 cm + Chamfer
- ETH3D evaluator output parser
- **Tolerance verdict**: a method is ADEQUATE for a check if its 95th-percentile error is <= 1/4 of the
  tolerance band width (the metrology 4:1 rule), MARGINAL up to 1/2, otherwise INADEQUATE.
  Bands come from ACI 117-10 / IS 456. **Verify the clause numbers before the paper is final.**
- Per-element signed deviation vs BIM surfaces + DBSCAN clustering of deviation blobs

`test_harness.py`: self-test that recovers known answers on our own dense cloud. **All 7 passed:**
scale error 7e-16, rotation 2e-6 deg, ICP back to noise level (4.8 mm at 3 mm noise),
known 25 mm column-face offset measured as 24.99 mm, one injected bulge found as exactly one cluster.

Usage: `python eval_harness.py compare --ref gt.ply --rec pm.ply 3dgs.ply 2dgs.ply --names PatchMatch 3DGS 2DGS --icp 0.1 --out results/`
