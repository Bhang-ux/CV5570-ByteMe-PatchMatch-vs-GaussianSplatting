set -e
export QT_QPA_PLATFORM=offscreen
cd "${PROJ:-.}"
colmap feature_extractor --database_path db.db --image_path images \
  --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL \
  --SiftExtraction.use_gpu 0 > fe.log 2>&1 && echo FE_DONE
colmap sequential_matcher --database_path db.db \
  --SequentialMatching.overlap 10 --SequentialMatching.loop_detection 0 \
  --SiftMatching.use_gpu 0 > sm.log 2>&1 && echo SM_DONE
mkdir -p sparse
colmap mapper --database_path db.db --image_path images --output_path sparse > map.log 2>&1 && echo MAP_DONE
ls sparse
colmap model_analyzer --path sparse/0 2>&1 | grep -E "Registered|Points|Observations|track|reprojection"
colmap image_undistorter --image_path images --input_path sparse/0 --output_path undist --output_type COLMAP > und.log 2>&1 && echo UNDIST_DONE
