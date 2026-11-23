@echo off
cd /d "%~dp0"
if not exist logs mkdir logs
if not exist markers mkdir markers
title CV5570 runner - leave this window open
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1
echo ==========================================================
echo  CV5570 Byte Me runner. Leave this window open and the PC on.
echo  Locking the screen is fine. Claude monitors and restarts it.
echo ==========================================================
set CONDA=%USERPROFILE%\miniconda3\Scripts\conda.exe

:cycle
echo %date% %time% cycle start >> status.txt
if not exist "%~dp0env\python.exe" (
  echo [setup] creating Python 3.10 environment ...
  "%CONDA%" create -y -p "%~dp0env" -c conda-forge --override-channels python=3.10 >> logs\setup.log 2>&1
)
if not exist markers\pip_done (
  echo [setup] installing PyTorch + gsplat + tools ...
  "%~dp0env\python.exe" -m pip install --upgrade pip >> logs\setup.log 2>&1
  "%~dp0env\python.exe" -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124 >> logs\setup.log 2>&1
  "%~dp0env\python.exe" -m pip install "https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3%%2Bpt24cu124-cp310-cp310-win_amd64.whl" >> logs\setup.log 2>&1
  "%~dp0env\python.exe" -m pip install "numpy<2" tyro viser imageio scikit-learn tqdm "torchmetrics[image]" opencv-python Pillow tensorboard pyyaml matplotlib plyfile scipy py7zr splines tensorly >> logs\setup.log 2>&1
  "%~dp0env\python.exe" -m pip install "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787" "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" >> logs\setup.log 2>&1
  "%~dp0env\python.exe" -c "import torch, gsplat, tyro, nerfview, pycolmap, py7zr, plyfile; from gsplat import rasterization; print('VERIFY torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'gsplat', gsplat.__version__)" >> logs\setup.log 2>&1
  if errorlevel 1 (
    echo %date% %time% SETUP FAILED - waiting for Claude >> status.txt
    goto wait
  )
  echo ok > markers\pip_done
)
echo [run] experiments running ...
"%~dp0env\python.exe" run_all.py
echo %date% %time% run_all.py exited - waiting for Claude >> status.txt

:wait
rem Claude restarts the run by creating restart.flag (e.g. after fixing a script), or ends it with stop.flag
if exist stop.flag goto end
if exist restart.flag (
  del restart.flag
  goto cycle
)
timeout /t 60 /nobreak >nul
goto wait

:end
del stop.flag
echo Stopped by Claude. You can close this window.
pause
