@echo off
setlocal

set PYTHON=venv\Scripts\python.exe
set MOD=models.train

echo ============================================================
echo  run_all.bat  --  mode=expanding  target-type=continuous
echo ============================================================

%PYTHON% -m %MOD% --model xgb      --horizon 1 --mode expanding --target-type continuous
if errorlevel 1 goto err
%PYTHON% -m %MOD% --model xgb      --horizon 5 --mode expanding --target-type continuous
if errorlevel 1 goto err

%PYTHON% -m %MOD% --model gru      --horizon 1 --mode expanding --target-type continuous
if errorlevel 1 goto err
%PYTHON% -m %MOD% --model gru      --horizon 5 --mode expanding --target-type continuous
if errorlevel 1 goto err

%PYTHON% -m %MOD% --model lstm     --horizon 1 --mode expanding --target-type continuous
if errorlevel 1 goto err
%PYTHON% -m %MOD% --model lstm     --horizon 5 --mode expanding --target-type continuous
if errorlevel 1 goto err

%PYTHON% -m %MOD% --model cnn_gru  --horizon 1 --mode expanding --target-type continuous
if errorlevel 1 goto err
%PYTHON% -m %MOD% --model cnn_gru  --horizon 5 --mode expanding --target-type continuous
if errorlevel 1 goto err

%PYTHON% -m %MOD% --model cnn_lstm --horizon 1 --mode expanding --target-type continuous
if errorlevel 1 goto err
%PYTHON% -m %MOD% --model cnn_lstm --horizon 5 --mode expanding --target-type continuous
if errorlevel 1 goto err

echo ============================================================
echo  All runs completed successfully.  Shutting down...
echo ============================================================
shutdown /s /t 60 /c "run_all.bat finished — shutting down in 60 s"
exit /b 0

:err
echo ============================================================
echo  ERROR: a run failed.  Shutdown cancelled.
echo ============================================================
exit /b 1
