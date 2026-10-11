@echo off
echo =======================================================================
echo  DEEP LEARNING ESSENTIALS - INNOVATIVE ASSIGNMENT II
echo  Topic: Image Noise Reduction Using Convolution
echo =======================================================================

REM Try Python in PATH first, else fallback to Microsoft Store Python
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python src\image_noise_reduction.py
) else (
    "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" src\image_noise_reduction.py
)

echo.
echo Experiment execution finished. Check 'results/' and 'screenshots/' folders.
pause
