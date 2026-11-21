@echo off
rem Double-click to run. Default: sample_01. Drag a video file onto this file to use it.
chcp 65001 >nul
cd /d %~dp0
set PY=python
where python >nul 2>nul || set PY=py
if "%~1"=="" (
    %PY% src\stream_demo.py --input samples\sample_01_highway_002.mp4
) else (
    %PY% src\stream_demo.py --input "%~1"
)
pause