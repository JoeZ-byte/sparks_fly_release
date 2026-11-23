@echo off
chcp 65001 >nul
cd /d %~dp0X86
if "%~1"=="" (
  python stream.py --input samples\sample_01_highway_002.mp4 --save-video out.mp4
) else (
  python stream.py %*
)