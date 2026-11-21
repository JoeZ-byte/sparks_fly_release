@echo off
rem 双击运行：默认播放 sample_01；把视频拖到本文件上也可用 %%1 传入
chcp 65001 >nul
cd /d %~dp0
if "%~1"=="" (
    stream_demo.exe --input samples\sample_01_highway_002.mp4
) else (
    stream_demo.exe --input "%~1"
)
pause