@echo off
chcp 65001 >nul
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set SSL_CERT_FILE=
D:\ProgramData\anaconda3\envs\rvc_env\python.exe backend\main.py
pause