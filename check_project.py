#!/usr/bin/env python3
"""
项目健康检查脚本 — 静态检查文件、依赖、服务状态。

用法:
    python check_project.py
    python check_project.py --rvc-model models/default.pth
"""

import argparse
import importlib
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── 各阶段核心文件 ──
CORE_FILES = {
    "阶段0 (基础)": [
        "requirements.txt",
        "README.md",
        "start_server.bat",
    ],
    "阶段1 (STT)": [
        "backend/stt_worker.py",
    ],
    "阶段2 (LLM)": [
        "backend/llm_worker.py",
    ],
    "阶段3 (TTS)": [
        "backend/tts_worker.py",
    ],
    "阶段4 (管道)": [
        "backend/main.py",
    ],
    "阶段5 (角色)": [
        "backend/role_manager.py",
        "config/roles/lisa.md",
    ],
    "阶段6 (记忆)": [
        "backend/memory_worker.py",
    ],
    "阶段7 (监控)": [
        "backend/monitor.py",
    ],
    "阶段8 (训练)": [
        "backend/train_rvc.py",
    ],
}

RESULTS = {"pass": 0, "fail": 0, "skip": 0}


def ok(msg: str):
    RESULTS["pass"] += 1
    print(f"  ✅ PASS  {msg}")


def fail(msg: str):
    RESULTS["fail"] += 1
    print(f"  ❌ FAIL  {msg}")


def skip(msg: str):
    RESULTS["skip"] += 1
    print(f"  ⏭  SKIP  {msg}")


# ────────────────────── 1. 文件检查 ──────────────────────


def check_files():
    print("\n═══════════════════════════════════════════")
    print("  1. 必需文件检查")
    print("═══════════════════════════════════════════")

    all_ok = True
    for stage, files in CORE_FILES.items():
        for f in files:
            full = ROOT / f
            if full.exists():
                ok(f"{stage} — {f}")
            else:
                fail(f"{stage} — {f}  (缺失)")
                all_ok = False
    return all_ok


# ────────────────────── 2. 依赖检查 ──────────────────────


def check_dependencies():
    print("\n═══════════════════════════════════════════")
    print("  2. Python 依赖检查")
    print("═══════════════════════════════════════════")

    req = ROOT / "requirements.txt"
    if not req.exists():
        skip("requirements.txt 不存在，跳过依赖检查")
        return False

    raw = req.read_text(encoding="utf-8")
    pkgs = [line.strip() for line in raw.splitlines() if line.strip() and not line.startswith("#")]

    # 映射包名到 import 名（部分不一致的）
    IMPORT_MAP = {
        "Pillow": "PIL",
        "sounddevice": "sounddevice",
        "edge-tts": "edge_tts",
        "faster-whisper": "faster_whisper",
        "RVC-inference": "rvc_inference",
        "pyyaml": "yaml",
        "chromadb": "chromadb",
        "sentence-transformers": "sentence_transformers",
    }

    all_ok = True
    for pkg in pkgs:
        try:
            # 处理版本限定符
            name = pkg.split(">=")[0].split("==")[0].split("<")[0].strip()
            import_name = IMPORT_MAP.get(name, name.replace("-", "_"))
            importlib.import_module(import_name)
            ok(f"{name}")
        except ImportError:
            fail(f"{name}  (未安装)")
            all_ok = False
    return all_ok


# ────────────────────── 3. Ollama 检查 ──────────────────────


def check_ollama():
    print("\n═══════════════════════════════════════════")
    print("  3. Ollama 服务检查")
    print("═══════════════════════════════════════════")

    # 检查进程
    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq ollama.exe", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            ollama_running = "ollama.exe" in r.stdout
        else:
            r = subprocess.run(
                ["pgrep", "-x", "ollama"],
                capture_output=True, timeout=5,
            )
            ollama_running = r.returncode == 0
    except Exception:
        ollama_running = False

    if not ollama_running:
        fail("Ollama 进程未运行")
        return False
    ok("Ollama 进程运行中")

    # 检查模型
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if "deepseek-r1:7b" in r.stdout:
            ok("deepseek-r1:7b 模型存在")
        else:
            fail("deepseek-r1:7b 模型未找到 (运行 ollama pull deepseek-r1:7b)")
            return False
    except FileNotFoundError:
        fail("ollama 命令未找到 (未安装或不在 PATH)")
        return False
    except Exception as e:
        fail(f"查询 Ollama 模型失败: {e}")
        return False

    return True


# ────────────────────── 4. RVC 模型检查 ──────────────────────


def check_rvc_model(rvc_path: str):
    print("\n═══════════════════════════════════════════")
    print("  4. RVC 模型文件检查")
    print("═══════════════════════════════════════════")

    if not rvc_path:
        skip("未指定 RVC 模型路径 (使用 --rvc-model)")
        return True

    model_file = Path(rvc_path)
    if not model_file.is_absolute():
        model_file = ROOT / model_file

    if model_file.exists():
        ok(f"RVC 模型: {model_file}")
        return True
    else:
        fail(f"RVC 模型不存在: {model_file}")
        return False


# ────────────────────── 5. 端口检查 ──────────────────────


def check_port(port: int = 8765):
    print("\n═══════════════════════════════════════════")
    print(f"  5. 端口 {port} 检查")
    print("═══════════════════════════════════════════")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", port))
        ok(f"端口 {port} 可用")
        return True
    except OSError:
        fail(f"端口 {port} 已被占用")
        return False
    finally:
        sock.close()


# ────────────────────── 主入口 ──────────────────────


def main():
    parser = argparse.ArgumentParser(description="语音聊天室项目健康检查")
    parser.add_argument("--rvc-model", default="", help="RVC 模型文件路径")
    args = parser.parse_args()

    print("=" * 50)
    print("  🩺 语音聊天室 — 项目健康检查")
    print("=" * 50)

    check_files()
    check_dependencies()
    check_ollama()
    check_rvc_model(args.rvc_model)
    check_port(8765)

    # ── 汇总报告 ──
    print("\n" + "=" * 50)
    print(f"  报告: ✅ {RESULTS['pass']}  PASS  ❌ {RESULTS['fail']}  FAIL  ⏭ {RESULTS['skip']}  SKIP")
    print("=" * 50)

    return 0 if RESULTS["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())