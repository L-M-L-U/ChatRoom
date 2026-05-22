"""
管道测试脚本 — 独立测试 STT → LLM → TTS 各模块是否正常。

用法:
    python test_pipeline.py
    python test_pipeline.py --text "你好，今天天气怎么样"
"""

import argparse
import asyncio
import sys
import time
import traceback

import numpy as np


def test_stt():
    print("\n" + "=" * 50)
    print("  [1/3] STT 测试 (faster-whisper)")
    print("=" * 50)
    try:
        from stt_worker import transcribe_audio

        # 生成 1 秒静音 + 模拟人声噪声
        t0 = time.perf_counter()
        dummy = (np.random.randn(16000) * 0.01).astype(np.float32)
        result = transcribe_audio(dummy, 16000)
        elapsed = time.perf_counter() - t0
        print(f"  ✅ STT 完成 ({elapsed:.2f}s)")
        print(f"  转写结果: \"{result}\"")
        return True
    except Exception as e:
        print(f"  ❌ STT 失败: {e}")
        traceback.print_exc()
        return False


def test_llm():
    print("\n" + "=" * 50)
    print("  [2/3] LLM 测试 (Ollama + deepseek-r1:7b)")
    print("=" * 50)
    try:
        from llm_worker import LLMWorker

        w = LLMWorker("你是一个友好的中文助手，请用一句话回答。")
        t0 = time.perf_counter()
        result = w.chat("你好，请简单介绍一下你自己")
        elapsed = time.perf_counter() - t0
        print(f"  ✅ LLM 完成 ({elapsed:.2f}s)")
        print(f"  回复: \"{result}\"")
        return True
    except Exception as e:
        print(f"  ❌ LLM 失败: {e}")
        traceback.print_exc()
        return False


async def test_tts():
    print("\n" + "=" * 50)
    print("  [3/3] TTS 测试 (edge-tts + RVC)")
    print("=" * 50)
    try:
        from tts_worker import synthesize

        # 测试纯 edge-tts（空模型路径触发降级）
        t0 = time.perf_counter()
        audio = await synthesize("你好，欢迎使用语音聊天室。", "")
        elapsed = time.perf_counter() - t0
        print(f"  ✅ TTS 完成 ({elapsed:.2f}s)")
        print(f"  音频大小: {len(audio)} bytes ({len(audio) / 1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"  ❌ TTS 失败: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="语音聊天室管道测试")
    parser.add_argument("--skip-stt", action="store_true", help="跳过 STT 测试")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM 测试")
    parser.add_argument("--skip-tts", action="store_true", help="跳过 TTS 测试")
    args = parser.parse_args()

    print("=" * 50)
    print("  语音聊天室 — 管道模块测试")
    print("=" * 50)

    results = []

    if not args.skip_stt:
        results.append(test_stt())

    if not args.skip_llm:
        results.append(test_llm())

    if not args.skip_tts:
        results.append(asyncio.run(test_tts()))

    print("\n" + "=" * 50)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"  结果: {passed}/{total} 通过")
    print("=" * 50)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())