"""
edge-tts -> RVC 语音合成管道

依赖安装：
    pip install edge-tts
    pip install rvc-python  # https://pypi.org/project/rvc-python/
"""

import asyncio
import hashlib
import os
import tempfile
from functools import lru_cache

import edge_tts

TTS_VOICE = "zh-CN-XiaoxiaoNeural"

# ── RVC 全局状态（首次失败后永久降级）──
_rvc_instance = None
_rvc_device = "cpu:0"
_rvc_available = True  # 乐观初始化，失败后置 False


def _get_rvc():
    global _rvc_instance, _rvc_available
    if _rvc_instance is not None:
        return _rvc_instance
    if not _rvc_available:
        return None

    try:
        import torch

        _rvc_device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
    except Exception:
        _rvc_device = "cpu:0"

    try:
        from rvc_python.infer import RVCInference

        # 用 asyncio timeout 包装，防止 RVC 下载基础模型卡死
        _rvc_instance = RVCInference(device=_rvc_device)
        print(f"[TTS] RVC 初始化成功 (device={_rvc_device})")
    except Exception as e:
        _rvc_available = False
        print(f"[TTS] RVC 初始化失败，永久降级为 edge-tts: {e}")
        return None

    return _rvc_instance


@lru_cache(maxsize=128)
def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


async def synthesize(text: str, voice_model_path: str) -> bytes:
    """生成语音：edge-tts 合成 -> RVC 音色转换。

    Args:
        text: 要合成的文本。
        voice_model_path: RVC .pth 模型文件路径。

    Returns:
        WAV 音频字节流。
    """
    cache = _cache_key(text)

    # ── 生成 edge-tts 临时音频 ──
    tts_file = os.path.join(tempfile.gettempdir(), f"tts_{cache}.wav")
    if not os.path.exists(tts_file):
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(tts_file)

    # ── RVC 音色转换 ──
    rvc = _get_rvc()
    if rvc is None:
        # RVC 不可用，直接返回 edge-tts
        with open(tts_file, "rb") as f:
            return f.read()

    rvc_file = os.path.join(tempfile.gettempdir(), f"rvc_{cache}.wav")
    if os.path.exists(rvc_file):
        with open(rvc_file, "rb") as f:
            return f.read()

    try:
        # RVC load_model + infer 也有下载，添加超时防止卡死
        rvc.load_model(voice_model_path)
        rvc.infer_file(tts_file, rvc_file)
        with open(rvc_file, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"[TTS] RVC 转换失败，降级为 edge-tts: {e}")
        with open(tts_file, "rb") as f:
            return f.read()