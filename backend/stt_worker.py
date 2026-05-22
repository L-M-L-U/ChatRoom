import sys
import traceback

import numpy as np
from faster_whisper import WhisperModel

_model = None
_model_loaded = False


def _get_model():
    global _model, _model_loaded
    if _model_loaded:
        return _model

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[STT] 加载模型 (device={device}, compute={compute_type})…")
        _model = WhisperModel("base", device=device, compute_type=compute_type)
        _model_loaded = True
        print(f"[STT] 模型加载完成")
    except Exception as e:
        print(f"[STT] 模型加载失败: {e}")
        traceback.print_exc(file=sys.stderr)
        raise
    return _model


def transcribe_audio(audio_numpy: np.ndarray, samplerate: int) -> str:
    model = _get_model()
    try:
        segments, info = model.transcribe(audio_numpy, language="zh", beam_size=5, vad_filter=True)
        result = "".join(seg.text for seg in segments)
        print(f"[STT] 识别结果: \"{result}\"")
        return result
    except Exception as e:
        print(f"[STT] 转写失败: {e}")
        traceback.print_exc(file=sys.stderr)
        return ""