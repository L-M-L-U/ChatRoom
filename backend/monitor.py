import functools
import os
import time
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "timings.log")


def _ensure_log():
    os.makedirs(LOG_DIR, exist_ok=True)


def _log_line(line: str):
    _ensure_log()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_time(stage: str = None):
    """装饰器 — 记录函数执行耗时。

    用法:
        @log_time("stt")
        def transcribe_audio(...): ...

    也可不带参数直接使用 @log_time。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            label = stage or func.__name__
            ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            _log_line(f"{ts},{label},{elapsed:.4f}")
            return result
        return wrapper
    return decorator


class PipelineTimer:
    """上下文管理器，追踪单次请求各阶段耗时。"""

    def __init__(self):
        self._marks = {}
        self._start = None

    def start(self):
        self._start = time.perf_counter()
        return self

    def mark(self, name: str):
        self._marks[name] = time.perf_counter()

    def report(self) -> dict:
        if self._start is None:
            return {}
        now = time.perf_counter()
        stt = self._marks.get("stt", self._start) - self._start
        llm = self._marks.get("llm", now) - self._marks.get("stt", self._start)
        tts = self._marks.get("tts", now) - self._marks.get("llm", self._start)
        total = now - self._start
        return {"stt": stt, "llm": llm, "tts": tts, "total": total}

    def log(self):
        data = self.report()
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        _ensure_log()
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"{ts},{data['stt']:.4f},{data['llm']:.4f},{data['tts']:.4f},{data['total']:.4f}\n"
            )