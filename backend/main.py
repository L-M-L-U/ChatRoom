"""
Voice Chat Room — WebSocket server

Pipeline: Audio → STT → LLM → TTS → Audio
"""

import asyncio
import json
import os
import struct
import sys
import time
import traceback

import numpy as np
import websockets
import websockets.asyncio.server

from memory_worker import MemoryWorker
from monitor import PipelineTimer
from role_manager import get_role_list, load_role
from stt_worker import transcribe_audio
from llm_worker import LLMWorker
from tts_worker import synthesize

HOST = "localhost"
PORT = 8765
DEFAULT_ROLE = "lisa"
RVC_MODEL = os.environ.get("RVC_MODEL_PATH", "models/default.pth")

_memory = MemoryWorker()

# ── 角色状态 ──
_current_role = {}


def _update_role(role_name: str) -> bool:
    config = load_role(role_name)
    if not config:
        return False
    _current_role["name"] = config["name"]
    _current_role["system_prompt"] = config["system_prompt"]
    _current_role["model_name"] = config.get("model_name", "deepseek-r1:7b")
    _current_role["rvc_model"] = config.get("rvc_model_path", RVC_MODEL)
    print(f"[Role] 切换至: {config['name']}")
    return True


_update_role(DEFAULT_ROLE)


def _int16_to_float32(buf: bytes) -> np.ndarray:
    samples = struct.unpack(f"<{len(buf) // 2}h", buf)
    return np.array(samples, dtype=np.float32) / 32768.0


async def handle_ws(ws):
    """长连接：循环处理控制消息和音频消息。"""
    while True:
        try:
            raw = await ws.recv()
        except websockets.exceptions.ConnectionClosed:
            print("[WS] 客户端断开连接")
            break

        pipeline_start = time.perf_counter()

        # ── 文本 → JSON 控制消息 ──
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
                if data.get("type") == "select_role":
                    ok = _update_role(data["role"])
                    roles = get_role_list()
                    await ws.send(json.dumps({
                        "type": "role_selected",
                        "success": ok,
                        "current_role": _current_role["name"],
                        "roles": roles,
                    }))
                    print(f"[WS] 角色切换: {data['role']} -> {'成功' if ok else '失败'}")
            except json.JSONDecodeError as e:
                print(f"[WS] JSON 解析失败: {e}")
            continue

        # ── 二进制 → PCM 音频 ──
        audio_len = len(raw)
        print(f"[WS] 收到音频: {audio_len} bytes")

        try:
            # ── STT ──
            print("[PIPE] 开始 STT…")
            t0 = time.perf_counter()
            audio_float32 = _int16_to_float32(raw)
            stt_text = await asyncio.to_thread(transcribe_audio, audio_float32, 16000)
            t_stt = time.perf_counter() - t0
            print(f"[STT] 用户说: \"{stt_text}\"")

            if not stt_text.strip():
                print("[STT] 转写结果为空，使用默认提示")
                stt_text = "请礼貌地请用户再说一遍"

            # ── LLM ──
            print("[PIPE] 开始 LLM…")
            t0 = time.perf_counter()
            worker = LLMWorker(_current_role.get("system_prompt", ""), memory=_memory)
            reply_text = await asyncio.to_thread(worker.chat, stt_text)
            t_llm = time.perf_counter() - t0
            print(f"[LLM] 回复: \"{reply_text}\"")

            # 发送 JSON 文本回复（包含 STT 识别文字和 LLM 回复）
            try:
                await ws.send(json.dumps({
                    "type": "reply",
                    "stt_text": stt_text,
                    "text": reply_text,
                    "role": _current_role["name"],
                }))
            except websockets.exceptions.ConnectionClosed:
                print("[WS] 客户端在 LLM 阶段断开")
                break

            # ── TTS ──
            print("[PIPE] 开始 TTS (synthesize)…")
            t0 = time.perf_counter()
            audio_bytes = await synthesize(reply_text, _current_role["rvc_model"])
            t_tts = time.perf_counter() - t0
            print(f"[TTS] 耗时 {t_tts:.2f}s | 音频大小: {len(audio_bytes)} bytes")

            # 发送音频回复
            try:
                await ws.send(audio_bytes)
            except websockets.exceptions.ConnectionClosed:
                print("[WS] 客户端在 TTS 阶段断开")
                break

            total = time.perf_counter() - pipeline_start
            print(f"[PIPE] 总耗时 {total:.2f}s | STT={t_stt:.2f}s LLM={t_llm:.2f}s TTS={t_tts:.2f}s")

        except Exception as e:
            print(f"[ERROR] 管道异常: {e}")
            traceback.print_exc(file=sys.stderr)
            try:
                fb = "处理请求时出错，请重试。"
                await ws.send(json.dumps({"type": "reply", "text": fb}))
                try:
                    fb_audio = await synthesize(fb, _current_role["rvc_model"])
                    await ws.send(fb_audio)
                except Exception as tts_err:
                    print(f"[ERROR] 降级 TTS 也失败: {tts_err}")
            except Exception as send_err:
                print(f"[ERROR] 发送错误消息失败: {send_err}")


async def warmup():
    """预热模型，避免首请求超慢。"""
    print("[Warmup] 预热 STT 模型…")
    import numpy as np
    _ = await asyncio.to_thread(transcribe_audio, np.zeros(16000, dtype=np.float32), 16000)
    print("[Warmup] STT 预热完成")

    print("[Warmup] 预热 LLM (Ollama)…")
    try:
        w = LLMWorker("你是一个助手。")
        _ = await asyncio.to_thread(w.chat, "你好")
        print("[Warmup] LLM 预热完成")
    except Exception as e:
        print(f"[Warmup] LLM 预热失败 (可忽略): {e}")


async def main():
    print(f"可用角色: {[r['name'] for r in get_role_list()]}")
    print("[Warmup] 开始预热模型，首次启动可能需要 1-2 分钟…")
    try:
        await asyncio.wait_for(warmup(), timeout=120)
    except asyncio.TimeoutError:
        print("[Warmup] 预热超时，继续启动服务")
    print("[Warmup] 预热结束，启动 WebSocket 服务")

    async with websockets.serve(handle_ws, HOST, PORT):
        print(f"语音聊天室 WebSocket 服务器启动于 ws://{HOST}:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Server] 收到退出信号，服务器关闭")