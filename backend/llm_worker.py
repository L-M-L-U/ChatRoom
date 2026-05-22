import re
import sys
import traceback
from datetime import datetime, timezone

from ollama import Client

from memory_worker import MemoryWorker

_client = Client()


def _strip_thinking(text: str) -> str:
    """去除 deepseek-r1 的 思维链标签。"""
    text = re.sub(r"<.*?>", "", text, flags=re.DOTALL)
    return text.strip()


class LLMWorker:
    def __init__(self, system_prompt: str, memory: MemoryWorker = None):
        self._system_prompt = system_prompt
        self._memory = memory

    def chat(self, user_text: str) -> str:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        sys_msg = self._system_prompt

        if self._memory:
            context = self._memory.retrieve_relevant(user_text, top_k=3)
            if context:
                sys_msg = f"{sys_msg}\n\n相关历史对话：\n{context}"

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_text},
        ]
        print(f"[LLM] 发送给 Ollama 的完整消息:")
        print(f"  system: {sys_msg}")
        print(f"  user: {user_text}")

        try:
            print(f"[LLM] 调用 Ollama deepseek-r1:7b…")
            resp = _client.chat(
                model="deepseek-r1:7b",
                messages=messages,
                options={"max_tokens": 512, "temperature": 0.7},
            )
            reply = _strip_thinking(resp["message"]["content"])
            print(f"[LLM] Ollama 返回: \"{reply[:120]}\" ({len(reply)} 字)")

            if self._memory:
                self._memory.add_message("user", user_text, ts)
                self._memory.add_message("assistant", reply, ts)

            return reply

        except Exception as e:
            print(f"[LLM] Ollama 调用失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return "抱歉，我现在无法思考，请稍后再试。"