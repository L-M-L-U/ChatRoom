"""
对话记忆模块 — 基于 chromadb 的短期+长期记忆

依赖：
    pip install chromadb sentence-transformers

集成到 llm_worker.py 的方式：
    from memory_worker import MemoryWorker

    # 在 LLMWorker.__init__ 中：
        self.memory = MemoryWorker()
        self.session_id = str(id(self))  # 或用固定用户 ID

    # 在 LLMWorker.chat(user_text) 中，构建消息前：
        context = self.memory.retrieve_relevant(user_text, top_k=3)
        if context:
            sys_msg = self._system_prompt + f"\n\n相关历史对话：\n{context}"
        else:
            sys_msg = self._system_prompt
        messages = [{"role": "system", "content": sys_msg}, ...]

    # 得到回复后保存：
        self.memory.add_message("user", user_text, timestamp)
        self.memory.add_message("assistant", reply_text, timestamp)
"""

import os
import uuid
from datetime import datetime

import chromadb
from chromadb.config import Settings


class MemoryWorker:
    def __init__(self, collection_name: str = "voice_chat_memory", persist_dir: str = None):
        if persist_dir is None:
            persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 延迟加载 embedding 模型
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu",
            )
        return self._embedder

    def _embed(self, text: str) -> list:
        return self._get_embedder().encode(text).tolist()

    def add_message(self, role: str, content: str, timestamp: str = None):
        if not content.strip():
            return

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        doc_id = str(uuid.uuid4())
        metadata = {
            "role": role,
            "timestamp": timestamp,
        }
        embedding = self._embed(content)

        self._collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata],
            embeddings=[embedding],
        )

    def retrieve_relevant(self, query: str, top_k: int = 3) -> str:
        if not query.strip():
            return ""

        try:
            query_emb = self._embed(query)
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=top_k,
            )

            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results["metadatas"] else []

            if not docs:
                return ""

            lines = []
            for doc, meta in zip(docs, metas):
                role = meta.get("role", "unknown")
                ts = meta.get("timestamp", "")[11:19]  # 只取 HH:MM:SS
                lines.append(f"[{ts}] {role}: {doc}")

            return "\n".join(lines)

        except Exception as e:
            print(f"[MemoryWorker] 检索失败: {e}")
            return ""