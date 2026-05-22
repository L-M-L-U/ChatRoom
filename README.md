# Voice Chat Room — 语音克隆聊天室

基于 STT → LLM → TTS 管道的实时语音聊天应用。

## 安装依赖

```bash
pip install -r requirements.txt
```

额外安装：
- [Ollama](https://ollama.com/) 并拉取模型：`ollama pull deepseek-r1:7b`
- 如需 RVC 音色转换：`pip install rvc-inference`

## 运行

1. 确保 Ollama 服务已启动。
2. 双击 `start_server.bat` 或运行：
   ```bash
   python backend/main.py
   ```
3. 用浏览器打开 `frontend/index.html`。
4. 按住录音按钮说话，松开发送，等待 AI 语音回复。

## 角色配置

角色文件置于 `config/roles/` 目录，格式为 Markdown + YAML frontmatter。