"""本地假 OpenAI 兼容服务：给集成测试用，不联网、不花钱。

只实现本项目用到的两个端点：/v1/embeddings 与 /v1/chat/completions（含 stream=True 的 SSE）。
"""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 32
PIECES = ("据【1】所述，", "缓粘结预应力", "需控制张拉时机与摩阻损失。")


def _embed(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 for b in (digest * 8)[:DIM]]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 静音
        pass

    def _json(self, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):  # noqa: N802 - http.server 的约定
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        if self.path.endswith("/embeddings"):
            texts = body["input"]
            if isinstance(texts, str):
                texts = [texts]
            self._json(
                {
                    "object": "list",
                    "model": body.get("model", "mock"),
                    "data": [
                        {"object": "embedding", "index": i, "embedding": _embed(t)}
                        for i, t in enumerate(texts)
                    ],
                }
            )
            return

        if self.path.endswith("/chat/completions"):
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in PIECES:
                    chunk = {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": body.get("model", "mock"),
                        "choices": [
                            {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            self._json(
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", "mock"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "".join(PIECES)},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
            return

        self.send_error(404)


def serve() -> ThreadingHTTPServer:
    """返回已启动的服务，调用方用 server.shutdown() 收尾。"""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def base_url(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}/v1"
