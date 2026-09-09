"""Loopback-only episode review UI; intentionally no recorder or robot controls."""

from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
from urllib.parse import parse_qs
from urllib.parse import urlparse

from PIL import Image

from .workflow import ContractError
from .workflow import read_frame


def make_server(catalog, port=8766):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, status, value, kind="application/json"):
            body = json.dumps(value, ensure_ascii=False).encode() if kind == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def allowed_host(self):
            return self.headers.get("Host") in (
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            )

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            if not self.allowed_host():
                return self.reply(403, {"error": "loopback host required"})
            try:
                url = urlparse(self.path)
                query = parse_qs(url.query)
                if url.path == "/":
                    html = (
                        Path(__file__).with_name("review.html").read_text(encoding="utf-8").replace("__CSRF__", token)
                    )
                    return self.reply(200, html.encode(), "text/html; charset=utf-8")
                if url.path == "/api/episodes":
                    return self.reply(
                        200,
                        {
                            "episodes": catalog.list(),
                            "profile_id": catalog.profile["profile_id"],
                            "synthetic": catalog.profile["status"] == "synthetic",
                        },
                    )
                if url.path in ("/api/frame", "/api/image"):
                    report = catalog.get(query["id"][0])
                    if not report["quality_pass"] or report["superseded"]:
                        raise ContractError("only current quality-pass episodes can be previewed")
                    frame = read_frame(
                        report["path"],
                        catalog.profile,
                        int(query.get("frame", ["0"])[0]),
                        verify_sha256=report["sha256"],
                    )
                    if url.path == "/api/frame":
                        return self.reply(
                            200,
                            {
                                "timestamp_ns": str(frame["timestamp_ns"]),
                                "state": frame["state"].tolist(),
                                "action": frame["action"].tolist(),
                                "slots": list(frame["images"]),
                            },
                        )
                    image = frame["images"][query["slot"][0]]
                    stream = io.BytesIO()
                    Image.fromarray(image).save(stream, format="PNG")
                    return self.reply(200, stream.getvalue(), "image/png")
                return self.reply(404, {"error": "not found"})
            except (ContractError, OSError, ValueError, KeyError, IndexError) as exc:
                return self.reply(400, {"error": str(exc)})

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            if not self.allowed_host() or self.headers.get("X-HV1-CSRF") != token:
                return self.reply(403, {"error": "CSRF/host check failed"})
            origin = self.headers.get("Origin")
            if origin and origin not in (
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            ):
                return self.reply(403, {"error": "cross-origin request denied"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ContractError("invalid request size")
                value = json.loads(self.rfile.read(size))
                if self.path == "/api/review":
                    return self.reply(
                        200, catalog.review(value["id"], value["outcome"], value["use"], value.get("reason", ""))
                    )
                if self.path == "/api/scan":
                    return self.reply(200, {"scanned": len(catalog.scan())})
                return self.reply(404, {"error": "not found"})
            except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
                return self.reply(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
