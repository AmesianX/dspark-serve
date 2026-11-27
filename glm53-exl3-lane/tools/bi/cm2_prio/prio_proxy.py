"""Logging proxy between cm2 and vLLM. MODE=pass | reject (400 when 'priority' in body).
Logs one JSON line per request: path, priority field (or absent), n_messages, stream, status."""
import http.client, http.server, json, os, sys, threading, time
UP_HOST, UP_PORT = "127.0.0.1", 8000
MODE = sys.argv[2] if len(sys.argv) > 2 else "pass"
LOG = open(sys.argv[3] if len(sys.argv) > 3 else "proxy.jsonl", "a", buffering=1)
KEY = os.environ["UPSTREAM_KEY"]
lock = threading.Lock()
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _rec(self, **kw):
        with lock: LOG.write(json.dumps({"t": round(time.time(), 3), **kw}, ensure_ascii=False) + "\n")
    def do_GET(self): self._fwd(None)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = self.rfile.read(n) if n else b""
        self._fwd(body)
    def _fwd(self, body):
        info = {"method": self.command, "path": self.path}
        if body:
            try:
                j = json.loads(body); info.update(priority=j.get("priority", "ABSENT"), n_msgs=len(j.get("messages", [])),
                                                 stream=j.get("stream"), model=j.get("model"))
                if MODE == "reject" and "priority" in j:
                    msg = json.dumps({"error": {"message": "priority scheduling is not enabled on this server", "type": "BadRequestError", "code": 400}}).encode()
                    self.send_response(400); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(msg))); self.end_headers(); self.wfile.write(msg)
                    self._rec(**info, status=400, injected=True); return
            except Exception as e:
                info["parse_error"] = str(e)
        c = http.client.HTTPConnection(UP_HOST, UP_PORT, timeout=3600)
        hdr = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "authorization", "content-length", "accept-encoding")}
        hdr["Authorization"] = "Bearer " + KEY
        if body: hdr["Content-Length"] = str(len(body))
        c.request(self.command, self.path, body=body, headers=hdr); r = c.getresponse()
        if r.status >= 400:
            eb = r.read(); self.send_response(r.status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(eb))); self.end_headers(); self.wfile.write(eb)
            self._rec(**info, status=r.status, err=eb[:400].decode(errors="replace")); return
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "content-length", "connection"): self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk: break
            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n"); self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        self._rec(**info, status=r.status)
http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
