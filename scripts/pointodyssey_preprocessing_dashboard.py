#!/usr/bin/env python3
"""Serve a live dashboard for PointOdyssey preprocessing progress.

The preprocessing process owns the progress JSON file and replaces it
atomically.  This server is deliberately read-only: every API request opens
that file again so the browser always sees the latest complete snapshot.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit


EXPECTED_SCHEMA_VERSION = 1
EXPECTED_FORMAT = "pointodyssey_preprocessing_progress"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PointOdyssey preprocessing</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #090d14; --panel: #111824; --panel2: #162131;
      --line: #26364b; --text: #ebf2fb; --muted: #91a2b8;
      --blue: #62a8ff; --green: #4fd6a0; --amber: #ffc766; --red: #ff747f;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, sans-serif; }
    main { width: min(1500px, 100%); margin: auto; padding: 22px; }
    header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 18px; }
    h1 { margin: 0; font-size: clamp(20px, 3vw, 30px); letter-spacing: -.025em; }
    h2 { margin: 0 0 12px; font-size: 15px; }
    .subtle, .label { color: var(--muted); }
    .status { display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--line); border-radius: 99px; padding: 7px 11px; background: var(--panel); }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--amber); box-shadow: 0 0 10px currentColor; }
    .status.completed .dot { background: var(--green); }
    .status.failed .dot, .status.stale .dot { background: var(--red); }
    .grid { display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); gap: 12px; }
    .panel { grid-column: span 12; min-width: 0; border: 1px solid var(--line); border-radius: 13px; background: var(--panel); padding: 16px; }
    .metrics { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }
    .metric { background: var(--panel2); border-radius: 9px; padding: 12px; min-width: 0; }
    .metric .value { margin-top: 2px; font-size: 22px; font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis; }
    .progress-list { display: grid; gap: 12px; }
    .progress-head { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 5px; }
    .bar { overflow: hidden; height: 8px; border-radius: 99px; background: #253142; }
    .fill { height: 100%; width: 0; border-radius: inherit; background: linear-gradient(90deg, var(--blue), var(--green)); transition: width .35s ease; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .details { display: grid; grid-template-columns: 155px minmax(0, 1fr); gap: 7px 12px; }
    .details > div:nth-child(even) { overflow-wrap: anywhere; }
    table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    th, td { padding: 8px 7px; border-bottom: 1px solid var(--line); text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    .timing-bar { display: inline-block; min-width: 2px; height: 6px; margin-left: 8px; border-radius: 9px; background: var(--blue); vertical-align: middle; }
    .stats-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .stats-card { min-width: 0; padding: 12px; border-radius: 9px; background: var(--panel2); }
    .stats-card h3 { margin: 0 0 6px; font-size: 14px; }
    .stats-card table td { padding: 6px 4px; overflow-wrap: anywhere; }
    .error { display: none; border-color: #642f37; background: #27151a; color: #ffd9dc; white-space: pre-wrap; }
    .error.visible { display: block; }
    #progress-panel { grid-column: span 7; }
    #current-panel { grid-column: span 5; }
    #timing-panel { grid-column: span 7; }
    #diagnostics-panel { grid-column: span 5; }
    @media (max-width: 1050px) {
      .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .stats-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      #progress-panel, #current-panel, #timing-panel, #diagnostics-panel { grid-column: span 12; }
    }
    @media (max-width: 600px) {
      main { padding: 14px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .stats-grid { grid-template-columns: minmax(0, 1fr); }
      .details { grid-template-columns: 110px minmax(0, 1fr); }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div><h1>PointOdyssey preprocessing</h1><div class="subtle" id="paths">Waiting for progress file…</div></div>
    <div class="status" id="status-pill"><span class="dot"></span><strong id="status">connecting</strong><span class="subtle" id="age"></span></div>
  </header>
  <div class="grid">
    <section class="panel error" id="error"></section>
    <section class="panel">
      <h2>Live throughput</h2>
      <div class="metrics" id="rates"></div>
    </section>
    <section class="panel" id="progress-panel">
      <h2>Overall progress</h2>
      <div class="progress-list" id="progress"></div>
    </section>
    <section class="panel" id="current-panel">
      <h2>Current work</h2>
      <div class="details" id="current"></div>
    </section>
    <section class="panel" id="timing-panel">
      <h2>Stage timing</h2>
      <table><thead><tr><th>Stage</th><th>Seconds</th><th>Share</th></tr></thead><tbody id="timings"></tbody></table>
    </section>
    <section class="panel" id="diagnostics-panel">
      <h2>Diagnostics and I/O</h2>
      <div class="details" id="diagnostics"></div>
    </section>
    <section class="panel">
      <h2>Detailed running statistics</h2>
      <div class="stats-grid" id="statistics"></div>
    </section>
  </div>
</main>
<script>
const rateFields = [
  ['sources_per_second','Sources/s'], ['scenes_per_second','Scenes/s'],
  ['frames_per_second','Frames/s'], ['camera_frames_per_second','Depth frames/s'],
  ['jpegs_per_second','JPEGs/s'], ['validated_jpegs_per_second','Validated JPEGs/s'],
  ['output_mib_per_second','Output MiB/s']
];
const progressFields = [
  ['sources','Sources'], ['scenes','Prepared scenes'], ['frames','Scene frames'],
  ['camera_frames','Depth camera frames'], ['jpegs','JPEG files'],
  ['validated_jpegs','Validated JPEG files'], ['output_bytes','JPEG output bytes']
];
const fmt = new Intl.NumberFormat(undefined, {maximumFractionDigits: 2});
const intFmt = new Intl.NumberFormat();
const esc = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const label = key => key.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
const valueOrDash = value => value === null || value === undefined ? '—' : esc(value);
function numeric(value, digits=2) { return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—'; }
function detailRow(name, value, cls='') { return `<div class="label">${esc(name)}</div><div class="${cls}">${valueOrDash(value)}</div>`; }
function flatten(object, prefix='') {
  if (!object || typeof object !== 'object' || Array.isArray(object)) return [[prefix, object]];
  return Object.entries(object).flatMap(([key,value]) => {
    const path=prefix ? `${prefix} · ${label(key)}` : label(key);
    return value && typeof value === 'object' && !Array.isArray(value) ? flatten(value,path) : [[path,value]];
  });
}
function render(data) {
  const now = Date.now();
  const updated = Date.parse(data.updated_at);
  const ageSeconds = Number.isFinite(updated) ? Math.max(0, (now-updated)/1000) : null;
  const stale = ageSeconds !== null && ageSeconds > 15 && !['completed','failed'].includes(data.status);
  const pill = document.getElementById('status-pill');
  pill.className = `status ${stale ? 'stale' : data.status}`;
  document.getElementById('status').textContent = stale ? `${data.status} · stale` : data.status;
  document.getElementById('age').textContent = ageSeconds === null ? '' : `updated ${numeric(ageSeconds,1)}s ago`;
  document.getElementById('paths').textContent = `${data.source_root} → ${data.output_root}`;

  document.getElementById('rates').innerHTML = rateFields.map(([key,name]) =>
    `<div class="metric"><div class="label">${name}</div><div class="value">${numeric(data.rates?.[key])}</div></div>`
  ).join('');
  document.getElementById('progress').innerHTML = progressFields.map(([key,name]) => {
    const item=data.progress?.[key] || {completed:0,total:null,percent:null};
    const completedValue=Number(item.completed);
    const completed = key === 'output_bytes' ? `${numeric(completedValue/1048576)} MiB` : intFmt.format(completedValue || 0);
    if (item.total === null || item.total === undefined || item.percent === null || item.percent === undefined) {
      return `<div><div class="progress-head"><span>${name}</span><span>${completed} / —</span></div></div>`;
    }
    const pct=Math.max(0,Math.min(100,Number(item.percent)));
    const total = key === 'output_bytes' ? `${numeric(Number(item.total)/1048576)} MiB` : intFmt.format(Number(item.total));
    return `<div><div class="progress-head"><span>${name}</span><span>${completed} / ${total} · ${numeric(pct,1)}%</span></div><div class="bar"><div class="fill" style="width:${pct}%"></div></div></div>`;
  }).join('');

  const active = data.active || {};
  const timing = data.timing || {current_stage_elapsed_seconds:null,stages_seconds:{}};
  document.getElementById('current').innerHTML =
    detailRow('Stage', active.stage) + detailRow('Phase', active.phase) +
    detailRow('Stage elapsed', `${numeric(timing.current_stage_elapsed_seconds)} s`) +
    detailRow('Layout', active.layout) + detailRow('Source', active.source_sequence, 'mono') +
    detailRow('Split', active.split) + detailRow('Scene', active.scene_id, 'mono') +
    detailRow('View', active.view) + detailRow('Workers', data.workers) +
    detailRow('Total elapsed', `${numeric(data.elapsed_seconds)} s`) + detailRow('Started', data.started_at, 'mono');

  const stages = Object.entries(timing.stages_seconds || {});
  const stageTotal = stages.reduce((sum, [,seconds]) => sum + Number(seconds), 0);
  document.getElementById('timings').innerHTML = stages.map(([stage,seconds]) => {
    const share = stageTotal > 0 ? 100 * Number(seconds) / stageTotal : 0;
    return `<tr><td>${esc(label(stage))}</td><td>${numeric(seconds,3)}</td><td>${numeric(share,1)}%<span class="timing-bar" style="width:${Math.max(2,share)}px"></span></td></tr>`;
  }).join('') || '<tr><td colspan="3" class="subtle">No completed stage timings yet</td></tr>';

  const p=data.progress || {}, d=data.diagnostics || {};
  const jpegProgress=p.jpegs || {completed:0,total:0};
  const validatedJpegProgress=p.validated_jpegs || {completed:0,total:0};
  const outputBytes=p.output_bytes || {completed:0,total:null};
  const cameraFrames=p.camera_frames || {completed:0,total:0};
  document.getElementById('diagnostics').innerHTML =
    detailRow('Semantic failures', intFmt.format(d.semantic_validation_failures || 0)) +
    detailRow('Invalid RGB frames', intFmt.format(d.invalid_rgb_frames || 0)) +
    detailRow('JPEG files', `${intFmt.format(jpegProgress.completed || 0)} / ${intFmt.format(jpegProgress.total || 0)}`) +
    detailRow('Validated JPEGs', `${intFmt.format(validatedJpegProgress.completed || 0)} / ${intFmt.format(validatedJpegProgress.total || 0)}`) +
    detailRow('Output bytes', `${numeric(Number(outputBytes.completed || 0)/1048576)} MiB`) +
    detailRow('Depth camera frames', `${intFmt.format(cameraFrames.completed || 0)} / ${intFmt.format(cameraFrames.total || 0)}`) +
    detailRow('Last snapshot', data.updated_at, 'mono');
  const statistics=data.statistics || {};
  const sectionOrder=['tracks','depth','visibility','rgb','io'];
  const ordered=[...sectionOrder.filter(key => key in statistics), ...Object.keys(statistics).filter(key => !sectionOrder.includes(key))];
  document.getElementById('statistics').innerHTML = ordered.map(section => {
    const rows=flatten(statistics[section]);
    return `<div class="stats-card"><h3>${esc(label(section))}</h3><table><tbody>${rows.map(([name,value]) =>
      `<tr><td class="label">${esc(name)}</td><td>${typeof value === 'number' ? fmt.format(value) : valueOrDash(value)}</td></tr>`
    ).join('')}</tbody></table></div>`;
  }).join('') || '<div class="subtle">Statistics will appear as conversion counters are populated.</div>';
  const error = document.getElementById('error');
  if (data.error) {
    error.className='panel error visible';
    error.textContent=`${data.error.type}: ${data.error.message}`;
  } else { error.className='panel error'; error.textContent=''; }
}
async function refresh() {
  try {
    const response=await fetch('/api/progress', {cache:'no-store'});
    const data=await response.json();
    if (!response.ok) throw new Error(data.error?.message || `HTTP ${response.status}`);
    render(data);
  } catch (err) {
    const box=document.getElementById('error'); box.className='panel error visible'; box.textContent=`Dashboard update failed: ${err.message}`;
    document.getElementById('status').textContent='waiting'; document.getElementById('status-pill').className='status stale';
  }
}
refresh(); setInterval(refresh, 1000);
</script>
</body>
</html>
"""


class ProgressFileError(RuntimeError):
    """The progress snapshot is unavailable or violates its root contract."""


def read_progress(path: Path) -> dict[str, Any]:
    """Read and minimally validate one atomically published progress snapshot."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProgressFileError(f"cannot read progress snapshot {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProgressFileError("progress snapshot root must be a JSON object")
    if payload.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ProgressFileError(
            f"unsupported progress schema_version {payload.get('schema_version')!r}; "
            f"expected {EXPECTED_SCHEMA_VERSION}"
        )
    if payload.get("format") != EXPECTED_FORMAT:
        raise ProgressFileError(
            f"unsupported progress format {payload.get('format')!r}; "
            f"expected {EXPECTED_FORMAT!r}"
        )
    return payload


def make_handler(progress_path: Path) -> type[BaseHTTPRequestHandler]:
    """Bind a progress path to an HTTP request-handler class."""

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "PointOdysseyProgress/1"

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            request_path = urlsplit(self.path).path
            if request_path == "/":
                self._send(HTTPStatus.OK, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if request_path == "/api/progress":
                try:
                    payload = read_progress(progress_path)
                except ProgressFileError as exc:
                    body = json.dumps(
                        {"error": {"type": type(exc).__name__, "message": str(exc)}},
                        separators=(",", ":"),
                    ).encode("utf-8")
                    self._send(HTTPStatus.SERVICE_UNAVAILABLE, body, "application/json")
                    return
                body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
                self._send(HTTPStatus.OK, body, "application/json")
                return
            self._send(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain; charset=utf-8")

        def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self.do_GET()

        def log_message(self, format: str, *args: Any) -> None:
            return

    return DashboardHandler


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return port


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-json", type=Path, required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=_port, default=DEFAULT_PORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    progress_path = args.progress_json.expanduser().resolve()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(progress_path))
    print(
        f"POINTODYSSEY_DASHBOARD http://{args.host}:{server.server_port} "
        f"progress={progress_path}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
