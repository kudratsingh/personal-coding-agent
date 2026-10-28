"""Make an offline coding-agent training dashboard from existing local run records.

No imports from model training or cloud code, no network I/O.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
METRIC_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,95}\Z")
SENSITIVE = re.compile(r"password|secret|api.?key|access.?token|authorization|credential", re.I)
METRIC_PREFIXES = ("validation/",)
METRIC_FIELDS = {
    "loss", "learning_rate", "gradient_norm", "tokens",
    "step_seconds", "tokens_per_second", "z_loss",
}
MAX_RUNS = 40
MAX_METRICS = 120
MAX_SAMPLES = 2400
MAX_LOG_BYTES = 32 * 1024 * 1024


def _numeric(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _valid_metric(key: object) -> bool:
    return (
        isinstance(key, str)
        and METRIC_NAME.fullmatch(key) is not None
        and not SENSITIVE.search(key)
        and (key in METRIC_FIELDS or key.startswith(METRIC_PREFIXES))
    )


def _read_events(path: Path, series: dict[str, list[list[float]]]) -> None:
    """Stream existing rank-0 metrics and validation events without retaining private data."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_LOG_BYTES:
        return
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or row.get("event") not in ("metrics", "validation"):
                continue
            fields = row.get("fields")
            if not isinstance(fields, dict):
                continue
            step, values = fields.get("step"), fields.get("metrics")
            if type(step) is not int or step < 0 or not isinstance(values, dict):
                continue
            for key, value in values.items():
                if not _valid_metric(key) or not _numeric(value):
                    continue
                if key not in series and len(series) >= MAX_METRICS:
                    continue
                points = series.setdefault(key, [])
                points.append([float(step), float(value)])
                if len(points) > MAX_SAMPLES:
                    points[:] = points[::2] + ([points[-1]] if len(points) % 2 == 0 else [])


def load_runs(root: Path) -> list[dict[str, Any]]:
    """Read only regular files in immediate run folders, never follow per-run symlinks."""
    if not root.is_dir():
        return []
    collected: list[dict[str, Any]] = []
    for folder in sorted(root.iterdir(), key=lambda x: x.name, reverse=True):
        if len(collected) >= MAX_RUNS:
            break
        if not folder.is_dir() or folder.is_symlink() or not RUN_NAME.fullmatch(folder.name):
            continue
        record = folder / "record.json"
        if not record.is_file() or record.is_symlink():
            continue
        try:
            metadata = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        metric_series: dict[str, list[list[float]]] = {}
        # Rank zero is the source of aggregate training metrics; don't duplicate other ranks.
        rank_zero = folder / "events-rank0.jsonl"
        if rank_zero.is_file() and not rank_zero.is_symlink():
            _read_events(rank_zero, metric_series)
        else:
            _read_events(folder / "events.jsonl", metric_series)
        status = metadata.get("status")
        cost = metadata.get("actual_cost_usd")
        commit = metadata.get("git_commit")
        seed = metadata.get("seed")
        collected.append(
            {
                "id": folder.name,
                "status": status if status in ("running", "completed", "failed") else "unknown",
                "commit": (
                    commit[:8]
                    if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit)
                    else None
                ),
                "seed": seed if type(seed) is int and seed >= 0 else None,
                "cost_usd": float(cost) if _numeric(cost) and cost >= 0 else None,
                "series": metric_series,
            }
        )
    return collected


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Coding Agent · Research Observatory</title><style>
:root{color-scheme:dark;background:#0d1321;color:#e6eaf2;font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{max-width:1250px;margin:0 auto;padding:32px 28px 72px}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;flex-wrap:wrap;margin-bottom:26px}
h1{font-size:clamp(27px,3vw,40px);letter-spacing:-.04em;margin:6px 0}
h2{font-size:18px;margin:0 0 6px;letter-spacing:-.02em}
.eyebrow{color:#9fc8f6;text-transform:uppercase;letter-spacing:.15em;font-size:11px;font-weight:700}
p,small{color:#a7b4c7;line-height:1.45;margin:0}small{display:block;margin-top:8px}
.panel{background:#141e30;border:1px solid #2b3850;border-radius:16px;padding:22px;margin:18px 0}
.row{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
.tag{border:1px solid #3c4b61;border-radius:999px;padding:8px 12px;color:#c1cbdc;font-size:12px}
select{background:#1a2940;color:#e6eaf2;border:1px solid #516179;border-radius:9px;padding:10px 12px;max-width:100%;min-width:170px}
select:focus-visible,input:focus-visible{outline:2px solid #8ec6ff}
#chart{display:block;width:100%;height:auto;min-height:240px;margin-top:12px}
.chart-label{font-size:12px;fill:#a0b0c8}.gridline{stroke:#34435b;stroke-width:1}
#legend{display:flex;flex-wrap:wrap;gap:8px 18px}.legend{display:flex;align-items:center;gap:7px;font-size:12px;color:#c3cfe1}
.swatch{width:12px;height:12px;display:inline-block;border-radius:3px}
.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}
th,td{text-align:left;padding:12px;border-bottom:1px solid #2b3850}th{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:#9cabca}td{color:#e2e8f4}
#benchmarkBody tr:last-child td,#runBody tr:last-child td{border-bottom:0}
.empty{color:#a9b9cd;padding:30px 12px;text-align:center}.pos{color:#6bd9a4}.neg{color:#f1a3a3}
footer{color:#8495ae;font-size:12px;margin-top:22px}
</style></head><body>
<header><div><div class="eyebrow">Local-first · Language-model research</div><h1>Coding Agent Observatory</h1><p>Training dynamics, held-out validation, and run comparisons.</p></div><div class="tag">Offline · Read-only inputs</div></header>
<section class="panel"><div class="row"><div><h2>Training metric explorer</h2><p>Compare loss, gradient norm, throughput and held-out validation metrics across runs.</p></div><label for="metric">Metric <select id="metric"></select></label></div><svg id="chart" role="img" aria-label="Selected metric over training steps" viewBox="0 0 960 345"></svg><div id="legend"></div></section>

<section class="panel"><h2>Run inventory</h2><p>Recorded local run status and cost, not live RunPod billing.</p><div class="tablewrap"><table><thead><tr><th>Run</th><th>Status</th><th>Steps seen</th><th>Training loss</th><th>Recorded cost</th><th>Commit</th></tr></thead><tbody id="runBody"></tbody></table></div></section>
<footer>Only numerical training and validation metrics and minimal provenance are displayed. No prompts, source code, hidden tests, full configurations, training data, stack traces or credentials are included.</footer>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const runs=JSON.parse(document.getElementById("data").textContent),ns="http://www.w3.org/2000/svg";
const colors=["#60a5fa","#52d4b6","#f4c16e","#e88fb4","#a78bfa","#4ecdec","#f59673"];
const metric=document.getElementById("metric"),chart=document.getElementById("chart"),legend=document.getElementById("legend");
const enabled=new Set(runs.map(r=>r.id));
function node(tag,attrs,text){const x=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs||{}))x.setAttribute(k,String(v));if(text!==undefined)x.textContent=text;chart.appendChild(x);return x;}
function cell(tr,value,cls){const td=document.createElement("td");td.textContent=value;if(cls)td.className=cls;tr.appendChild(td);}
function fmt(n){return n===null||!Number.isFinite(n)?"—":Number(n).toLocaleString(undefined,{maximumFractionDigits:4});}
function latest(r,key){const p=r.series[key]||[];return p.length?p[p.length-1][1]:null;}
const keys=[...new Set(runs.flatMap(r=>Object.keys(r.series)))];
const favorites=["loss","validation/english/bits_per_byte","validation/python/bits_per_byte","gradient_norm","tokens_per_second","step_seconds","learning_rate"];
keys.sort((a,b)=>(favorites.indexOf(a)<0?99:favorites.indexOf(a))-(favorites.indexOf(b)<0?99:favorites.indexOf(b))||a.localeCompare(b));
for(const key of keys){const o=document.createElement("option");o.value=key;o.textContent=key;metric.appendChild(o);}
function renderChart(){chart.replaceChildren();legend.replaceChildren();const key=metric.value;
 const active=runs.map((r,i)=>({r,i,p:r.series[key]||[]})).filter(x=>enabled.has(x.r.id)&&x.p.length);
 if(!active.length){node("text",{x:480,y:165,"text-anchor":"middle",class:"chart-label"},key?"No enabled runs have this metric":"No numeric training metrics found");}
 else {let xmin=Number.POSITIVE_INFINITY,xmax=Number.NEGATIVE_INFINITY,ymin=Number.POSITIVE_INFINITY,ymax=Number.NEGATIVE_INFINITY;
 for(const {p} of active)for(const [x,y] of p){xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);}
 if(xmin===xmax)xmax=xmin+1;if(ymin===ymax){const pad=Math.abs(ymin)*.05||.1;ymin-=pad;ymax+=pad;}
 const left=80,right=934,top=23,bottom=300,px=x=>left+(x-xmin)*(right-left)/(xmax-xmin),py=y=>bottom-(y-ymin)*(bottom-top)/(ymax-ymin);
 for(let t=0;t<=4;t++){const v=ymin+t*(ymax-ymin)/4,y=py(v);node("line",{x1:left,y1:y,x2:right,y2:y,class:"gridline"});node("text",{x:left-9,y:y+4,"text-anchor":"end",class:"chart-label"},fmt(v));const x=xmin+t*(xmax-xmin)/4;node("text",{x:px(x),y:326,"text-anchor":"middle",class:"chart-label"},fmt(x));}
 node("text",{x:505,y:343,"text-anchor":"middle",class:"chart-label"},"Training step");
 for(const {i,p} of active)node("polyline",{points:p.map(([x,y])=>px(x).toFixed(2)+","+py(y).toFixed(2)).join(" "),fill:"none",stroke:colors[i%colors.length],"stroke-width":2.4,"stroke-linejoin":"round"});
 }
 runs.forEach((r,i)=>{const label=document.createElement("label");label.className="legend";const cb=document.createElement("input");cb.type="checkbox";cb.checked=enabled.has(r.id);cb.addEventListener("change",()=>{if(cb.checked)enabled.add(r.id);else enabled.delete(r.id);renderChart();});const swatch=document.createElement("span");swatch.className="swatch";swatch.style.background=colors[i%colors.length];const name=document.createElement("span");name.textContent=r.id;label.append(cb,swatch,name);legend.appendChild(label);});
}
metric.addEventListener("change",renderChart);renderChart();
for(const r of runs){const tr=document.createElement("tr");let maxStep=null;for(const series of Object.values(r.series))for(const point of series)maxStep=maxStep===null?point[0]:Math.max(maxStep,point[0]);cell(tr,r.id);cell(tr,r.status);cell(tr,fmt(maxStep));const latestLoss=latest(r,"loss");cell(tr,fmt(latestLoss));cell(tr,r.cost_usd===null?"—":"$"+fmt(r.cost_usd));cell(tr,r.commit||"—");document.getElementById("runBody").appendChild(tr);}
if(!runs.length){const tr=document.createElement("tr"),td=document.createElement("td");td.colSpan=6;td.className="empty";td.textContent="No run records found. Choose a local runs/ directory.";tr.appendChild(td);document.getElementById("runBody").appendChild(tr);}
</script></body></html>"""


def write_dashboard(runs_root: Path, output: Path) -> int:
    """Write only an offline report. Never overwrite a run record or training artifact."""
    root = runs_root.resolve()
    destination = output.resolve()
    if destination.is_relative_to(root) and (
        not destination.relative_to(root).parts
        or destination.relative_to(root).parts[0] != "observability"
    ):
        raise ValueError("output inside runs/ must be in runs/observability/")
    data = load_runs(root)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", dir=destination.parent, encoding="utf-8", delete=False
    ) as file:
        temp = Path(file.name)
        try:
            file.write(HTML.replace("__PAYLOAD__", payload))
            file.flush()
            os.fsync(file.fileno())
            os.chmod(temp, 0o600)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, destination)
    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, help="default: <runs-root>/observability/index.html")
    args = parser.parse_args()
    output = args.output or args.runs_root / "observability" / "index.html"
    try:
        count = write_dashboard(args.runs_root, output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Read {count} local runs; dashboard: {output}")


if __name__ == "__main__":
    main()
