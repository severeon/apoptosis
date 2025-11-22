# reporter_tui.py
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

# Prefer textual if present, otherwise fallback to rich live
try:
    from textual.app import App, ComposeResult
    from textual.containers import Vertical, Horizontal
    from textual.widgets import Header, Footer, Static
    USE_TEXTUAL = True
except Exception:
    USE_TEXTUAL = False

from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.text import Text
from rich.progress import BarColumn, Progress
from rich.columns import Columns

DB_PATH = "experiments.db"
REFRESH = 3  # seconds
TOP_K = 8

def open_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def fetch_top_runs(n=TOP_K):
    con = open_db()
    cur = con.cursor()
    cur.execute("SELECT id, run_name, status, final_loss, total_apoptosis_events, timestamp FROM experiments WHERE status='done' ORDER BY final_loss ASC LIMIT ?", (n,))
    rows = cur.fetchall()
    con.close()
    return rows

def fetch_recent_runs(n=8):
    con = open_db()
    cur = con.cursor()
    cur.execute("SELECT id, run_name, status, final_loss, total_apoptosis_events, timestamp FROM experiments ORDER BY id DESC LIMIT ?", (n,))
    rows = cur.fetchall()
    con.close()
    return rows

def fetch_latest_metrics_for_run(exp_id, limit=10):
    con = open_db()
    cur = con.cursor()
    cur.execute("SELECT step, layer, mean, p50, p95, hist_json FROM metrics WHERE experiment_id=? ORDER BY step DESC LIMIT ?", (exp_id, limit))
    rows = cur.fetchall()
    con.close()
    return rows

def spark_hist_from_json(hist_json, width=24):
    try:
        hist = json.loads(hist_json)
    except Exception:
        return "nohist"
    # normalize
    mx = max(hist) if any(hist) else 1
    bars = []
    for v in hist:
        frac = v / mx if mx else 0.0
        level = int(frac * (width - 1))
        bars.append("█" * max(1, level))
    return " ".join(bars[:8])  # clip for display

console = Console()

def render_dashboard():
    top = fetch_top_runs()
    recent = fetch_recent_runs()

    t = Table(title=f"Top {TOP_K} Runs (done)", box=box.MINIMAL_DOUBLE_HEAD)
    t.add_column("id", style="dim", width=6)
    t.add_column("run_name", width=24)
    t.add_column("loss", justify="right")
    t.add_column("apoptosis", justify="right")
    t.add_column("ts", width=20)

    for r in top:
        t.add_row(str(r["id"]), r["run_name"][:24], f"{r['final_loss']:.4f}" if r["final_loss"] is not None else "-", str(r["total_apoptosis_events"] or 0), r["timestamp"][:19])

    t2 = Table(title="Recent Runs", box=box.SIMPLE)
    t2.add_column("id", width=6)
    t2.add_column("name", width=24)
    t2.add_column("status", width=10)
    t2.add_column("loss", justify="right")
    t2.add_column("apoptosis", justify="right")
    t2.add_column("ts", width=20)
    for r in recent:
        t2.add_row(str(r["id"]), r["run_name"][:24], r["status"], f"{r['final_loss']:.4f}" if r["final_loss"] is not None else "-", str(r["total_apoptosis_events"] or 0), r["timestamp"][:19])

    # Show a small preview of latest metrics for the top run
    metrics_panel = None
    if top:
        top_id = top[0]["id"]
        metrics = fetch_latest_metrics_for_run(top_id, limit=8)
        panels = []
        for m in metrics:
            hist_spark = spark_hist_from_json(m["hist_json"] or "[]")
            p = Panel.fit(f"step {m['step']} | layer {m['layer']}\n p50={m.get('p50'):.3f} p95={m.get('p95'):.3f}\n{hist_spark}", title=f"step {m['step']}", width=36)
            panels.append(p)
        metrics_panel = Columns(panels)

    layout = []
    layout.append(t)
    layout.append(t2)
    if metrics_panel:
        layout.append(Panel(metrics_panel, title="Latest metrics (top run)"))

    return layout

def rich_loop():
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            layout = render_dashboard()
            live.update(Columns(layout))
            time.sleep(REFRESH)

# Textual app fallback - simple
if USE_TEXTUAL:
    class SimpleWidget(Static):
        def __init__(self, refresh=REFRESH):
            super().__init__()
            self.refresh = refresh

        def on_mount(self):
            self.set_interval(self.refresh, self.refresh_view)

        async def refresh_view(self):
            content = ""
            top = fetch_top_runs()
            content += "[b]Top runs[/b]\n"
            for r in top:
                content += f"{r['id']}: {r['run_name']} loss={r['final_loss']}\n"
            self.update(content)

    class DashboardApp(App):
        CSS = ""
        def compose(self) -> ComposeResult:
            yield Header()
            yield SimpleWidget()
            yield Footer()

    def textual_main():
        app = DashboardApp()
        app.run()

if __name__ == "__main__":
    if USE_TEXTUAL:
        textual_main()
    else:
        rich_loop()
