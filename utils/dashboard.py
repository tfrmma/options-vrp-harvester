"""
utils/dashboard.py - Rich terminal dashboard.

Displays a live-updating terminal UI with:
  - Vol surface summary (ATM IV, VRP, skew)
  - Current regime and signal state
  - Open positions with P&L
  - Portfolio Greeks
  - Recent signals log

Run standalone: python -m utils.dashboard
"""
import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.align import Align

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from core.derive_client import DeriveRESTClient
from data.vol_surface import VolSurface
from db.database import init_db, get_open_positions, get_realized_pnl_total
from signals.signal_engine import SignalEngine, VolRegime

console = Console()

def regime_color(regime: Optional[VolRegime]) -> str:
    if regime is None:
        return "white"
    return {
        VolRegime.LOW: "green",
        VolRegime.MEDIUM: "yellow",
        VolRegime.HIGH: "red",
        VolRegime.SPIKE: "bold red",
    }.get(regime, "white")

def make_surface_table(surface: VolSurface) -> Table:
    """Vol surface summary table."""
    table = Table(
        title="📊 Vol Surface - ETH",
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        min_width=60,
    )
    table.add_column("Metric", style="white", width=25)
    table.add_column("Value", justify="right", width=20)
    table.add_column("Status", justify="center", width=12)

    vrp = surface.vrp_signal()
    skew_30d = surface.put_call_skew(30)

    if vrp:
        iv7 = vrp["iv_7d"]
        iv14 = vrp["iv_14d"]
        rv = vrp["rv_composite"]
        vrp_val = vrp["vrp_7d"]

        vrp_color = "green" if vrp_val >= cfg.vrp_threshold else "yellow"
        vrp_status = "✅ SIGNAL" if vrp_val >= cfg.vrp_threshold else "⏳ WAIT"

        table.add_row("Spot Price", f"${surface.spot:,.2f}", "")
        table.add_row("ATM IV (7D)", f"{iv7:.1f}%", "")
        table.add_row("ATM IV (14D)", f"{iv14:.1f}%", "")
        table.add_row("RV Composite", f"{rv:.1f}%", "")
        table.add_row(
            "VRP (IV-RV)",
            f"[{vrp_color}]{vrp_val:+.1f}pts[/{vrp_color}]",
            f"[{vrp_color}]{vrp_status}[/{vrp_color}]"
        )
    else:
        table.add_row("Vol Data", "[yellow]Loading...[/yellow]", "")

    if skew_30d:
        rr = skew_30d["risk_reversal"]
        rr_color = "red" if rr < -2 else "green" if rr > 2 else "white"
        table.add_row(
            "Risk Reversal (30D)",
            f"[{rr_color}]{rr:+.1f}%[/{rr_color}]",
            "Put" if rr < 0 else "Call" if rr > 0 else "Flat"
        )
        table.add_row("25D Put IV", f"{skew_30d['put_25d_iv']:.1f}%", "")
        table.add_row("25D Call IV", f"{skew_30d['call_25d_iv']:.1f}%", "")

    return table

async def make_positions_table() -> Table:
    """Open positions table."""
    table = Table(
        title="📋 Open Positions",
        show_header=True,
        header_style="bold magenta",
        border_style="magenta",
    )
    table.add_column("Instrument", style="white", min_width=28)
    table.add_column("Dir", justify="center", width=5)
    table.add_column("Amt", justify="right", width=6)
    table.add_column("Entry $", justify="right", width=10)
    table.add_column("Curr $", justify="right", width=10)
    table.add_column("Δ", justify="right", width=7)
    table.add_column("Θ/day", justify="right", width=8)
    table.add_column("DTE", justify="center", width=5)

    positions = await get_open_positions(mode=cfg.mode)

    if not positions:
        table.add_row("[dim]No open positions[/dim]", "", "", "", "", "", "", "")
    else:
        for pos in positions:
            direction = pos.get("direction", "")
            dir_color = "green" if direction == "buy" else "red"
            delta = float(pos.get("delta", 0) or 0)
            theta = float(pos.get("theta", 0) or 0)
            dte = pos.get("dte", "?")

            table.add_row(
                pos.get("instrument_name", "")[-28:],
                f"[{dir_color}]{direction[:4].upper()}[/{dir_color}]",
                f"{float(pos.get('amount', 0)):.1f}",
                f"${float(pos.get('entry_price', 0)):.4f}",
                f"${float(pos.get('current_price', 0) or 0):.4f}",
                f"{delta:+.3f}",
                f"{theta:+.4f}",
                str(dte),
            )

    return table

def make_portfolio_panel(
    net_delta: float,
    net_theta: float,
    net_vega: float,
    unrealized: float,
    realized: float,
    spot: float,
    regime: Optional[VolRegime],
) -> Panel:
    """Portfolio summary panel."""
    rc = regime_color(regime)
    regime_name = regime.value.upper() if regime else "UNKNOWN"

    content = (
        f"[bold]Regime:[/bold]     [{rc}]{regime_name}[/{rc}]\n"
        f"[bold]Spot:[/bold]       ${spot:,.2f}\n"
        f"[bold]Net Δ:[/bold]      [{'red' if abs(net_delta) > 0.15 else 'white'}]{net_delta:+.4f}[/]\n"
        f"[bold]Net Θ:[/bold]      [{'green' if net_theta > 0 else 'red'}]{net_theta:+.4f}/day[/]\n"
        f"[bold]Net ν:[/bold]      {net_vega:+.4f}\n"
        f"[bold]Unrealized:[/bold] [{'green' if unrealized >= 0 else 'red'}]${unrealized:+.2f}[/]\n"
        f"[bold]Realized:[/bold]   [{'green' if realized >= 0 else 'red'}]${realized:+.2f}[/]\n"
        f"[bold]Total PnL:[/bold]  [{'green' if (unrealized+realized) >= 0 else 'red'}]"
        f"${unrealized+realized:+.2f}[/]\n"
    )

    return Panel(
        content,
        title="[bold white]📈 Portfolio[/bold white]",
        border_style="white",
        expand=True,
    )

def make_header() -> Text:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    mode_color = "yellow" if cfg.is_paper else "red"
    t = Text()
    t.append("  🤖 Derive Premium Recycling Bot", style="bold white")
    t.append(f"  [{cfg.mode.upper()}]", style=f"bold {mode_color}")
    t.append(f"  {now}", style="dim white")
    return t

async def run_dashboard():
    """Live updating dashboard."""
    init_db()
    rest = DeriveRESTClient()
    surface = VolSurface(rest)
    signal_engine = SignalEngine(surface)

    console.print("[cyan]Connecting to Derive and loading vol surface...[/cyan]")
    await surface.refresh()

    with Live(console=console, refresh_per_second=0.2, screen=True) as live:
        while True:
            try:
                regime = signal_engine.detect_regime()
                vrp = surface.vrp_signal()

                positions = await get_open_positions(mode=cfg.mode)
                net_delta = sum(
                    (1 if p["direction"] == "buy" else -1) * float(p.get("delta", 0) or 0)
                    for p in positions
                )
                net_theta = sum(
                    (1 if p["direction"] == "buy" else -1) * float(p.get("theta", 0) or 0)
                    for p in positions
                )
                net_vega = sum(
                    (1 if p["direction"] == "buy" else -1) * float(p.get("vega", 0) or 0)
                    for p in positions
                )
                realized = await get_realized_pnl_total(mode=cfg.mode)

                layout = Layout()
                layout.split_column(
                    Layout(Align(make_header(), align="center"), size=3),
                    Layout(name="main"),
                    Layout(size=1),
                )
                layout["main"].split_row(
                    Layout(make_surface_table(surface), name="surface"),
                    Layout(
                        make_portfolio_panel(
                            net_delta, net_theta, net_vega,
                            0.0, realized, surface.spot, regime
                        ),
                        name="portfolio",
                        ratio=1,
                    ),
                )

                positions_table = await make_positions_table()
                layout.split_column(
                    Layout(Align(make_header(), align="center"), size=3),
                    Layout(name="top"),
                    Layout(positions_table, size=min(len(positions) + 5, 18)),
                )
                layout["top"].split_row(
                    Layout(make_surface_table(surface)),
                    Layout(
                        make_portfolio_panel(
                            net_delta, net_theta, net_vega,
                            0.0, realized, surface.spot, regime
                        ),
                        ratio=1,
                    ),
                )

                live.update(layout)
                await asyncio.sleep(5)

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Dashboard error: {e}[/red]")
                await asyncio.sleep(5)

    await rest.close()

if __name__ == "__main__":
    asyncio.run(run_dashboard())
