import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import Optional

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from core.derive_client import DeriveRESTClient
from data.vol_surface import VolSurface
from db.database import get_open_positions, get_realized_pnl_total, init_db
from signals.signal_engine import SignalEngine, VolRegime

console = Console()

_REGIME_COLORS = {
    VolRegime.LOW:    "green",
    VolRegime.MEDIUM: "yellow",
    VolRegime.HIGH:   "red",
    VolRegime.SPIKE:  "bold red",
}


def _regime_color(regime: Optional[VolRegime]) -> str:
    return _REGIME_COLORS.get(regime, "white") if regime else "white"


def _make_header() -> Text:
    now        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    mode_color = "yellow" if cfg.is_paper else "red"
    t = Text()
    t.append("  options vrp harvester", style="bold white")
    t.append(f"  [{cfg.mode.upper()}]", style=f"bold {mode_color}")
    t.append(f"  {now}", style="dim white")
    return t


def _make_surface_table(surface: VolSurface, vrp: Optional[dict], skew: Optional[dict]) -> Table:
    table = Table(
        title=f"Vol Surface  {cfg.underlying}",
        show_header=True, header_style="bold cyan",
        border_style="cyan", min_width=60,
    )
    table.add_column("Metric",  style="white", width=22)
    table.add_column("Value",   justify="right", width=16)
    table.add_column("",        justify="center", width=14)

    if vrp:
        ratio      = vrp.get("vrp_ratio", 0)
        vrp_val    = vrp["vrp_7d"]
        triggered  = ratio >= cfg.vrp_threshold / 75.0 and vrp_val >= 1.5
        vc         = "green" if triggered else "yellow"
        vs         = "SIGNAL" if triggered else "wait"

        table.add_row("Spot",         f"${surface.spot:,.2f}", "")
        table.add_row("ATM IV 7D",    f"{vrp['iv_7d']:.1f}%",        "")
        table.add_row("ATM IV 14D",   f"{vrp['iv_14d']:.1f}%",       "")
        table.add_row("RV composite", f"{vrp['rv_composite']:.1f}%",  "")
        table.add_row(
            "VRP",
            f"[{vc}]{vrp_val:+.1f}pts ({ratio:.1%})[/{vc}]",
            f"[{vc}]{vs}[/{vc}]",
        )
    else:
        table.add_row("Vol Data", "[yellow]loading...[/yellow]", "")

    if skew:
        rr  = skew["risk_reversal"]
        rc  = "red" if rr < -2 else "green" if rr > 2 else "white"
        lbl = "put skew" if rr < 0 else "call skew" if rr > 0 else "flat"
        table.add_row("RR 30D",      f"[{rc}]{rr:+.1f}%[/{rc}]", lbl)
        table.add_row("25D put IV",  f"{skew['put_25d_iv']:.1f}%",  "")
        table.add_row("25D call IV", f"{skew['call_25d_iv']:.1f}%", "")
        table.add_row("Cheap side",  skew.get("cheap_side", "?"),    "")

    return table


def _make_portfolio_panel(
    positions: list,
    realized:  float,
    spot:      float,
    regime:    Optional[VolRegime],
) -> Panel:
    rc   = _regime_color(regime)
    name = regime.value.upper() if regime else "UNKNOWN"

    # compute net greeks and unrealized from positions
    nd = nt = nv = upnl = 0.0
    for p in positions:
        sign   = 1.0 if p["direction"] == "buy" else -1.0
        amount = float(p.get("amount", 1))
        nd    += sign * amount * float(p.get("delta", 0) or 0)
        nt    += sign * amount * float(p.get("theta", 0) or 0)
        nv    += sign * amount * float(p.get("vega",  0) or 0)
        entry  = float(p.get("entry_price",   0) or 0)
        curr   = float(p.get("current_price", 0) or 0)
        if curr > 0 and entry > 0:
            upnl += sign * amount * (curr - entry)

    total = realized + upnl

    def _c(v: float, positive_good: bool = True) -> str:
        good = v >= 0 if positive_good else v <= 0
        return "green" if good else "red"

    lines = (
        f"[bold]Regime:[/bold]    [{rc}]{name}[/{rc}]\n"
        f"[bold]Spot:[/bold]      ${spot:,.2f}\n"
        f"[bold]Net delta:[/bold] [{'red' if abs(nd) > cfg.max_portfolio_delta else 'white'}]{nd:+.4f}[/]\n"
        f"[bold]Net theta:[/bold] [{_c(nt)}]{nt:+.4f}/day[/]\n"
        f"[bold]Net vega:[/bold]  {nv:+.4f}\n"
        f"[bold]Unrealized:[/bold][{_c(upnl)}] ${upnl:+.2f}[/]\n"
        f"[bold]Realized:[/bold]  [{_c(realized)}]${realized:+.2f}[/]\n"
        f"[bold]Total PnL:[/bold] [{_c(total)}]${total:+.2f}[/]\n"
    )
    return Panel(lines, title="[bold white]Portfolio[/bold white]", border_style="white", expand=True)


async def _make_positions_table() -> Table:
    table = Table(
        title="Open Positions",
        show_header=True, header_style="bold magenta",
        border_style="magenta",
    )
    table.add_column("Instrument",  style="white", min_width=28)
    table.add_column("Dir",  justify="center", width=5)
    table.add_column("Amt",  justify="right",  width=6)
    table.add_column("Entry $",    justify="right", width=10)
    table.add_column("Curr $",     justify="right", width=10)
    table.add_column("uPnL",       justify="right", width=9)
    table.add_column("Delta",      justify="right", width=7)
    table.add_column("Theta",      justify="right", width=8)
    table.add_column("DTE",        justify="center", width=5)

    positions = await get_open_positions(mode=cfg.mode)
    if not positions:
        table.add_row("[dim]no open positions[/dim]", *[""] * 8)
        return table

    for p in positions:
        direction = p.get("direction", "")
        dc        = "green" if direction == "buy" else "red"
        delta     = float(p.get("delta", 0) or 0)
        theta     = float(p.get("theta", 0) or 0)
        entry     = float(p.get("entry_price",   0) or 0)
        curr      = float(p.get("current_price", 0) or 0)
        amount    = float(p.get("amount", 1))
        sign      = 1.0 if direction == "buy" else -1.0
        upnl      = sign * amount * (curr - entry) if curr > 0 and entry > 0 else 0.0
        upnl_c    = "green" if upnl >= 0 else "red"

        table.add_row(
            p.get("instrument_name", "")[-28:],
            f"[{dc}]{direction[:4].upper()}[/{dc}]",
            f"{amount:.1f}",
            f"${entry:.4f}",
            f"${curr:.4f}" if curr > 0 else "-",
            f"[{upnl_c}]${upnl:+.2f}[/{upnl_c}]",
            f"{delta:+.3f}",
            f"{theta:+.4f}",
            str(p.get("dte", "?")),
        )

    return table


def _build_layout(
    surface_table: Table,
    portfolio_panel: Panel,
    positions_table: Table,
    n_positions: int,
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(Align(_make_header(), align="center"), size=3, name="header"),
        Layout(name="top"),
        Layout(positions_table, size=min(n_positions + 5, 20), name="positions"),
    )
    layout["top"].split_row(
        Layout(surface_table,   name="surface"),
        Layout(portfolio_panel, name="portfolio", ratio=1),
    )
    return layout


async def run_dashboard() -> None:
    init_db()
    rest    = DeriveRESTClient()
    surface = VolSurface(rest)
    engine  = SignalEngine(surface)

    console.print("[cyan]loading vol surface...[/cyan]")
    await surface.refresh()

    with Live(console=console, refresh_per_second=0.2, screen=True) as live:
        while True:
            try:
                # fetch once per cycle, reuse across components
                vrp       = surface.vrp_signal()
                skew      = surface.put_call_skew(30)
                regime    = engine.detect_regime(vrp)
                positions = await get_open_positions(mode=cfg.mode)
                realized  = await get_realized_pnl_total(mode=cfg.mode)
                spot      = surface.spot

                layout = _build_layout(
                    surface_table   = _make_surface_table(surface, vrp, skew),
                    portfolio_panel = _make_portfolio_panel(positions, realized, spot, regime),
                    positions_table = await _make_positions_table(),
                    n_positions     = len(positions),
                )
                live.update(layout)
                await asyncio.sleep(5)

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]error: {e}[/red]")
                await asyncio.sleep(5)

    await rest.close()


if __name__ == "__main__":
    asyncio.run(run_dashboard())
