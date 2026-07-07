"""
utils/report.py -- Daily performance report.

Reads from DB, prints to terminal. No external dependencies beyond what
we already have. Run with: python main.py --report [--days N]
"""
import asyncio
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DB_PATH = Path(__file__).parent.parent / "db" / "derive_bot.db"

W = 56  # print width


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _q(sql: str, params: tuple = ()) -> List[Dict]:
    with _conn() as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]


def _q1(sql: str, params: tuple = ()):
    with _conn() as c:
        r = c.execute(sql, params).fetchone()
        return r[0] if r else None


# section helpers

def _header(title: str) -> None:
    print(f"\n{title}")
    print("-" * W)


def _row(label: str, value: str, indent: int = 2) -> None:
    pad = W - indent - len(label) - len(value)
    print(" " * indent + label + " " * max(1, pad) + value)


def _bar(value: float, max_val: float, width: int = 20, fill: str = "#") -> str:
    if max_val == 0:
        return ""
    n = int(round(abs(value) / max_val * width))
    return fill * min(n, width)


# report sections

def _section_summary(days: int, mode: str) -> Dict:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    realized = _q1(
        "SELECT COALESCE(SUM(realized_pnl),0) FROM positions WHERE status='closed' AND mode=? AND closed_at>=?",
        (mode, since)
    ) or 0.0
    n_closed  = _q1("SELECT COUNT(*) FROM positions WHERE status='closed' AND mode=? AND closed_at>=?", (mode, since)) or 0
    n_wins    = _q1("SELECT COUNT(*) FROM positions WHERE status='closed' AND mode=? AND closed_at>=? AND realized_pnl>0", (mode, since)) or 0
    n_open    = _q1("SELECT COUNT(*) FROM positions WHERE status='open' AND mode=?", (mode,)) or 0
    n_signals = _q1("SELECT COUNT(*) FROM signals WHERE triggered=1 AND ts>=?", (since,)) or 0

    win_rate = n_wins / n_closed if n_closed > 0 else 0.0

    _header(f"SUMMARY  ({days}d  mode={mode})")
    _row("Realized PnL",     f"${realized:+.2f}")
    _row("Open positions",   str(n_open))
    _row("Closed trades",    str(n_closed))
    _row("Win rate",         f"{win_rate:.1%}  ({n_wins}/{n_closed})")
    _row("VRP signals fired",str(n_signals))

    return {"realized": realized, "n_closed": n_closed, "win_rate": win_rate}


def _section_open_positions(mode: str) -> None:
    rows = _q("SELECT * FROM positions WHERE status='open' AND mode=? ORDER BY opened_at DESC", (mode,))
    if not rows:
        return

    _header("OPEN POSITIONS")
    for p in rows:
        entry   = float(p.get("entry_price", 0) or 0)
        curr    = float(p.get("current_price", 0) or 0)
        amount  = float(p.get("amount", 1))
        sign    = 1 if p["direction"] == "buy" else -1
        upnl    = sign * amount * (curr - entry) if curr > 0 else 0
        delta   = float(p.get("delta", 0) or 0)
        theta   = float(p.get("theta", 0) or 0)
        dte     = p.get("dte", "?")
        inst    = p["instrument_name"][-30:]
        _row(inst, f"DTE={dte}  d={delta:+.3f}  th={theta:+.4f}  uPnL=${upnl:+.2f}")


def _section_closed_trades(days: int, mode: str) -> None:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows  = _q("""
        SELECT instrument_name, direction, amount, entry_price, close_price,
               realized_pnl, opened_at, closed_at, strategy_type
        FROM positions
        WHERE status='closed' AND mode=? AND closed_at>=?
        ORDER BY closed_at DESC
        LIMIT 20
    """, (mode, since))

    if not rows:
        return

    _header(f"CLOSED TRADES  (last {days}d, max 20)")
    for p in rows:
        pnl    = float(p.get("realized_pnl", 0) or 0)
        sign   = "+" if pnl >= 0 else ""
        closed = (p.get("closed_at") or "")[:10]
        inst   = p["instrument_name"][-24:]
        bar    = _bar(pnl, 50, width=12, fill="+" if pnl >= 0 else "-")
        _row(f"{closed}  {inst}", f"${sign}{pnl:.2f}  {bar}")


def _section_pnl_curve(days: int, mode: str) -> None:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows  = _q("""
        SELECT DATE(ts) as day, SUM(realized_pnl) as daily_pnl
        FROM pnl_snapshots
        WHERE mode=? AND ts>=?
        GROUP BY DATE(ts)
        ORDER BY day
    """, (mode, since))

    if not rows:
        return

    _header("DAILY PnL")
    values = [float(r["daily_pnl"] or 0) for r in rows]
    max_abs = max((abs(v) for v in values), default=1)

    for r, v in zip(rows, values):
        sign = "+" if v >= 0 else ""
        bar  = _bar(v, max_abs, width=24, fill="+" if v >= 0 else "-")
        _row(r["day"], f"${sign}{v:.2f}  {bar}")

    # summary stats
    if len(values) > 1:
        arr    = np.array(values)
        sharpe = float((arr.mean() / arr.std()) * math.sqrt(365)) if arr.std() > 0 else 0
        print()
        _row("Sharpe (annualized)", f"{sharpe:.2f}")
        _row("Best day",  f"${max(values):+.2f}")
        _row("Worst day", f"${min(values):+.2f}")


def _section_vol_surface(days: int) -> None:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows  = _q("""
        SELECT DATE(ts) as day,
               AVG(spot_price)  as spot,
               AVG(iv_atm_7d)   as iv7,
               AVG(rv_24h)      as rv24,
               AVG(vrp_7d)      as vrp
        FROM vol_surface_snapshots
        WHERE ts>=?
        GROUP BY DATE(ts)
        ORDER BY day
    """, (since,))

    if not rows:
        return

    _header("VOL SURFACE  (daily avg)")
    print(f"  {'date':<12} {'spot':>8} {'IV7d':>7} {'RV24h':>7} {'VRP':>6}")
    print("  " + "-" * 42)
    for r in rows:
        spot = float(r["spot"] or 0)
        iv7  = float(r["iv7"]  or 0) * 100
        rv24 = float(r["rv24"] or 0) * 100
        vrp  = float(r["vrp"]  or 0)
        print(f"  {r['day']:<12} {spot:>8,.0f} {iv7:>6.1f}% {rv24:>6.1f}% {vrp:>+6.1f}")


def _section_signals(days: int) -> None:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows  = _q("""
        SELECT DATE(ts) as day,
               COUNT(*) as total,
               SUM(triggered) as fired
        FROM signals
        WHERE ts>=?
        GROUP BY DATE(ts)
        ORDER BY day DESC
        LIMIT 14
    """, (since,))

    if not rows:
        return

    _header("VRP SIGNALS")
    for r in rows:
        fired = int(r["fired"] or 0)
        total = int(r["total"] or 0)
        bar   = _bar(fired, max(total, 1), width=16, fill="*")
        _row(r["day"], f"{fired}/{total} triggered  {bar}")


# entry point

async def run_report(days: int = 7, mode: str = "paper") -> None:
    if not DB_PATH.exists():
        print("no database found -- run the bot first")
        return

    print("=" * W)
    print(f"  thetavore  //  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * W)

    _section_summary(days, mode)
    _section_open_positions(mode)
    _section_closed_trades(days, mode)
    _section_pnl_curve(days, mode)
    _section_vol_surface(days)
    _section_signals(days)

    print("\n" + "=" * W)


if __name__ == "__main__":
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    asyncio.run(run_report(days=days))
