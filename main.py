"""
Usage:
  python main.py              # paper trading loop
  python main.py --live       # live trading
  python main.py --scan       # single scan, no trading
  python main.py --status     # print portfolio state
  python main.py --backtest   # run backtester
  python main.py --dashboard  # terminal dashboard
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from db.database import init_db
from utils.logger import setup_logger

def _args():
    p = argparse.ArgumentParser(description="Derive premium recycling bot")
    p.add_argument("--live",      action="store_true")
    p.add_argument("--scan",      action="store_true")
    p.add_argument("--status",    action="store_true")
    p.add_argument("--backtest",  action="store_true")
    p.add_argument("--dashboard", action="store_true")
    p.add_argument("--days", type=int, default=30, help="backtest window (days)")
    return p.parse_args()

async def _scan_once():
    from core.derive_client import DeriveRESTClient
    from data.vol_surface import VolSurface
    from signals.signal_engine import SignalEngine

    init_db()
    rest    = DeriveRESTClient()
    surface = VolSurface(rest)
    engine  = SignalEngine(surface)

    print("loading surface...")
    await surface.refresh()

    vrp = surface.vrp_signal()
    if vrp:
        print(f"\nVRP snapshot:")
        print(f"  IV 7D:   {vrp['iv_7d']:.1f}%")
        print(f"  RV comp: {vrp['rv_composite']:.1f}%")
        print(f"  VRP:     {vrp['vrp_7d']:+.1f}pts  ratio={vrp['vrp_ratio']:.1%}")
        print(f"  trigger: {'YES' if vrp['vrp_7d'] >= cfg.vrp_threshold else 'NO'}")

    skew = surface.put_call_skew(30)
    if skew:
        print(f"\nSkew 30D:")
        print(f"  RR:     {skew['risk_reversal']:+.1f}%")
        print(f"  25D put IV:  {skew['put_25d_iv']:.1f}%")
        print(f"  25D call IV: {skew['call_25d_iv']:.1f}%")
        print(f"  cheap side:  {skew['cheap_side']}")

    vrp_data, condors, calendars = await engine.scan()
    if condors:
        c = condors[0]
        print(f"\nBest IC: score={c.score:.1f} credit=${c.net_credit:.2f} max_loss=${c.max_loss:.2f}")
        for leg in c.legs:
            print(f"  {leg.direction.upper()} {leg.option_type} k={leg.strike:.0f} "
                  f"DTE={leg.dte} IV={leg.iv*100:.1f}% Δ={leg.delta:.3f} mid=${leg.mid:.2f}")
    else:
        print("\nno IC candidates")

    if calendars:
        cal = calendars[0]
        print(f"\nBest calendar: score={cal.score:.1f} debit=${abs(cal.net_credit):.2f} - {cal.notes}")

    await rest.close()

async def _status():
    from db.database import get_open_positions, get_realized_pnl_total
    init_db()
    positions = await get_open_positions(mode=cfg.mode)
    realized  = await get_realized_pnl_total(mode=cfg.mode)
    print(f"mode={cfg.mode}  open={len(positions)}  realized_pnl=${realized:.2f}")
    for p in positions:
        print(f"  {p['direction'].upper()} {p['instrument_name']} "
              f"x{p['amount']} @ ${float(p['entry_price']):.4f}  DTE={p.get('dte','?')}")

def main():
    args = _args()
    setup_logger(cfg.log_level)

    if args.live:
        os.environ["MODE"] = "live"
        cfg.mode = "live"

    if args.dashboard:
        from utils.dashboard import run_dashboard
        asyncio.run(run_dashboard())
    elif args.scan:
        asyncio.run(_scan_once())
    elif args.status:
        asyncio.run(_status())
    elif args.backtest:
        from utils.backtester import Backtester
        async def _run():
            bt = Backtester(capital=cfg.total_capital_usd, days=args.days)
            (await bt.run()).print_summary()
        asyncio.run(_run())
    else:
        from core.bot import main as bot_main
        asyncio.run(bot_main())

if __name__ == "__main__":
    main()
