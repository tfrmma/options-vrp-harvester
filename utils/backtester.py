from __future__ import annotations

import math
import sqlite3
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import cfg
from data.vol_surface import bs_price
from db.database import init_db
from utils.logger import logger

# OU vol params calibrated to ETH - feel free to argue about these
_OU_MEAN_IV   = 0.75
_OU_KAPPA     = 0.15
_OU_VOL_OF_VOL = 0.25
_SPREAD_PCT   = 0.08   # conservative OTM bid-ask

class BacktestResult:
    def __init__(self):
        self.trades:       List[Dict]  = []
        self.daily_pnl:    List[float] = []
        self.equity_curve: List[float] = []
        self.signals_fired  = 0
        self.trades_opened  = 0
        self.trades_closed  = 0

    def add_trade(self, t: Dict) -> None:
        self.trades.append(t)

    def add_day(self, pnl: float, equity: float) -> None:
        self.daily_pnl.append(pnl)
        self.equity_curve.append(equity)

    @property
    def total_pnl(self) -> float:
        return sum(t.get("realized_pnl", 0) for t in self.trades if t.get("status") == "closed")

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.get("status") == "closed"]
        return sum(1 for t in closed if t.get("realized_pnl", 0) > 0) / len(closed) if closed else 0.0

    @property
    def avg_win(self) -> float:
        wins = [t["realized_pnl"] for t in self.trades if t.get("status") == "closed" and t.get("realized_pnl", 0) > 0]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t["realized_pnl"] for t in self.trades if t.get("status") == "closed" and t.get("realized_pnl", 0) < 0]
        return float(np.mean(losses)) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t["realized_pnl"] for t in self.trades if t.get("status") == "closed" and t.get("realized_pnl", 0) > 0)
        gross_loss   = abs(sum(t["realized_pnl"] for t in self.trades if t.get("status") == "closed" and t.get("realized_pnl", 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def sharpe_ratio(self) -> float:
        if len(self.daily_pnl) < 2:
            return 0.0
        arr = np.array(self.daily_pnl)
        return float((arr.mean() / arr.std()) * math.sqrt(365)) if arr.std() > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        curve = np.array(self.equity_curve)
        return float(((curve - np.maximum.accumulate(curve)) / np.maximum.accumulate(curve)).min())

    @property
    def annualized_return(self) -> float:
        if not self.equity_curve or len(self.daily_pnl) < 2:
            return 0.0
        total_ret = self.equity_curve[-1] / self.equity_curve[0] - 1
        return float((1 + total_ret) ** (365 / len(self.daily_pnl)) - 1)

    def print_summary(self) -> None:
        w = 48
        print("\n" + "-" * w)
        print(f"  signals={self.signals_fired}  opened={self.trades_opened}  closed={self.trades_closed}")
        print(f"  win_rate={self.win_rate:.1%}  avg_win=${self.avg_win:.2f}  avg_loss=${self.avg_loss:.2f}")
        print(f"  profit_factor={self.profit_factor:.2f}x  total_pnl=${self.total_pnl:.2f}")
        print(f"  ann_return={self.annualized_return:.1%}  sharpe={self.sharpe_ratio:.2f}  maxdd={self.max_drawdown:.1%}")
        print("-" * w)

class Backtester:
    """
    Runs against DB snapshots if available, synthetic OU data otherwise.
    P&L model: theta (sqrt-T decay) + gamma (spot² term) + vega.
    Good enough to validate signal quality; not a substitute for real fills.
    TODO: add vol regime transitions to synthetic generator - OU is too smooth
    """

    def __init__(self, capital: float = 2000.0, days: int = 30):
        self.capital = capital
        self.days    = days
        self.result  = BacktestResult()

    async def run(self) -> BacktestResult:
        init_db()
        snaps = self._load_db_snapshots() or self._synthetic_snapshots(self.days)
        logger.info(f"backtest: {len(snaps)} snapshots, {self.days} days")
        self._simulate(snaps)
        return self.result

    def _load_db_snapshots(self) -> List[Dict]:
        db = Path(__file__).parent.parent / "db" / "derive_bot.db"
        if not db.exists():
            return []
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM vol_surface_snapshots ORDER BY ts ASC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _synthetic_snapshots(self, days: int) -> List[Dict]:
        np.random.seed(42)
        n   = days * 24
        dt  = 1 / 24 / 365
        iv  = _OU_MEAN_IV
        spot = 3000.0
        base = datetime.now(timezone.utc) - timedelta(days=days)
        snaps = []

        for i in range(n):
            # OU for log(iv)
            log_iv = math.log(iv)
            log_iv += _OU_KAPPA * (math.log(_OU_MEAN_IV) - log_iv) * dt + _OU_VOL_OF_VOL * np.random.normal(0, math.sqrt(dt))
            iv = np.clip(math.exp(log_iv), 0.20, 2.50)

            spot *= math.exp((-0.5 * iv**2) * dt + iv * np.random.normal(0, math.sqrt(dt)))
            spot  = max(spot, 100.0)

            rv     = np.clip(iv * (0.80 + np.random.normal(0, 0.08)), 0.10, 2.0)
            iv_7d  = iv * (1.0 + np.random.normal(0.03, 0.02))

            snaps.append({
                "ts":          (base + timedelta(hours=i)).isoformat(),
                "underlying":  cfg.underlying,
                "spot_price":  round(spot, 2),
                "rv_24h":      round(rv, 4),
                "iv_atm_7d":   round(iv_7d, 4),
                "vrp_7d":      round((iv_7d - rv) * 100, 2),
            })
        return snaps

    def _simulate(self, snaps: List[Dict]) -> None:
        equity       = self.capital
        open_trades: List[Dict] = []
        daily_pnl    = 0.0
        prev_day     = None

        for snap in snaps:
            ts   = datetime.fromisoformat(snap["ts"])
            day  = ts.date()

            if prev_day is not None and day != prev_day:
                self.result.add_day(daily_pnl, equity)
                daily_pnl = 0.0
            prev_day = day

            spot   = snap.get("spot_price", 3000)
            iv_now = snap.get("iv_atm_7d", 0.75)
            vrp_7d = snap.get("vrp_7d", 0)

            to_close = []
            for t in open_trades:
                pnl, closed = self._step_trade(t, spot, iv_now)
                t["unrealized_pnl"] = t.get("unrealized_pnl", 0) + pnl
                if closed:
                    to_close.append(t)

            for t in to_close:
                pnl = t["unrealized_pnl"]
                equity     += pnl
                daily_pnl  += pnl
                t.update({"realized_pnl": pnl, "status": "closed"})
                self.result.add_trade(t)
                self.result.trades_closed += 1
                open_trades.remove(t)

            # signal check: normalized VRP ratio matches live engine
            iv_pct = iv_now * 100
            vrp_ratio = vrp_7d / iv_pct if iv_pct > 0 else 0
            if vrp_ratio >= cfg.vrp_threshold / 75.0 and vrp_7d >= 1.5 and len(open_trades) < 3:
                t = self._new_trade(spot, iv_now, vrp_7d, ts)
                if t:
                    open_trades.append(t)
                    self.result.signals_fired  += 1
                    self.result.trades_opened  += 1

        # mark remaining open at end
        for t in open_trades:
            pnl = t["unrealized_pnl"]
            equity += pnl
            t.update({"realized_pnl": pnl, "status": "closed", "close_reason": "end_of_backtest"})
            self.result.add_trade(t)
            self.result.trades_closed += 1

        if daily_pnl:
            self.result.add_day(daily_pnl, equity)

    def _step_trade(self, t: Dict, spot: float, iv_now: float):
        """
        Full BS revaluation per step. No more Taylor approximation.
        The gamma approximation (0.5*g*dS^2) breaks down near strikes
        in the last 2 DTE -- exactly when it matters most for a short IC.
        """
        t["dte"] -= 1 / 24
        dte_now = max(t["dte"], 0.01)
        T_now   = dte_now / 365.0
        r       = 0.0

        # reprice all 4 legs at current spot/iv/T and sum the portfolio value
        portfolio_value = 0.0
        for leg in t["legs"]:
            price = bs_price(spot, leg["strike"], T_now, r, iv_now, leg["opt"])
            # short leg: we are short the option, so portfolio gains when price drops
            sign  = -1.0 if leg["direction"] == "sell" else 1.0
            portfolio_value += sign * price * t["size"]

        period_pnl = portfolio_value - t.get("last_portfolio_value", t["entry_portfolio_value"])
        t["last_portfolio_value"] = portfolio_value
        upnl       = t.get("unrealized_pnl", 0) + period_pnl
        max_profit = t["max_profit"]

        if upnl >= max_profit * cfg.credit_close_pct:
            t["close_reason"] = "profit_target"
            return period_pnl, True
        if dte_now <= 1 / 24:
            t["close_reason"] = "expiry"
            return period_pnl, True
        if upnl < -2.0 * max_profit:
            t["close_reason"] = "stop_loss"
            return period_pnl, True
        return period_pnl, False

    def _new_trade(self, spot: float, iv: float, vrp: float, ts: datetime) -> Optional[Dict]:
        T     = 10 / 365.0
        r     = 0.0
        sigma = iv

        # pick strikes at target deltas using BS
        short_put_strike  = self._strike_for_delta(spot, T, r, sigma, cfg.credit_delta_target, "P")
        long_put_strike   = self._strike_for_delta(spot, T, r, sigma, cfg.credit_delta_target * 0.45, "P")
        short_call_strike = self._strike_for_delta(spot, T, r, sigma, cfg.credit_delta_target, "C")
        long_call_strike  = self._strike_for_delta(spot, T, r, sigma, cfg.credit_delta_target * 0.45, "C")

        legs = [
            {"strike": short_put_strike,  "opt": "P", "direction": "sell"},
            {"strike": long_put_strike,   "opt": "P", "direction": "buy"},
            {"strike": short_call_strike, "opt": "C", "direction": "sell"},
            {"strike": long_call_strike,  "opt": "C", "direction": "buy"},
        ]

        # fill prices: sell at bid, buy at ask
        def fill(strike, opt, direction):
            mid = bs_price(spot, strike, T, r, sigma, opt)
            return mid * (1 - _SPREAD_PCT / 2) if direction == "sell" else mid * (1 + _SPREAD_PCT / 2)

        net_credit = sum(
            fill(l["strike"], l["opt"], l["direction"]) * (-1 if l["direction"] == "buy" else 1)
            for l in legs
        )
        if net_credit <= 0:
            return None

        put_spread_width  = short_put_strike  - long_put_strike
        call_spread_width = long_call_strike  - short_call_strike
        max_loss = max(put_spread_width, call_spread_width) - net_credit
        if max_loss <= 0:
            return None

        size = min(1.0, max(0.001, self.capital * cfg.max_position_pct / max_loss))

        # compute entry portfolio value for BS revaluation baseline
        entry_portfolio_value = sum(
            (-1 if l["direction"] == "sell" else 1) * bs_price(spot, l["strike"], T, r, sigma, l["opt"]) * size
            for l in legs
        )

        return {
            "id":                    f"bt_{self.result.trades_opened}",
            "dte":                   10.0,
            "spot_at_open":          spot,
            "iv_at_open":            iv,
            "vrp_at_open":           vrp,
            "legs":                  legs,
            "size":                  size,
            "net_credit":            net_credit * size,
            "max_profit":            net_credit * size,
            "max_loss":              max_loss   * size,
            "entry_portfolio_value": entry_portfolio_value,
            "last_portfolio_value":  entry_portfolio_value,
            "unrealized_pnl":        0.0,
            "status":                "open",
        }

    @staticmethod
    def _strike_for_delta(spot: float, T: float, r: float, sigma: float, target_delta: float, opt: str) -> float:
        """Invert BS delta to find strike. Newton-Raphson with fallback to search."""
        from scipy.optimize import brentq
        from data.vol_surface import bs_greeks
        try:
            if opt == "P":
                target = -abs(target_delta)
                f = lambda K: bs_greeks(spot, K, T, r, sigma, "P")["delta"] - target
                return brentq(f, spot * 0.50, spot * 0.999, xtol=0.01)
            else:
                target = abs(target_delta)
                f = lambda K: bs_greeks(spot, K, T, r, sigma, "C")["delta"] - target
                return brentq(f, spot * 1.001, spot * 1.50, xtol=0.01)
        except Exception:
            # fallback: rough approximation if brentq fails
            offset = sigma * math.sqrt(T) * spot
            return spot - offset if opt == "P" else spot + offset
