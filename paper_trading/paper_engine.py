from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import cfg
from data.vol_surface import VolSurface, parse_instrument
from db.database import (
    insert_position, update_position_close,
    get_open_positions, get_position_by_id,
    update_position_greeks as db_update_greeks,
)
from risk.risk_engine import RiskDecision
from signals.signal_engine import Leg, Opportunity
from utils.logger import logger

_MAX_CONTRACTS = 10
_MARKET_IMPACT = 0.001  # taker slippage on top of bid/ask - slightly pessimistic on purpose

class PaperFill:
    __slots__ = ("instrument_name", "direction", "amount", "fill_price", "slippage", "fill_id", "ts")

    def __init__(self, instrument_name: str, direction: str, amount: float, fill_price: float, slippage: float):
        self.instrument_name = instrument_name
        self.direction  = direction
        self.amount     = amount
        self.fill_price = fill_price
        self.slippage   = slippage
        self.fill_id    = str(uuid.uuid4())[:8]
        self.ts         = datetime.now(timezone.utc)

    @property
    def cost(self) -> float:
        sign = 1.0 if self.direction == "buy" else -1.0
        return sign * self.fill_price * self.amount

class PaperTradingEngine:
    """
    Simulated execution that mirrors the live engine interface.
    Fill model: always taker (ask for buys, bid for sells) + small market impact.
    Good enough for realistic paper trading; don't expect it to be tighter than live.
    """

    def __init__(self, surface: VolSurface):
        self._surface = surface
        self._fills:  List[PaperFill] = []
        self._positions: Dict[str, Dict] = {}  # position_id → pos dict

    # fills
    def _fill(self, leg: Leg, amount: float) -> PaperFill:
        if leg.direction == "buy":
            base  = leg.ask if leg.ask > 0 else leg.mid * 1.005
            price = base * (1 + _MARKET_IMPACT)
            slip  = (price - leg.mid) / leg.mid if leg.mid > 0 else 0
        else:
            base  = leg.bid if leg.bid > 0 else leg.mid * 0.995
            price = base * (1 - _MARKET_IMPACT)
            slip  = (leg.mid - price) / leg.mid if leg.mid > 0 else 0
        return PaperFill(leg.instrument_name, leg.direction, amount, max(price, 1e-4), slip)

    def _n_contracts(self, opp: Opportunity, decision: RiskDecision) -> float:
        allocated = cfg.total_capital_usd * cfg.max_position_pct * decision.adjusted_size
        if opp.max_loss <= 0:
            return 1.0
        # max_loss is USD (bid/ask prices are USD on Derive) - no spot multiply needed
        n = allocated / opp.max_loss
        return max(1.0, min(float(_MAX_CONTRACTS), math.floor(n)))

    # open / close
    async def open_position(self, opp: Opportunity, decision: RiskDecision) -> List[str]:
        n         = self._n_contracts(opp, decision)
        opened_at = datetime.now(timezone.utc).isoformat()
        # group_key ties all legs of this strategy instance together for atomic closes
        group_key = f"{opp.strategy_type.value}|{opened_at[:16]}"

        logger.info(f"PAPER open: {opp.strategy_type.value} x{n:.0f} | credit=${opp.net_credit * n:.2f}")

        pids = []
        for leg in opp.legs:
            fill = self._fill(leg, n)
            pid  = f"paper_{fill.fill_id}_{leg.instrument_name}"
            pos  = {
                "position_id":     pid,
                "strategy_type":   opp.strategy_type.value,
                "instrument_name": leg.instrument_name,
                "direction":       leg.direction,
                "amount":          n,
                "entry_price":     fill.fill_price,
                "current_price":   fill.fill_price,
                "delta": leg.delta, "gamma": leg.gamma,
                "theta": leg.theta, "vega":  leg.vega,
                "iv":   leg.iv, "dte": leg.dte,
                "opened_at": opened_at,
                "status": "open", "mode": "paper",
                # persist opportunity-level fields so risk engine can use them
                # without reconstructing from fills - see _eval_group
                "raw_json": str({
                    "score":      opp.score,
                    "vrp":        opp.vrp,
                    "net_credit": opp.net_credit * n,   # scaled to actual contracts
                    "max_profit": opp.max_profit * n,
                    "group_key":  group_key,
                    "fill":       fill.fill_price,
                    "slip":       fill.slippage,
                }),
            }
            await insert_position(pos)
            self._positions[pid] = pos
            self._fills.append(fill)
            pids.append(pid)
            logger.info(f"  {leg.direction.upper()} {leg.option_type} k={leg.strike:.0f} "
                        f"DTE={leg.dte} fill=${fill.fill_price:.4f} slip={fill.slippage:.2%}")
        return pids

    async def close_position(
        self, position_id: str, reason: str, surface: Optional[VolSurface] = None,
    ) -> Optional[Dict]:
        sfc = surface or self._surface
        pos = self._positions.get(position_id) or await self._load_from_db(position_id)
        if pos is None:
            logger.warning(f"position {position_id} not found")
            return None

        inst   = pos["instrument_name"]
        direct = pos["direction"]
        amount = float(pos["amount"])
        entry  = float(pos["entry_price"])

        close_px = self._market_price(inst, direct, sfc) or float(pos.get("current_price", entry) or entry)
        pnl = (entry - close_px) * amount if direct == "sell" else (close_px - entry) * amount

        if not await update_position_close(position_id, close_px, pnl):
            return None  # already closed - race condition guard did its job

        self._positions.pop(position_id, None)
        logger.info(f"PAPER close: {inst} {direct} x{amount:.0f} | "
                    f"entry=${entry:.4f} close=${close_px:.4f} | pnl=${pnl:.2f} | {reason}")
        return {"position_id": position_id, "instrument_name": inst,
                "close_price": close_px, "realized_pnl": pnl, "reason": reason}

    async def close_all_positions(self, reason: str = "manual") -> float:
        positions = await get_open_positions(mode="paper")
        results   = [await self.close_position(p["position_id"], reason) for p in positions]
        total     = sum(r["realized_pnl"] for r in results if r)
        logger.info(f"closed all positions: pnl=${total:.2f}")
        return total

    async def _load_from_db(self, position_id: str) -> Optional[Dict]:
        return await get_position_by_id(position_id)

    def _market_price(self, inst: str, direction: str, surface: Optional[VolSurface]) -> Optional[float]:
        if not surface:
            return None
        for strikes in surface.surface.values():
            for sides in strikes.values():
                for data in sides.values():
                    if data.get("instrument_name") == inst:
                        # closing: long → sell at bid, short → buy at ask
                        return data.get("bid" if direction == "buy" else "ask", data.get("mid"))
        return None

    # greeks refresh
    async def update_position_greeks(self, surface: VolSurface) -> None:
        """
        Refresh greeks + DTE for every open position.
        DTE is re-derived from the instrument name each time - the DB value
        must decay or the expiry-close trigger never fires.
        """
        positions = await get_open_positions(mode="paper")
        inst_map: Dict[str, List[str]] = {}
        for p in positions:
            inst_map.setdefault(p["instrument_name"], []).append(p["position_id"])

        for strikes in surface.surface.values():
            for sides in strikes.values():
                for data in sides.values():
                    inst = data.get("instrument_name")
                    if not inst or inst not in inst_map:
                        continue
                    mid = data.get("mid", 0)
                    if mid <= 0:
                        continue
                    parsed   = parse_instrument(inst)
                    live_dte = parsed["dte"] if parsed else data.get("dte", 0)

                    for pid in inst_map[inst]:
                        if pid in self._positions:
                            self._positions[pid].update({
                                "current_price": mid, "dte": live_dte,
                                "delta": data.get("delta", 0), "gamma": data.get("gamma", 0),
                                "theta": data.get("theta", 0), "vega":  data.get("vega",  0),
                            })
                        await db_update_greeks(pid, mid,
                            data.get("delta", 0), data.get("gamma", 0),
                            data.get("theta", 0), data.get("vega",  0), live_dte)

    def summary(self) -> Dict:
        return {
            "total_fills":     len(self._fills),
            "total_cost_paid": sum(f.cost for f in self._fills),
            "open_positions":  len(self._positions),
            "fills": [
                {"instrument": f.instrument_name, "direction": f.direction,
                 "amount": f.amount, "price": f.fill_price, "ts": f.ts.isoformat()}
                for f in self._fills[-10:]
            ],
        }
