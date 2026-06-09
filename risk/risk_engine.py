from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config import cfg
from db.database import get_open_positions, get_realized_pnl_total, insert_pnl_snapshot
from signals.signal_engine import Opportunity, StrategyType, VolRegime
from utils.logger import logger

# 40% stress buffer - approximates Derive's ±25% spot / ±50% vol scenario.
# TODO: replace with /private/get_margin?simulated_positions=[...] for live
_MARGIN_STRESS = 1.40

# credit strategies that must be closed as a group, never leg-by-leg
_CREDIT_STYPES = {"iron_condor", "short_put_spread", "short_call_spread"}

@dataclass
class PortfolioState:
    net_delta:      float = 0.0
    net_gamma:      float = 0.0
    net_theta:      float = 0.0
    net_vega:       float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl:   float = 0.0
    capital_used:   float = 0.0
    margin_available: float = 0.0
    num_positions:  int   = 0
    spot:           float = 0.0

@dataclass
class RiskDecision:
    approved:      bool
    reason:        str
    adjusted_size: float = 1.0
    notes:         str   = ""

class RiskEngine:

    _DD_BREACH_THRESHOLD = 3  # consecutive unrealized DD checks before halting

    def __init__(self):
        self._peak_capital    = cfg.total_capital_usd
        self._current_capital = cfg.total_capital_usd
        self._state           = PortfolioState()
        self._circuit_broken  = False
        self._circuit_reason  = ""
        self._dd_breach_count = 0

    async def update_portfolio_state(
        self,
        positions: Optional[List[Dict]] = None,
        spot: float = 0.0,
    ) -> PortfolioState:
        if positions is None:
            positions = await get_open_positions(mode=cfg.mode)

        nd = ng = nt = nv = upnl = capital = 0.0
        for pos in positions:
            sign   = 1.0 if pos["direction"] == "buy" else -1.0
            amount = float(pos.get("amount", 1.0))
            nd     += sign * amount * float(pos.get("delta", 0) or 0)
            ng     += sign * amount * float(pos.get("gamma", 0) or 0)
            nt     += sign * amount * float(pos.get("theta", 0) or 0)
            nv     += sign * amount * float(pos.get("vega",  0) or 0)
            entry  = float(pos.get("entry_price",   0) or 0)
            curr   = float(pos.get("current_price", 0) or 0)
            if curr > 0:
                upnl += sign * amount * (curr - entry)
            capital += entry * amount

        rpnl = await get_realized_pnl_total(mode=cfg.mode)
        self._current_capital = cfg.total_capital_usd + rpnl
        self._peak_capital    = max(self._peak_capital, self._current_capital)

        self._state = PortfolioState(
            net_delta=nd, net_gamma=ng, net_theta=nt, net_vega=nv,
            unrealized_pnl=upnl, realized_pnl=rpnl,
            capital_used=capital,
            margin_available=self._current_capital - capital,
            num_positions=len(positions),
            spot=spot,
        )
        await insert_pnl_snapshot({
            "ts": datetime.now(timezone.utc).isoformat(),
            "total_pnl": rpnl + upnl, "unrealized_pnl": upnl, "realized_pnl": rpnl,
            "net_delta": nd, "net_theta": nt, "net_vega": nv,
            "capital_used": capital, "mode": cfg.mode,
        })
        return self._state

    # circuit breakers
    def check_circuit_breakers(self, state: PortfolioState) -> Tuple[bool, str]:
        # realized DD: permanent capital loss, halt immediately -- no debate
        realized_capital = cfg.total_capital_usd + state.realized_pnl
        realized_dd      = (self._peak_capital - realized_capital) / self._peak_capital
        if realized_dd >= cfg.max_drawdown_pct:
            reason = f"REALIZED DRAWDOWN: {realized_dd:.1%} >= {cfg.max_drawdown_pct:.1%}"
            self._circuit_broken, self._circuit_reason = True, reason
            logger.critical(f"circuit breaker: {reason}")
            return True, reason

        # unrealized DD: could be a stale/wide quote -- require N consecutive breaches
        total_dd = (self._peak_capital - (realized_capital + state.unrealized_pnl)) / self._peak_capital
        if total_dd >= cfg.max_drawdown_pct:
            self._dd_breach_count += 1
            if self._dd_breach_count >= self._DD_BREACH_THRESHOLD:
                reason = f"TOTAL DRAWDOWN: {total_dd:.1%} ({self._dd_breach_count} consecutive checks)"
                self._circuit_broken, self._circuit_reason = True, reason
                logger.critical(f"circuit breaker: {reason}")
                return True, reason
            logger.warning(f"drawdown breach {self._dd_breach_count}/{self._DD_BREACH_THRESHOLD}: {total_dd:.1%}")
        else:
            self._dd_breach_count = 0

        if abs(state.net_delta) > cfg.max_portfolio_delta * 3:
            reason = f"EXCESSIVE DELTA: {state.net_delta:.3f}"
            logger.warning(f"circuit breaker warning: {reason}")
            return True, reason

        self._circuit_broken = False
        return False, ""

    def reset_circuit_breaker(self) -> None:
        self._circuit_broken, self._circuit_reason = False, ""
        logger.warning("circuit breaker manually reset")

    # pre-trade
    async def approve_trade(
        self, opp: Opportunity, state: Optional[PortfolioState] = None,
    ) -> RiskDecision:
        if self._circuit_broken:
            return RiskDecision(False, f"circuit breaker: {self._circuit_reason}")

        state = state or self._state
        margin_needed = self._margin_estimate(opp)
        available = state.margin_available - self._current_capital * cfg.margin_buffer_pct

        if margin_needed > available:
            return RiskDecision(False, f"capital: need ${margin_needed:.0f}, have ${available:.0f}")

        max_notional  = cfg.total_capital_usd * cfg.max_position_pct
        adjusted_size = min(1.0, max_notional / margin_needed) if margin_needed > 0 else 1.0
        notes         = ""

        new_delta = state.net_delta + opp.net_delta * adjusted_size
        if abs(new_delta) > cfg.max_portfolio_delta * 1.5:
            notes = f"delta will be {new_delta:.3f} after fill - hedge immediately"

        if opp.regime == VolRegime.SPIKE:
            return RiskDecision(False, "spike regime - no new credit positions")
        if opp.regime == VolRegime.HIGH:
            adjusted_size *= 0.60
            notes += " | size -40% (HIGH regime)"

        if opp.score < 30:
            return RiskDecision(False, f"score {opp.score:.1f} < 30")

        bad_legs = [l for l in opp.legs if l.spread_pct > 0.25]
        if bad_legs:
            return RiskDecision(False, f"spread too wide: {bad_legs[0].instrument_name} ({bad_legs[0].spread_pct:.1%})")

        logger.info(f"trade approved: {opp.strategy_type.value} score={opp.score:.1f} size={adjusted_size:.2f}x")
        return RiskDecision(True, "ok", adjusted_size=round(adjusted_size, 2), notes=notes)

    def _margin_estimate(self, opp: Opportunity) -> float:
        if opp.strategy_type == StrategyType.IRON_CONDOR:
            return opp.max_loss * _MARGIN_STRESS
        if opp.strategy_type == StrategyType.CALENDAR_SPREAD:
            return abs(opp.net_credit) * _MARGIN_STRESS
        return abs(opp.net_credit) * 2.0

    # delta hedge
    def delta_hedge_required(self, state: Optional[PortfolioState] = None) -> Optional[Dict]:
        state = state or self._state
        nd    = state.net_delta
        if abs(nd) <= cfg.max_portfolio_delta:
            return None
        direction = "sell" if nd > 0 else "buy"
        amount    = round(abs(nd), 4)
        logger.info(f"delta hedge: {direction} {amount} {cfg.underlying}-PERP (net_delta={nd:.4f})")
        return {
            "instrument_name": f"{cfg.underlying}-PERP",
            "direction": direction,
            "amount": amount,
            "reason": f"delta hedge net_delta={nd:.4f}",
        }

    # close signals
    def check_close_signals(
        self, positions: List[Dict], current_prices: Dict[str, float],
    ) -> List[Dict]:
        """
        Atomic close per strategy group - never close one IC leg without the others.
        group_key is read from raw_json (persisted at open time) to avoid collision
        when two ICs are opened in the same minute.
        """
        to_close  = []
        groups: Dict[str, List[Dict]] = defaultdict(list)
        standalone: List[Dict] = []

        for pos in positions:
            stype = pos.get("strategy_type", "")
            if stype in _CREDIT_STYPES:
                key = self._read_group_key(pos)
                groups[key].append(pos)
            else:
                standalone.append(pos)

        for key, legs in groups.items():
            if any((pos.get("dte") or 99) <= 1 for pos in legs):
                to_close.extend({**p, "close_reason": "DTE≤1 gamma risk", "urgency": "urgent"} for p in legs)
                continue
            to_close.extend(self._eval_group(key, legs, current_prices))

        for pos in standalone:
            result = self._eval_standalone(pos, current_prices)
            if result:
                to_close.append(result)

        return to_close

    @staticmethod
    def _read_group_key(pos: Dict) -> str:
        """
        Read group_key from raw_json. Falls back to opened_at[:16] if not found -
        covers positions opened before this field was added.
        """
        raw = pos.get("raw_json", "")
        if raw:
            try:
                data = ast.literal_eval(raw)
                key  = data.get("group_key", "")
                if key:
                    return key
            except Exception:
                pass
        # legacy fallback
        stype  = pos.get("strategy_type", "unknown")
        opened = pos.get("opened_at", "")[:16]
        return f"{stype}|{opened}"

    def _eval_group(self, key: str, legs: List[Dict], prices: Dict[str, float]) -> List[Dict]:
        # max_profit comes from the opportunity, persisted in raw_json at open time.
        # Reconstructing it from fill prices is wrong - fills include slippage and
        # the long wings reduce max_profit but would be ignored in a sell-only sum.
        max_profit = self._group_max_profit(legs)

        upnl = 0.0
        for pos in legs:
            entry  = float(pos.get("entry_price", 0) or 0)
            curr   = prices.get(pos.get("instrument_name", ""), 0)
            amount = float(pos.get("amount", 1.0))
            if curr <= 0 or entry <= 0:
                return []
            if pos["direction"] == "sell":
                upnl += (entry - curr) * amount
            else:
                upnl += (curr - entry) * amount

        if max_profit > 0 and upnl >= max_profit * cfg.credit_close_pct:
            pct = upnl / max_profit
            return [{**p, "close_reason": f"profit target {pct:.1%} ({key})", "urgency": "normal"} for p in legs]
        return []

    @staticmethod
    def _group_max_profit(legs: List[Dict]) -> float:
        """
        Read max_profit from raw_json persisted at open time.
        Falls back to fill reconstruction if raw_json is missing or corrupt.
        """
        for pos in legs:
            raw = pos.get("raw_json", "")
            if raw:
                try:
                    data = ast.literal_eval(raw)
                    mp = float(data.get("max_profit", 0))
                    if mp > 0:
                        return mp
                except Exception:
                    pass
        # fallback: sum sell fills minus buy fills (imprecise but better than 0)
        sell_sum = sum(
            float(p.get("entry_price", 0)) * float(p.get("amount", 1))
            for p in legs if p.get("direction") == "sell"
        )
        buy_sum = sum(
            float(p.get("entry_price", 0)) * float(p.get("amount", 1))
            for p in legs if p.get("direction") == "buy"
        )
        return max(sell_sum - buy_sum, 0.0)

    def _eval_standalone(self, pos: Dict, prices: Dict[str, float]) -> Optional[Dict]:
        entry = float(pos.get("entry_price", 0) or 0)
        curr  = prices.get(pos.get("instrument_name", ""), 0)
        dte   = pos.get("dte", 99)
        if curr <= 0 or entry <= 0:
            return None
        if (dte or 99) <= 1:
            return {**pos, "close_reason": "DTE≤1 gamma risk", "urgency": "urgent"}
        if pos["direction"] == "buy":
            pnl_pct = (curr - entry) / entry
            if pnl_pct < -0.50:
                return {**pos, "close_reason": f"stop loss {pnl_pct:.1%}", "urgency": "urgent"}
        return None

    @property
    def portfolio(self) -> PortfolioState:
        return self._state

    @property
    def is_circuit_broken(self) -> bool:
        return self._circuit_broken
