from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import cfg
from data.vol_surface import VolSurface
from db.database import insert_signal
from utils.logger import logger

class VolRegime(Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    SPIKE  = "spike"

class StrategyType(Enum):
    IRON_CONDOR       = "iron_condor"
    SHORT_PUT_SPREAD  = "short_put_spread"
    SHORT_CALL_SPREAD = "short_call_spread"
    CALENDAR_SPREAD   = "calendar_spread"
    LONG_STRANGLE     = "long_strangle"

@dataclass
class Leg:
    instrument_name: str
    direction:   str    # buy | sell
    option_type: str    # C | P
    strike:      float
    dte:         int
    iv:          float
    mid:         float
    bid:         float
    ask:         float
    delta:       float
    gamma:       float
    theta:       float
    vega:        float
    spread_pct:  float

@dataclass
class Opportunity:
    strategy_type: StrategyType
    legs:       List[Leg]
    net_credit: float   # positive = collected
    net_delta:  float
    net_theta:  float
    net_vega:   float
    max_profit: float
    max_loss:   float
    score:      float
    vrp:        float
    regime:     VolRegime
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes:       str = ""

    @property
    def risk_reward(self) -> float:
        return abs(self.max_profit / self.max_loss) if self.max_loss else 0.0

    @property
    def theta_per_dollar_risk(self) -> float:
        return abs(self.net_theta / self.max_loss) if self.max_loss else 0.0

# regime boundaries calibrated to ETH - adjust if you add BTC
_REGIME_THRESHOLDS = [(40, VolRegime.LOW), (70, VolRegime.MEDIUM), (120, VolRegime.HIGH)]

def _classify_regime(iv_7d: float) -> VolRegime:
    for threshold, regime in _REGIME_THRESHOLDS:
        if iv_7d < threshold:
            return regime
    return VolRegime.SPIKE

# VRP scorer: normalize to 75% base IV so the signal is regime-invariant
_VRP_BASE_IV = 75.0

def _vrp_triggered(vrp_ratio: float, vrp_pts: float, regime: VolRegime) -> bool:
    ratio_threshold = cfg.vrp_threshold / _VRP_BASE_IV
    abs_min = 1.5  # always need at least 1.5 vol pts regardless of ratio
    if regime == VolRegime.LOW:
        ratio_threshold *= 0.75  # relax a bit - low-vol VRP is structurally smaller
    return vrp_ratio >= ratio_threshold and vrp_pts >= abs_min

class SignalEngine:

    def __init__(self, surface: VolSurface):
        self.surface = surface
        self._last_regime = VolRegime.MEDIUM

    def detect_regime(self, vrp: Optional[Dict] = None) -> VolRegime:
        data = vrp or self.surface.vrp_signal()
        if data is None:
            return self._last_regime
        self._last_regime = _classify_regime(data["iv_7d"])
        return self._last_regime

    async def check_vrp_signal(self) -> Optional[Dict]:
        vrp = self.surface.vrp_signal()
        if vrp is None:
            return None

        regime    = self.detect_regime(vrp)   # reuse already-fetched vrp, no second call
        vrp_ratio = vrp.get("vrp_ratio", 0)
        triggered = _vrp_triggered(vrp_ratio, vrp["vrp_7d"], regime)

        await insert_signal({
            "ts":          datetime.now(timezone.utc).isoformat(),
            "underlying":  cfg.underlying,
            "signal_type": "vrp",
            "value":       vrp["vrp_7d"],
            "threshold":   cfg.vrp_threshold / _VRP_BASE_IV,
            "triggered":   int(triggered),
            "metadata":    str(vrp),
        })

        if triggered:
            logger.info(
                f"VRP signal: IV7d={vrp['iv_7d']:.1f}% RV={vrp['rv_composite']:.1f}% "
                f"VRP={vrp['vrp_7d']:.1f}pts ({vrp_ratio:.1%}) [{regime.value}]"
            )
        else:
            logger.debug(f"VRP below threshold: {vrp_ratio:.1%} / {vrp['vrp_7d']:.1f}pts")

        return {**vrp, "triggered": triggered, "regime": regime}

    # Iron Condor
    async def scan_iron_condors(self, vrp_data: Dict) -> List[Opportunity]:
        regime = vrp_data.get("regime", VolRegime.MEDIUM)
        if regime == VolRegime.SPIKE:
            logger.info("spike regime - no ICs")
            return []
        opps = [o for dte in [7, 10, 14]
                if (o := self._build_ic(dte, vrp_data, regime)) is not None]
        return sorted(opps, key=lambda x: x.score, reverse=True)

    def _build_ic(self, dte: int, vrp_data: Dict, regime: VolRegime) -> Optional[Opportunity]:
        delta_short = cfg.credit_delta_target   # ~0.18
        delta_wing  = delta_short * 0.45        # ~0.08

        sp_cands = self.surface.get_strikes_by_delta(dte, delta_short, "P")
        lp_cands = self.surface.get_strikes_by_delta(dte, delta_wing,  "P")
        sc_cands = self.surface.get_strikes_by_delta(dte, delta_short, "C")
        lc_cands = self.surface.get_strikes_by_delta(dte, delta_wing,  "C")

        if not all([sp_cands, lp_cands, sc_cands, lc_cands]):
            return None

        sp, lp, sc, lc = sp_cands[0], lp_cands[0], sc_cands[0], lc_cands[0]

        # all 4 legs must share an expiry - cross-expiry IC is a different beast
        sp, lp, sc, lc = self._align_expiry(sp, lp, sc, lc, sp_cands, lp_cands, sc_cands, lc_cands)
        if sp is None:
            return None

        if lp["strike"] >= sp["strike"] or sc["strike"] >= lc["strike"]:
            return None

        # conservative fill: sell at bid, buy at ask
        net_credit = sp["bid"] + sc["bid"] - lp["ask"] - lc["ask"]
        if net_credit <= 0:
            return None

        spread_width = max(sp["strike"] - lp["strike"], lc["strike"] - sc["strike"])
        max_loss = spread_width - net_credit
        if max_loss <= 0:
            return None

        legs = [
            self._leg(sp, "sell", "P"), self._leg(lp, "buy", "P"),
            self._leg(sc, "sell", "C"), self._leg(lc, "buy", "C"),
        ]

        return Opportunity(
            strategy_type=StrategyType.IRON_CONDOR,
            legs=legs,
            net_credit=net_credit,
            net_delta=sp["delta"] + sc["delta"] - lp["delta"] - lc["delta"],
            net_theta=-(sp["theta"] + sc["theta"]) + lp["theta"] + lc["theta"],
            net_vega=-(sp["vega"]  + sc["vega"])  + lp["vega"]  + lc["vega"],
            max_profit=net_credit,
            max_loss=max_loss,
            score=self._score_ic(net_credit, max_loss, legs, vrp_data["vrp_7d"], dte, regime),
            vrp=vrp_data["vrp_7d"],
            regime=regime,
            notes=f"DTE={dte} put_width={sp['strike']-lp['strike']:.0f} call_width={lc['strike']-sc['strike']:.0f}",
        )

    @staticmethod
    def _align_expiry(sp, lp, sc, lc, sp_cands, lp_cands, sc_cands, lc_cands):
        dtes = [sp["dte"], lp["dte"], sc["dte"], lc["dte"]]
        if len(set(dtes)) == 1:
            return sp, lp, sc, lc
        target = Counter(dtes).most_common(1)[0][0]

        def pick(cands, dt):
            m = [c for c in cands if c["dte"] == dt]
            return m[0] if m else None

        sp, lp, sc, lc = pick(sp_cands, target), pick(lp_cands, target), \
                          pick(sc_cands, target), pick(lc_cands, target)
        if not all([sp, lp, sc, lc]) or len({sp["dte"], lp["dte"], sc["dte"], lc["dte"]}) > 1:
            return None, None, None, None
        return sp, lp, sc, lc

    @staticmethod
    def _leg(data: Dict, direction: str, opt_type: str) -> Leg:
        return Leg(
            instrument_name=data["instrument_name"], direction=direction, option_type=opt_type,
            strike=data["strike"], dte=data["dte"], iv=data["iv"],
            mid=data["mid"], bid=data["bid"], ask=data["ask"],
            delta=data["delta"], gamma=data["gamma"], theta=data["theta"],
            vega=data["vega"], spread_pct=data["spread_pct"],
        )

    @staticmethod
    def _score_ic(net_credit, max_loss, legs, vrp_pts, dte, regime) -> float:
        iv_approx = {VolRegime.LOW: 35.0, VolRegime.MEDIUM: 70.0,
                     VolRegime.HIGH: 100.0, VolRegime.SPIKE: 150.0}.get(regime, 70.0)

        vrp_score = min(30.0, (vrp_pts / iv_approx) * 150.0)
        rr_score  = min(25.0, (net_credit / max_loss) * 100) if max_loss > 0 else 0

        # net theta for the spread: short legs contribute positive theta, long legs negative
        net_theta = sum(
            (-l.theta if l.direction == "sell" else l.theta)
            for l in legs
        )
        theta_eff   = abs(net_theta) / max_loss if max_loss > 0 else 0
        theta_score = min(20.0, theta_eff * 5000)

        liq_score = max(0.0, 15.0 * (1 - np.mean([l.spread_pct for l in legs]) * 3))
        dte_score = max(0.0, 10.0 * (1 - abs(dte - 11) / 11))

        score = vrp_score + rr_score + theta_score + liq_score + dte_score
        if regime == VolRegime.HIGH:
            score *= 0.75
        return round(min(100.0, score), 2)

    # Calendar Spread
    async def scan_calendars(self, vrp_data: Dict, credit_received: float) -> List[Opportunity]:
        budget = credit_received * cfg.debit_premium_alloc
        regime = vrp_data.get("regime", VolRegime.MEDIUM)

        skew = self.surface.put_call_skew(30)
        if skew is None:
            return []

        cheap_side = skew.get("cheap_side", "call")
        opt_type   = "C" if cheap_side == "call" else "P"
        note       = f"RR={skew['risk_reversal']:+.1f}% → buy {cheap_side} calendar"

        opps = [o for dte in [30, 35, 45]
                if (o := self._build_calendar(dte, opt_type, budget, vrp_data, regime, note)) is not None]
        return sorted(opps, key=lambda x: x.score, reverse=True)

    def _build_calendar(self, dte_far, opt_type, budget, vrp_data, regime, notes) -> Optional[Opportunity]:
        near_dte = (cfg.credit_dte_min + cfg.credit_dte_max) // 2  # midpoint of IC range
        S = self.surface.spot

        near = self._true_atm(near_dte, opt_type, S)
        far  = self._true_atm(dte_far,  opt_type, S)
        if near is None or far is None:
            return None

        net_debit = far["ask"] - near["bid"]
        if net_debit <= 0 or net_debit > budget * 1.2:
            return None

        legs = [
            self._leg(near, "sell", opt_type),
            self._leg(far,  "buy",  opt_type),
        ]
        net_vega = far["vega"] - near["vega"]
        score = 50.0
        score += 15 if net_vega > 0 else 0
        score += 10 if regime == VolRegime.LOW else 0
        score -= np.mean([l.spread_pct for l in legs]) * 50

        return Opportunity(
            strategy_type=StrategyType.CALENDAR_SPREAD,
            legs=legs,
            net_credit=-net_debit,
            net_delta=far["delta"] - near["delta"],
            net_theta=near["theta"] - far["theta"],
            net_vega=net_vega,
            max_profit=net_debit * 1.5,  # rough - real P&L depends on vol path
            max_loss=net_debit,
            score=max(0.0, score),
            vrp=vrp_data.get("vrp_7d", 0),
            regime=regime,
            notes=f"DTE_near={near['dte']} DTE_far={far['dte']} | {notes}",
        )

    def _true_atm(self, dte: int, opt_type: str, spot: float, tol: int = 5) -> Optional[Dict]:
        """ATM via call-put delta parity - delta=0.50 is wrong when skew exists."""
        call_cands = self.surface.get_strikes_by_delta(dte, 0.50, "C", tolerance=tol)
        put_cands  = self.surface.get_strikes_by_delta(dte, 0.50, "P", tolerance=tol)
        if not call_cands or not put_cands:
            return None

        c_map = {c["strike"]: c for c in call_cands}
        p_map = {p["strike"]: p for p in put_cands}
        common = set(c_map) & set(p_map)

        if not common:
            return min(call_cands, key=lambda c: abs(c["strike"] - spot))

        best = min(common, key=lambda k: abs(c_map[k]["delta"] - abs(p_map[k]["delta"])))
        return c_map[best] if opt_type == "C" else p_map[best]

    # main
    async def scan(self) -> Tuple[Optional[Dict], List[Opportunity], List[Opportunity]]:
        vrp_data = await self.check_vrp_signal()
        if vrp_data is None:
            return None, [], []

        condors, calendars = [], []
        if vrp_data["triggered"]:
            condors = await self.scan_iron_condors(vrp_data)
            logger.info(f"{len(condors)} IC candidates")
            if condors:
                calendars = await self.scan_calendars(vrp_data, condors[0].net_credit)
                logger.info(f"{len(calendars)} calendar candidates")

        return vrp_data, condors, calendars
