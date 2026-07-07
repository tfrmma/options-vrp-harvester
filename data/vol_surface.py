import asyncio
import bisect
import itertools
import math
import re
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from config import cfg
from core.derive_client import DeriveRESTClient
from utils.logger import logger

# Black-Scholes
def _d1d2(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)

def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if opt == "C" else max(0.0, K - S)
    d1, d2 = _d1d2(S, K, T, r, sigma)
    if opt == "C":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(price: float, S: float, K: float, T: float, r: float, opt: str) -> Optional[float]:
    if T <= 0 or price <= 0:
        return None
    intrinsic = max(0.0, S - K) if opt == "C" else max(0.0, K - S)
    if price < intrinsic * 0.99:
        return None
    try:
        iv = brentq(lambda s: bs_price(S, K, T, r, s, opt) - price, 1e-6, 10.0, xtol=1e-5, maxiter=200)
        return iv if 0.01 <= iv <= 5.0 else None
    except (ValueError, RuntimeError):
        return None

def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> Dict[str, float]:
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    d1, d2 = _d1d2(S, K, T, r, sigma)
    nd1 = norm.pdf(d1)
    delta = norm.cdf(d1) if opt == "C" else norm.cdf(d1) - 1.0
    gamma = nd1 / (S * sigma * math.sqrt(T))
    vega  = S * nd1 * math.sqrt(T) / 100.0  # per 1% vol move
    theta = (
        -(S * nd1 * sigma) / (2 * math.sqrt(T))
        - r * K * math.exp(-r * T) * (norm.cdf(d2) if opt == "C" else norm.cdf(-d2))
    ) / 365.0
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

# Instrument name
_INST_RE = re.compile(r"^(\w+)-(\d{8})-(\d+(?:\.\d+)?)-([CP])$")

def parse_instrument(name: str) -> Optional[Dict]:
    m = _INST_RE.match(name)
    if not m:
        return None
    currency, exp_str, strike_str, opt_type = m.groups()
    try:
        expiry_dt = datetime.strptime(exp_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        dte = (expiry_dt - datetime.now(timezone.utc)).days
        return {"currency": currency, "expiry_str": exp_str, "expiry_dt": expiry_dt,
                "strike": float(strike_str), "option_type": opt_type, "dte": dte}
    except ValueError:
        return None

# RV Estimator
class RVEstimator:
    """
    Multi-window close-to-close RV with EWMA.
    Keeps 48h of price history; self-calibrates EWMA lambda to actual sample rate.
    """
    _WINDOW_MS = 48 * 3600 * 1000  # 48h in ms

    def __init__(self):
        self._ts:  deque = deque()   # timestamps ms
        self._px:  deque = deque()   # prices

    def add_price(self, ts_ms: float, price: float) -> None:
        self._ts.append(ts_ms)
        self._px.append(price)
        cutoff = ts_ms - self._WINDOW_MS
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()
            self._px.popleft()

    def _returns(self, hours: float) -> np.ndarray:
        if len(self._px) < 2:
            return np.array([])
        cutoff  = self._ts[-1] - hours * 3600 * 1000
        # find the first index >= cutoff using the median interval as a shortcut
        # rather than copying the full deque for bisect
        interval_ms = self._median_interval_min() * 60_000
        n_approx    = int(hours * 3600 * 1000 / interval_ms) + 2
        # take only the tail we need - avoids O(n) full copy
        ts_tail  = list(itertools.islice(reversed(self._ts), n_approx))[::-1]
        px_tail  = list(itertools.islice(reversed(self._px), n_approx))[::-1]
        idx      = bisect.bisect_left(ts_tail, cutoff)
        prices   = px_tail[idx:]
        return np.diff(np.log(prices)) if len(prices) >= 2 else np.array([])

    def rv_annualized(self, hours: float = 24.0) -> Optional[float]:
        r = self._returns(hours)
        if len(r) < 3:
            return None
        # use the actual median sample interval, not the window-derived one -
        # otherwise rv_annualized(1h) and rv_annualized(24h) are incomparable
        interval_h  = self._median_interval_min() / 60.0
        ann_factor  = 365 * 24 / interval_h
        return float(np.std(r, ddof=1) * math.sqrt(ann_factor))

    def ewma_rv(self, lambda_daily: float = 0.94) -> Optional[float]:
        r = self._returns(hours=48)
        if len(r) < 5:
            return None
        # scale lambda to actual sample interval - 0.94 is for daily data,
        # using it raw on 5-min samples is badly wrong
        interval_min = self._median_interval_min()
        lam = lambda_daily ** (interval_min / 1440.0)
        var = r[0] ** 2
        for ret in r[1:]:
            var = lam * var + (1 - lam) * ret ** 2
        ann_factor = (365 * 1440) / interval_min
        return float(math.sqrt(var * ann_factor))

    def _median_interval_min(self) -> float:
        if len(self._ts) < 2:
            return 5.0
        n  = min(20, len(self._ts) - 1)
        ts = list(self._ts)
        intervals = [ts[i+1] - ts[i] for i in range(n)]
        return max(float(np.median(intervals)) / 60_000, 0.1)

    def composite_rv(self) -> Optional[float]:
        estimates = [
            (self.rv_annualized(1),  0.15),
            (self.rv_annualized(4),  0.25),
            (self.rv_annualized(24), 0.40),
            (self.ewma_rv(),         0.20),
        ]
        valid = [(v, w) for v, w in estimates if v is not None and 0.01 < v < 5.0]
        if not valid:
            return None
        total_w = sum(w for _, w in valid)
        return sum(v * w for v, w in valid) / total_w

# Vol Surface
class VolSurface:
    """
    Snapshot of the full ETH options surface.
    Structure: surface[expiry_str][strike][side] = {iv, mid, bid, ask, greeks, ...}
    Refreshed every N minutes by the bot's main loop.
    """
    _BATCH = 20
    _MAX_SPREAD = 0.30  # skip anything wider than 30% of mid

    def __init__(self, client: DeriveRESTClient, underlying: Optional[str] = None, max_dte: int = 60):
        self._client = client
        self._underlying = underlying or cfg.underlying
        self._max_dte = max_dte
        self.spot: float = 0.0
        self.surface: Dict = {}
        self.rv_estimator = RVEstimator()
        self.last_update: Optional[datetime] = None

    async def refresh(self) -> None:
        first_run = self.last_update is None
        logger.info("refreshing vol surface...")

        try:
            self.spot = await self._client.get_index_price(f"{self._underlying.lower()}_usd")
            self.rv_estimator.add_price(datetime.now(timezone.utc).timestamp() * 1000, self.spot)
        except Exception as e:
            logger.warning(f"spot price fetch failed: {e}")
            if self.spot == 0:
                return

        if first_run:
            await self._bootstrap_rv()

        instruments = await self._client.get_instruments(currency=self._underlying)
        if not instruments:
            logger.warning("no instruments returned")
            return

        candidates = [
            (inst["instrument_name"], parsed)
            for inst in instruments
            if (parsed := parse_instrument(inst.get("instrument_name", "")))
            and 0 < parsed["dte"] <= self._max_dte
        ]
        logger.debug(f"{len(candidates)} instruments in DTE range 1-{self._max_dte}")

        new_surface: Dict = {}
        for i in range(0, len(candidates), self._BATCH):
            batch = candidates[i:i + self._BATCH]
            tickers = await asyncio.gather(
                *[self._safe_ticker(name) for name, _ in batch],
                return_exceptions=True,
            )
            for (name, parsed), ticker in zip(batch, tickers):
                if not isinstance(ticker, Exception) and ticker is not None:
                    self._ingest(new_surface, name, parsed, ticker)
            await asyncio.sleep(0.1)

        self.surface = new_surface
        self.last_update = datetime.now(timezone.utc)
        logger.success(f"surface updated: {len(self.surface)} expiries, spot={self.spot:.2f}")

    async def _bootstrap_rv(self) -> None:
        # seed rv estimator with real trades on startup, otherwise we're blind for hours
        perp   = f"{self._underlying}-PERP"
        trades = await self._client.get_spot_trade_history(perp, count=200)
        # API returns desc (newest first) - must ingest oldest-first for deque order
        trades = sorted(trades, key=lambda t: float(t.get("timestamp", 0)))
        seeded = 0
        for t in trades:
            ts = float(t.get("timestamp", 0))
            px = float(t.get("price", 0) or t.get("trade_price", 0))
            if ts > 0 and px > 0:
                self.rv_estimator.add_price(ts, px)
                seeded += 1
        rv = self.rv_estimator.composite_rv()
        logger.info(f"RV bootstrap: {seeded} trades, composite RV={((rv or 0)*100):.1f}%")

    async def _safe_ticker(self, name: str) -> Optional[Dict]:
        try:
            return await self._client.get_ticker(name)
        except Exception:
            return None

    def _ingest(self, surface: Dict, name: str, parsed: Dict, ticker: Dict) -> None:
        try:
            t = ticker.get("result", ticker)
            t = t.get("instrument_ticker", t)

            bid = float(t.get("best_bid_price", 0) or 0)
            ask = float(t.get("best_ask_price", 0) or 0)
            if bid <= 0 and ask <= 0:
                return

            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else (bid or ask)
            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            if spread_pct > self._MAX_SPREAD:
                return

            S, K = self.spot, parsed["strike"]
            T, r  = parsed["dte"] / 365.0, 0.0
            opt   = parsed["option_type"]

            iv_raw = t.get("iv") or t.get("implied_volatility")
            if iv_raw:
                iv = float(iv_raw)
                if iv > 10:
                    iv /= 100.0  # sometimes comes back as percentage
            else:
                iv = implied_vol(mid, S, K, T, r, opt) or 0.0

            if iv <= 0:
                return

            greeks = bs_greeks(S, K, T, r, iv, opt)
            exp_str = parsed["expiry_str"]
            side    = "call" if opt == "C" else "put"

            surface.setdefault(exp_str, {}).setdefault(K, {})[side] = {
                "iv": iv, "mid": mid, "bid": bid, "ask": ask,
                "spread_pct": spread_pct,
                "delta": greeks["delta"], "gamma": greeks["gamma"],
                "theta": greeks["theta"], "vega":  greeks["vega"],
                "instrument_name": name, "dte": parsed["dte"],
            }
        except Exception as e:
            logger.debug(f"ingest error [{name}]: {e}")

    # analytics
    def atm_iv(self, dte_target: int, tolerance: int = 3) -> Optional[float]:
        # two-pass: find closest expiry, then ATM strike within it
        best_exp, best_diff = None, float("inf")
        for exp_str, strikes in self.surface.items():
            for sides in strikes.values():
                # one side per strike is enough to get the DTE
                data = next(iter(sides.values()), None)
                if data is None:
                    continue
                diff = abs(data["dte"] - dte_target)
                if diff < best_diff:
                    best_diff, best_exp = diff, exp_str

        if best_exp is None or best_diff > tolerance:
            return None

        atm_k = min(self.surface[best_exp].keys(), key=lambda k: abs(k - self.spot))
        ivs   = [d["iv"] for d in self.surface[best_exp][atm_k].values() if d.get("iv", 0) > 0]
        return float(np.mean(ivs)) if ivs else None

    def put_call_skew(self, dte_target: int) -> Optional[Dict[str, float]]:
        # RR = put_25d_iv - call_25d_iv  (standard convention, negative = put skew)
        candidates = [
            {**data, "strike": strike, "side": side}
            for exp_str, strikes in self.surface.items()
            for strike, sides in strikes.items()
            for side, data in sides.items()
            if abs(data["dte"] - dte_target) <= 2
        ]
        if len(candidates) < 6:
            return None

        df   = pd.DataFrame(candidates)
        puts  = df[df["side"] == "put"].copy()
        calls = df[df["side"] == "call"].copy()
        if puts.empty or calls.empty:
            return None

        puts["dd"]  = (puts["delta"].abs() - 0.25).abs()
        calls["dd"] = (calls["delta"] - 0.25).abs()
        put_25d  = puts.nsmallest(1, "dd").iloc[0]
        call_25d = calls.nsmallest(1, "dd").iloc[0]
        atm = self.atm_iv(dte_target)
        if atm is None:
            return None

        put_iv, call_iv = float(put_25d["iv"]), float(call_25d["iv"])
        rr = put_iv - call_iv  # negative = put premium (normal crypto skew)
        return {
            "put_25d_iv": put_iv, "call_25d_iv": call_iv, "atm_iv": atm,
            "risk_reversal": rr,
            "butterfly": (put_iv + call_iv) / 2 - atm,
            "put_skew": put_iv - atm, "call_skew": call_iv - atm,
            "cheap_side": "call" if rr < 0 else "put",
        }

    def vrp_signal(self) -> Optional[Dict]:
        iv_7d  = self.atm_iv(7)
        iv_14d = self.atm_iv(14)
        rv     = self.rv_estimator.composite_rv()
        if iv_7d is None or rv is None:
            return None

        iv7_pct  = iv_7d * 100
        vrp_7d   = iv7_pct - rv * 100
        vrp_14d  = ((iv_14d or iv_7d) * 100) - rv * 100
        vrp_ratio = vrp_7d / iv7_pct if iv7_pct > 0 else 0.0

        return {
            "iv_7d": iv7_pct,
            "iv_14d": (iv_14d or 0) * 100,
            "rv_composite": rv * 100,
            "rv_24h": (self.rv_estimator.rv_annualized(24) or 0) * 100,
            "rv_4h":  (self.rv_estimator.rv_annualized(4)  or 0) * 100,
            "vrp_7d": vrp_7d, "vrp_14d": vrp_14d,
            "vrp_ratio": round(vrp_ratio, 4),
            "spot": self.spot,
        }

    def get_strikes_by_delta(
        self, dte_target: int, delta_target: float, opt_type: str, tolerance: int = 2,
    ) -> List[Dict]:
        side = "call" if opt_type == "C" else "put"
        target_abs = abs(delta_target)
        out = []
        for exp_str, strikes in self.surface.items():
            for strike, sides in strikes.items():
                if side not in sides:
                    continue
                data = sides[side]
                dte_diff = abs(data["dte"] - dte_target)
                if dte_diff > tolerance:
                    continue
                out.append({
                    **data, "strike": strike,
                    "delta_diff": abs(abs(data["delta"]) - target_abs),
                    "dte_diff": dte_diff,
                })
        return sorted(out, key=lambda x: (x["delta_diff"], x["dte_diff"]))

    # WS feed

    def ws_channels(self) -> List[str]:
        """
        Return the list of ticker channels for all instruments currently in the surface.
        Called once after the initial REST refresh to build the subscription list.
        Channel format: ticker.{instrument_name}.1  (1 = every update)
        """
        channels = []
        for strikes in self.surface.values():
            for sides in strikes.values():
                for data in sides.values():
                    inst = data.get("instrument_name")
                    if inst:
                        channels.append(f"ticker.{inst}.1")
        # deduplicate -- same instrument can appear in multiple strikes
        return list(dict.fromkeys(channels))

    def on_ws_ticker(self, channel: str, data: Dict) -> None:
        """
        WS handler for ticker.{instrument_name}.1 updates.
        Updates a single instrument in-place without rebuilding the surface.
        Called from DeriveWSClient dispatch -- must be non-blocking.
        """
        try:
            inst_name = channel.split(".")[1] if "." in channel else ""
            if not inst_name:
                return

            parsed = parse_instrument(inst_name)
            if not parsed or not (0 < parsed["dte"] <= self._max_dte):
                return

            ticker = data.get("instrument_ticker", data)
            self._ingest(self.surface, inst_name, parsed, {"instrument_ticker": ticker})

            # keep spot fresh from index_price field if available
            idx = data.get("index_price") or data.get("instrument_ticker", {}).get("index_price")
            if idx:
                px = float(idx)
                if px > 0:
                    self.spot = px
                    self.rv_estimator.add_price(
                        datetime.now(timezone.utc).timestamp() * 1000, px
                    )
        except Exception as e:
            logger.debug(f"ws_ticker error [{channel}]: {e}")
