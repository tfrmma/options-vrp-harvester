from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime, timezone

from config import cfg
from core.derive_client import DeriveRESTClient, DeriveWSClient
from data.vol_surface import VolSurface
from db.database import init_db, get_open_positions, insert_vol_snapshot
from paper_trading.paper_engine import PaperTradingEngine
from risk.risk_engine import RiskEngine
from signals.signal_engine import SignalEngine
from utils.logger import logger, setup_logger

if cfg.is_live:
    from execution.live_engine import LiveExecutionEngine


class DeriveBot:
    # REST refresh still runs as a fallback and to pick up new instruments
    # that listed after the last WS subscription batch
    _SURFACE_INTERVAL = 300   # 5min
    _SCAN_INTERVAL    = 60    # 1min
    _MONITOR_INTERVAL = 120   # 2min

    def __init__(self):
        setup_logger(cfg.log_level)
        self._rest    = DeriveRESTClient()
        self._ws      = DeriveWSClient()
        self._surface = VolSurface(self._rest)
        self._signals = SignalEngine(self._surface)
        self._risk    = RiskEngine()
        self._exec    = PaperTradingEngine(self._surface) if cfg.is_paper else LiveExecutionEngine(self._rest)
        self._running = False

        mode_label = "PAPER" if cfg.is_paper else "LIVE"
        logger.info(f"DeriveBot starting [{mode_label}] {cfg.underlying} capital=${cfg.total_capital_usd:,.0f}")

    async def start(self) -> None:
        init_db()
        try:
            cfg.validate()
        except AssertionError as e:
            logger.critical(f"config error: {e}")
            sys.exit(1)

        # initial REST snapshot -- gives us the instrument list for WS subs
        await self._surface.refresh()

        # subscribe WS to all active ticker channels
        await self._ws.connect()
        channels = self._surface.ws_channels()
        if channels:
            await self._ws.subscribe(channels)
            self._ws.on_channel_prefix("ticker.", self._surface.on_ws_ticker)
            logger.info(f"WS: subscribed to {len(channels)} ticker channels")
        else:
            logger.warning("no channels to subscribe -- surface empty?")

        self._running = True

    async def run(self) -> None:
        await self.start()
        tasks = [
            asyncio.create_task(self._ws.listen()),          # WS feed (continuous)
            asyncio.create_task(self._loop_surface()),        # REST fallback refresh
            asyncio.create_task(self._loop_scan()),
            asyncio.create_task(self._loop_monitor()),
        ]
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_event_loop().add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    # background loops

    async def _loop_surface(self) -> None:
        """
        Periodic REST rebuild. Catches new listings and corrects any drift
        that accumulates from WS updates. Also re-subscribes to any new
        instruments that weren't in the original subscription batch.
        """
        while self._running:
            await asyncio.sleep(self._SURFACE_INTERVAL)
            try:
                await self._surface.refresh()
                await self._snapshot_surface()
                # re-subscribe to any instruments that appeared since last refresh
                new_channels = [
                    ch for ch in self._surface.ws_channels()
                    if ch not in self._ws._subscribed
                ]
                if new_channels:
                    await self._ws.subscribe(new_channels)
                    logger.info(f"WS: added {len(new_channels)} new channels")
            except Exception as e:
                logger.error(f"surface refresh: {e}")
                await asyncio.sleep(30)

    async def _snapshot_surface(self) -> None:
        """Persist a vol surface snapshot to DB for backtesting and reporting."""
        s   = self._surface
        vrp = s.vrp_signal()
        if vrp is None:
            return
        skew = s.put_call_skew(30)
        try:
            await insert_vol_snapshot({
                "ts":            datetime.now(timezone.utc).isoformat(),
                "underlying":    cfg.underlying,
                "spot_price":    s.spot,
                "rv_1h":         vrp.get("rv_composite", 0) / 100,
                "rv_4h":         vrp.get("rv_4h", 0) / 100,
                "rv_24h":        vrp.get("rv_24h", 0) / 100,
                "iv_atm_7d":     vrp.get("iv_7d", 0) / 100,
                "iv_atm_14d":    vrp.get("iv_14d", 0) / 100,
                "iv_atm_30d":    s.atm_iv(30) or 0,
                "vrp_7d":        vrp.get("vrp_7d", 0),
                "put_skew_30d":  skew.get("put_skew", 0) if skew else 0,
                "call_skew_30d": skew.get("call_skew", 0) if skew else 0,
            })
        except Exception as e:
            logger.debug(f"vol snapshot failed (non-fatal): {e}")

    async def _loop_scan(self) -> None:
        while self._running:
            await asyncio.sleep(self._SCAN_INTERVAL)
            try:
                await self._scan_cycle()
            except Exception as e:
                logger.error(f"scan cycle: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def _loop_monitor(self) -> None:
        while self._running:
            await asyncio.sleep(self._MONITOR_INTERVAL)
            try:
                await self._monitor_positions()
            except Exception as e:
                logger.error(f"monitor: {e}")
                await asyncio.sleep(30)

    # core logic

    async def _scan_cycle(self) -> None:
        positions = await get_open_positions(mode=cfg.mode)
        state     = await self._risk.update_portfolio_state(positions=positions, spot=self._surface.spot)

        broken, reason = self._risk.check_circuit_breakers(state)
        if broken:
            logger.warning(f"circuit breaker: {reason}")
            return

        vrp_data, condors, calendars = await self._signals.scan()
        if vrp_data is None:
            return

        logger.info(
            f"VRP={vrp_data.get('vrp_7d',0):.1f}pts "
            f"IV7d={vrp_data.get('iv_7d',0):.1f}% "
            f"RV={vrp_data.get('rv_composite',0):.1f}% "
            f"[{vrp_data.get('regime','?')}]"
        )

        if not vrp_data.get("triggered"):
            return

        if condors:
            await self._try_open(condors[0], state)
            if calendars:
                await self._try_open(calendars[0], state)

        await self._maybe_hedge(state)

    async def _try_open(self, opp, state) -> None:
        # live mode: pass REST client so risk engine can call private/get_margin
        rest = self._rest if cfg.is_live else None
        decision = await self._risk.approve_trade(opp, state, rest_client=rest)
        if decision.approved:
            if decision.notes:
                logger.warning(decision.notes)
            await self._exec.open_position(opp, decision)
        else:
            logger.info(f"trade rejected: {decision.reason}")

    async def _monitor_positions(self) -> None:
        positions = await get_open_positions(mode=cfg.mode)
        if not positions:
            return

        prices = {
            data["instrument_name"]: data.get("mid", 0)
            for strikes in self._surface.surface.values()
            for sides in strikes.values()
            for data in sides.values()
            if data.get("instrument_name")
        }

        for pos in self._risk.check_close_signals(positions, prices):
            urgency = pos.get("urgency", "normal")
            logger.info(f"{'URGENT ' if urgency=='urgent' else ''}close: {pos.get('instrument_name')} - {pos['close_reason']}")
            await self._exec.close_position(pos["position_id"], pos["close_reason"], surface=self._surface)

        if cfg.is_paper:
            await self._exec.update_position_greeks(self._surface)

    async def _maybe_hedge(self, state) -> None:
        hedge = self._risk.delta_hedge_required(state)
        if not hedge:
            return
        if cfg.is_paper:
            logger.info(f"PAPER hedge (not executed): {hedge['direction']} {hedge['amount']} {hedge['instrument_name']}")
        else:
            await self._exec.hedge_delta(hedge)

    async def shutdown(self) -> None:
        logger.info("shutting down...")
        self._running = False
        await self._ws.disconnect()
        await self._rest.close()


async def main():
    await DeriveBot().run()

if __name__ == "__main__":
    asyncio.run(main())
