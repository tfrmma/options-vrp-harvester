from __future__ import annotations

import asyncio
import random
import time
from typing import Dict, List, Optional

from eth_account import Account

from config import cfg
from core.derive_client import DeriveRESTClient
from signals.signal_engine import Leg, Opportunity
from utils.logger import logger

def _nonce() -> int:
    # ts_ms + 3-digit random suffix - Derive rejects duplicate nonces
    return int(f"{int(time.time() * 1000)}{random.randint(0, 999):03d}")

class LiveExecutionEngine:
    """
    Live execution against Derive. Wraps DeriveRESTClient with signing.
    Only instantiated when MODE=live - don't call this from paper mode.
    """

    def __init__(self, client: DeriveRESTClient):
        if not cfg.wallet_private_key:
            raise ValueError("WALLET_PRIVATE_KEY required")
        self._client  = client
        self._account = Account.from_key(cfg.wallet_private_key)
        logger.info(f"live engine: wallet={cfg.wallet_address}")

    def _sign_order(self, params: Dict) -> str:
        """
        Signing uses Derive's v2 ActionSigning contract - do NOT hand-roll
        the EIP-712 structure. Their domain separator is non-standard and
        any byte-level deviation causes silent rejection with no useful error.

        Install the official SDK first:
            pip install git+https://github.com/derivexyz/v2-action-signing-python
        """
        try:
            from derive_action_signing import sign_order  # type: ignore
            return sign_order(
                private_key=cfg.wallet_private_key,
                subaccount_id=cfg.subaccount_id,
                instrument_name=params["instrument_name"],
                direction=params["direction"],
                amount=float(params["amount"]),
                limit_price=float(params["limit_price"]),
                max_fee=float(params["max_fee"]),
                nonce=params["nonce"],
                expiry=params["signature_expiry_sec"],
            )
        except ImportError:
            raise NotImplementedError(
                "pip install git+https://github.com/derivexyz/v2-action-signing-python"
            )

    def _limit_order(self, leg: Leg, amount: float, post_only: bool = True) -> Dict:
        # options: always limit, never market - slippage on OTM options is ugly
        if leg.direction == "buy":
            price = (leg.ask * 1.005) if leg.ask > 0 else leg.mid * 1.01
        else:
            price = (leg.bid * 0.995) if leg.bid > 0 else leg.mid * 0.99

        price   = round(max(price, 1e-4), 6)
        max_fee = round(leg.mid * 0.03 * amount + 0.50, 4)
        nonce   = _nonce()
        expiry  = int(time.time()) + 120

        order = {
            "instrument_name":     leg.instrument_name,
            "subaccount_id":       cfg.subaccount_id,
            "direction":           leg.direction,
            "amount":              str(round(amount, 4)),
            "limit_price":         str(price),
            "max_fee":             str(max_fee),
            "order_type":          "limit",
            "time_in_force":       "post_only" if post_only else "gtc",
            "nonce":               nonce,
            "signature_expiry_sec": expiry,
            "signer":              cfg.wallet_address,
            "label":               f"bot_{leg.option_type}_{leg.strike:.0f}",
            "mmp":                 False,
        }
        order["signature"] = self._sign_order(order)
        return order

    async def open_position(self, opp: Opportunity, contracts: float) -> List[Dict]:
        if not cfg.is_live:
            raise RuntimeError("live engine called in non-live mode")

        # sells first so premium hits the account before we pay for wings
        legs   = sorted(opp.legs, key=lambda l: 0 if l.direction == "sell" else 1)
        filled: List[Dict] = []
        oids:   List[str]  = []

        logger.info(f"LIVE open: {opp.strategy_type.value} x{contracts:.0f} VRP={opp.vrp:.1f}pts")
        for leg in legs:
            try:
                order = self._limit_order(leg, contracts)
                logger.info(f"  {leg.direction.upper()} {leg.instrument_name} x{contracts} @ ${float(order['limit_price']):.4f}")
                resp = await self._client.place_order(order)
                filled.append(resp)
                oid  = resp.get("order_id") or resp.get("result", {}).get("order_id")
                if oid:
                    oids.append(oid)
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"leg failed {leg.instrument_name}: {e}")
                if oids:
                    await self._cancel_all(oids)
                raise

        logger.info(f"LIVE open complete: {len(filled)} legs")
        return filled

    async def close_position(
        self, instrument_name: str, direction: str, amount: float, current_price: float,
    ) -> Optional[Dict]:
        close_dir = "buy" if direction == "sell" else "sell"
        leg = Leg(
            instrument_name=instrument_name, direction=close_dir,
            option_type="?", strike=0, dte=0, iv=0,
            mid=current_price, bid=current_price * 0.99, ask=current_price * 1.01,
            delta=0, gamma=0, theta=0, vega=0, spread_pct=0.02,
        )
        return await self._client.place_order(self._limit_order(leg, amount, post_only=False))

    async def hedge_delta(self, spec: Dict) -> Optional[Dict]:
        # IOC on perp - delta hedge is time-sensitive, GTC would be weird here
        order = {
            "instrument_name":     spec["instrument_name"],
            "subaccount_id":       cfg.subaccount_id,
            "direction":           spec["direction"],
            "amount":              str(spec["amount"]),
            "limit_price":         "0",
            "max_fee":             "50",
            "order_type":          "limit",
            "time_in_force":       "ioc",
            "nonce":               _nonce(),
            "signature_expiry_sec": int(time.time()) + 30,
            "signer":              cfg.wallet_address,
            "label":               "delta_hedge",
            "mmp":                 False,
        }
        order["signature"] = self._sign_order(order)
        logger.info(f"LIVE hedge: {spec['direction']} {spec['amount']} {spec['instrument_name']}")
        return await self._client.place_order(order)

    async def _cancel_all(self, order_ids: List[str]) -> None:
        logger.warning(f"emergency cancel: {order_ids}")
        for oid in order_ids:
            try:
                await self._client.cancel_order(oid)
            except Exception as e:
                logger.error(f"cancel failed {oid}: {e}")
