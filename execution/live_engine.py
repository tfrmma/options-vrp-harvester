from __future__ import annotations

import asyncio
import random
import time
from decimal import Decimal
from typing import Dict, List, Optional

from web3 import Web3
from derive_action_signing import SignedAction, TradeModuleData, utils as sdk_utils

from config import cfg
from core.derive_client import DeriveRESTClient
from signals.signal_engine import Leg, Opportunity
from utils.logger import logger

# Protocol constants -- from docs.derive.xyz/reference/protocol-constants
# These are mainnet values. Testnet uses different addresses.
_MAINNET_TRADE_MODULE  = "0xB8D20c2B7a1Ad2EE33Bc50eF10876eD3035b5e7b"
_TESTNET_TRADE_MODULE  = "0x87F2863866D85E3192a35A73b388BD625D83f2be"
_ACTION_TYPEHASH       = "4d7a9f27c403ff9c0f19bce61d76d82f9aa29f8d6d4b0c5474607d9770d1af17"

# DOMAIN_SEPARATOR is also in the protocol constants table.
# Must be fetched once at startup or read from env -- don't hardcode here.
# See .env.example: DERIVE_DOMAIN_SEPARATOR


def _nonce() -> int:
    return sdk_utils.get_action_nonce()


def _trade_module() -> str:
    return _MAINNET_TRADE_MODULE if cfg.is_live else _TESTNET_TRADE_MODULE


class LiveExecutionEngine:
    """
    Live execution using the official Derive v2 action signing SDK.

    Key difference from the old stub: orders are signed via SignedAction +
    TradeModuleData, not a hand-rolled EIP-712 dict. The SDK handles the
    domain separator and action typehash correctly.

    Required env vars (in addition to the usual):
        DERIVE_WALLET_ADDRESS     -- smart contract wallet on Derive Chain
                                     (NOT your EOA -- find it in Home > Developers > Derive Wallet)
        DERIVE_DOMAIN_SEPARATOR   -- from protocol constants table on docs.derive.xyz
    """

    def __init__(self, client: DeriveRESTClient):
        if not cfg.wallet_private_key:
            raise ValueError("WALLET_PRIVATE_KEY required")
        if not cfg.derive_wallet_address:
            raise ValueError("DERIVE_WALLET_ADDRESS required (smart contract wallet, not EOA)")
        if not cfg.domain_separator:
            raise ValueError("DERIVE_DOMAIN_SEPARATOR required -- see docs.derive.xyz/reference/protocol-constants")

        self._client       = client
        self._session_key  = Web3().eth.account.from_key(cfg.wallet_private_key)
        self._domain_sep   = bytes.fromhex(cfg.domain_separator.lstrip("0x"))
        self._typehash     = bytes.fromhex(_ACTION_TYPEHASH)
        logger.info(f"live engine: signer={self._session_key.address}")

    def _sign(self, leg: Leg, amount: float, ticker: Dict) -> Dict:
        """
        Build and sign a single-leg order using the official SDK.
        Returns the signed payload ready for /private/order.
        """
        is_bid     = leg.direction == "buy"
        limit_px   = Decimal(str(leg.ask * 1.005 if is_bid else leg.bid * 0.995))
        limit_px   = max(limit_px, Decimal("0.0001"))
        max_fee    = Decimal(str(round(leg.mid * 0.03 * amount + 0.50, 4)))

        action = SignedAction(
            subaccount_id        = cfg.subaccount_id,
            owner                = cfg.derive_wallet_address,
            signer               = self._session_key.address,
            signature_expiry_sec = sdk_utils.MAX_INT_32,
            nonce                = _nonce(),
            module_address       = _trade_module(),
            module_data          = TradeModuleData(
                asset        = ticker["base_asset_address"],
                sub_id       = int(ticker["base_asset_sub_id"]),
                limit_price  = limit_px,
                amount       = Decimal(str(round(amount, 4))),
                max_fee      = max_fee,
                recipient_id = cfg.subaccount_id,
                is_bid       = is_bid,
            ),
            DOMAIN_SEPARATOR = self._domain_sep,
            ACTION_TYPEHASH  = self._typehash,
        )
        action.sign(self._session_key.key)

        return {
            "instrument_name":      leg.instrument_name,
            "subaccount_id":        cfg.subaccount_id,
            "direction":            leg.direction,
            "amount":               str(round(amount, 4)),
            "limit_price":          str(limit_px),
            "max_fee":              str(max_fee),
            "order_type":           "limit",
            "time_in_force":        "post_only",
            "nonce":                action.nonce,
            "signature":            action.signature,
            "signature_expiry_sec": action.signature_expiry_sec,
            "signer":               self._session_key.address,
            "label":                f"thetavore_{leg.option_type}_{leg.strike:.0f}",
            "mmp":                  False,
        }

    async def _ticker(self, instrument_name: str) -> Dict:
        """Fetch ticker to get base_asset_address and base_asset_sub_id for signing."""
        raw = await self._client.get_ticker(instrument_name)
        t   = raw.get("result", raw)
        t   = t.get("instrument_ticker", t)
        assert "base_asset_address" in t, f"ticker missing base_asset_address: {instrument_name}"
        return t

    async def open_position(self, opp: Opportunity, contracts: float) -> List[Dict]:
        if not cfg.is_live:
            raise RuntimeError("live engine called in non-live mode")

        # sells first -- collect premium before paying for wings
        legs   = sorted(opp.legs, key=lambda l: 0 if l.direction == "sell" else 1)
        filled: List[Dict] = []
        oids:   List[str]  = []

        logger.info(f"LIVE open: {opp.strategy_type.value} x{contracts:.0f} VRP={opp.vrp:.1f}pts")

        for leg in legs:
            try:
                ticker = await self._ticker(leg.instrument_name)
                order  = self._sign(leg, contracts, ticker)
                logger.info(f"  {leg.direction.upper()} {leg.instrument_name} x{contracts} @ ${float(order['limit_price']):.4f}")
                resp   = await self._client.place_order(order)
                filled.append(resp)
                oid    = resp.get("order_id") or resp.get("result", {}).get("order_id")
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
        ticker    = await self._ticker(instrument_name)
        leg       = Leg(
            instrument_name=instrument_name, direction=close_dir,
            option_type="?", strike=0, dte=0, iv=0,
            mid=current_price, bid=current_price * 0.99, ask=current_price * 1.01,
            delta=0, gamma=0, theta=0, vega=0, spread_pct=0.02,
        )
        order = self._sign(leg, amount, ticker)
        order["time_in_force"] = "gtc"  # closer -- don't post-only
        return await self._client.place_order(order)

    async def hedge_delta(self, spec: Dict) -> Optional[Dict]:
        ticker = await self._ticker(spec["instrument_name"])
        leg    = Leg(
            instrument_name=spec["instrument_name"],
            direction=spec["direction"],
            option_type="?", strike=0, dte=0, iv=0,
            mid=0, bid=0, ask=0,
            delta=0, gamma=0, theta=0, vega=0, spread_pct=0.0,
        )
        # for perp hedge: use ioc, price doesn't matter much
        order = self._sign(leg, spec["amount"], ticker)
        order["time_in_force"] = "ioc"
        logger.info(f"LIVE hedge: {spec['direction']} {spec['amount']} {spec['instrument_name']}")
        return await self._client.place_order(order)

    async def _cancel_all(self, order_ids: List[str]) -> None:
        logger.warning(f"emergency cancel: {order_ids}")
        for oid in order_ids:
            try:
                await self._client.cancel_order(oid)
            except Exception as e:
                logger.error(f"cancel failed {oid}: {e}")
