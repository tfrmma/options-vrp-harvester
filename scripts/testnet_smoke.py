"""
scripts/testnet_smoke.py -- Testnet signing smoke test.

Verifies the full signing pipeline against api-demo.lyra.finance
BEFORE putting real money in. Places one signed order at 1% of ask
(will never fill) then immediately cancels it.

Usage:
    python scripts/testnet_smoke.py

Required .env:
    WALLET_PRIVATE_KEY, DERIVE_WALLET_ADDRESS, DERIVE_DOMAIN_SEPARATOR, SUBACCOUNT_ID
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decimal import Decimal
from web3 import Web3
from derive_action_signing import SignedAction, TradeModuleData, utils as sdk_utils

from config import cfg
from core.derive_client import DeriveRESTClient
from utils.logger import logger, setup_logger

_TESTNET_TRADE_MODULE = "0x87F2863866D85E3192a35A73b388BD625D83f2be"
_ACTION_TYPEHASH      = "4d7a9f27c403ff9c0f19bce61d76d82f9aa29f8d6d4b0c5474607d9770d1af17"


async def run():
    setup_logger("DEBUG")

    missing = [k for k, v in {
        "WALLET_PRIVATE_KEY":    cfg.wallet_private_key,
        "DERIVE_WALLET_ADDRESS": cfg.derive_wallet_address,
        "DERIVE_DOMAIN_SEPARATOR": cfg.domain_separator,
        "SUBACCOUNT_ID":         cfg.subaccount_id,
    }.items() if not v]
    if missing:
        logger.error(f"missing env vars: {missing}")
        sys.exit(1)

    rest        = DeriveRESTClient()
    session_key = Web3().eth.account.from_key(cfg.wallet_private_key)
    domain_sep  = bytes.fromhex(cfg.domain_separator.lstrip("0x"))
    typehash    = bytes.fromhex(_ACTION_TYPEHASH)

    logger.info(f"signer={session_key.address} subaccount={cfg.subaccount_id}")

    # find a live instrument with a valid ticker
    instruments = await rest.get_instruments("ETH", "option")
    test_inst   = None
    for inst in instruments[:20]:
        name = inst.get("instrument_name", "")
        try:
            raw    = await rest.get_ticker(name)
            t      = raw.get("result", raw)
            t      = t.get("instrument_ticker", t)
            if t.get("base_asset_address") and float(t.get("best_ask_price") or 0) > 0:
                test_inst = (name, t)
                break
        except Exception:
            continue

    if not test_inst:
        logger.error("no usable instrument found on testnet")
        await rest.close()
        sys.exit(1)

    inst_name, ticker = test_inst
    ask_price   = float(ticker["best_ask_price"])
    dummy_price = Decimal(str(round(ask_price * 0.01, 4)))
    logger.info(f"instrument={inst_name} ask=${ask_price:.4f} test_bid=${dummy_price}")

    action = SignedAction(
        subaccount_id        = cfg.subaccount_id,
        owner                = cfg.derive_wallet_address,
        signer               = session_key.address,
        signature_expiry_sec = sdk_utils.MAX_INT_32,
        nonce                = sdk_utils.get_action_nonce(),
        module_address       = _TESTNET_TRADE_MODULE,
        module_data          = TradeModuleData(
            asset        = ticker["base_asset_address"],
            sub_id       = int(ticker["base_asset_sub_id"]),
            limit_price  = dummy_price,
            amount       = Decimal("0.1"),
            max_fee      = Decimal("10"),
            recipient_id = cfg.subaccount_id,
            is_bid       = True,
        ),
        DOMAIN_SEPARATOR = domain_sep,
        ACTION_TYPEHASH  = typehash,
    )
    action.sign(session_key.key)
    logger.info(f"signing OK: {action.signature[:24]}...")

    payload = {
        "instrument_name":      inst_name,
        "subaccount_id":        cfg.subaccount_id,
        "direction":            "buy",
        "amount":               "0.1",
        "limit_price":          str(dummy_price),
        "max_fee":              "10",
        "order_type":           "limit",
        "time_in_force":        "gtc",
        "nonce":                action.nonce,
        "signature":            action.signature,
        "signature_expiry_sec": action.signature_expiry_sec,
        "signer":               session_key.address,
        "label":                "smoke_test",
        "mmp":                  False,
    }

    try:
        resp     = await rest.place_order(payload)
        order_id = resp.get("order_id") or resp.get("result", {}).get("order_id")
        logger.success(f"order accepted: {order_id}")
        if order_id:
            await rest.cancel_order(order_id)
            logger.success("order cancelled")
        logger.success("SMOKE TEST PASSED")
    except Exception as e:
        logger.error(f"SMOKE TEST FAILED: {e}")
        logger.error("check DERIVE_DOMAIN_SEPARATOR and DERIVE_WALLET_ADDRESS")
        await rest.close()
        sys.exit(1)

    await rest.close()


if __name__ == "__main__":
    asyncio.run(run())
