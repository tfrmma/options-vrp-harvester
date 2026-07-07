import asyncio
import json
import uuid
from typing import Callable, Dict, List, Optional

import httpx
import websockets

from config import cfg
from utils.logger import logger

class DeriveRESTClient:
    """JSON-RPC 2.0 over HTTP. Nothing fancy."""

    def __init__(self):
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url,
            timeout=15.0,
            headers={"Content-Type": "application/json"},
        )

    async def _rpc(self, method: str, params: Dict, auth: bool = False) -> Dict:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params,
        }
        # endpoint is always /public/... or /private/...
        prefix = "private" if auth else "public"
        endpoint = f"/{prefix}/{method.split('/')[-1]}"

        try:
            resp = await self._client.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise ValueError(f"API error on {method}: {data['error']}")
            return data.get("result", data)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} on {method}: {e.response.text[:200]}")
            raise
        except Exception as e:
            logger.error(f"RPC failed [{method}]: {e}")
            raise

    # --- public ---

    async def get_instruments(self, currency: str = "ETH", instrument_type: str = "option") -> List[Dict]:
        result = await self._rpc("public/get_instruments", {
            "currency": currency,
            "instrument_type": instrument_type,
            "expired": False,
        })
        return result if isinstance(result, list) else result.get("instruments", [])

    async def get_ticker(self, instrument_name: str) -> Dict:
        return await self._rpc("public/get_ticker", {"instrument_name": instrument_name})

    async def get_orderbook(self, instrument_name: str, depth: int = 5) -> Dict:
        return await self._rpc("public/get_order_book", {
            "instrument_name": instrument_name,
            "depth": depth,
        })

    async def get_index_price(self, index_name: str = "eth_usd") -> float:
        result = await self._rpc("public/get_index_price", {"index_name": index_name})
        return float(result.get("index_price", 0))

    async def get_spot_trade_history(self, instrument_name: str, count: int = 200) -> List[Dict]:
        # used at startup to seed the RV estimator - don't care if it fails
        try:
            result = await self._rpc("public/get_last_trades_by_instrument", {
                "instrument_name": instrument_name,
                "count": count,
                "sorting": "desc",
            })
            return result if isinstance(result, list) else result.get("trades", [])
        except Exception as e:
            logger.warning(f"Trade history unavailable for {instrument_name}: {e}")
            return []

    async def get_funding_rate(self, instrument_name: str) -> Dict:
        return await self._rpc("public/get_funding_rate", {"instrument_name": instrument_name})

    # --- private (auth required) ---

    async def get_subaccount(self) -> Dict:
        return await self._rpc("private/get_subaccount", {"subaccount_id": cfg.subaccount_id}, auth=True)

    async def get_margin(self, simulated_position_changes: Optional[List] = None) -> Dict:
        """
        Fetch real margin state. With simulated_position_changes, returns
        post_initial_margin and is_valid_trade for a hypothetical trade.

        simulated_position_changes items: {instrument_name: str, amount: str}
        Positive amount = long, negative = short.
        """
        params: Dict = {"subaccount_id": cfg.subaccount_id}
        if simulated_position_changes:
            params["simulated_position_changes"] = simulated_position_changes
        return await self._rpc("private/get_margin", params, auth=True)

    async def place_order(self, order_params: Dict) -> Dict:
        return await self._rpc("private/order", order_params, auth=True)

    async def cancel_order(self, order_id: str) -> Dict:
        return await self._rpc("private/cancel", {
            "subaccount_id": cfg.subaccount_id,
            "order_id": order_id,
        }, auth=True)

    async def cancel_all(self, instrument_name: Optional[str] = None) -> Dict:
        params: Dict = {"subaccount_id": cfg.subaccount_id}
        if instrument_name:
            params["instrument_name"] = instrument_name
        return await self._rpc("private/cancel_all", params, auth=True)

    async def get_open_orders(self) -> List[Dict]:
        result = await self._rpc("private/get_open_orders", {"subaccount_id": cfg.subaccount_id}, auth=True)
        return result if isinstance(result, list) else result.get("orders", [])

    async def get_positions(self) -> List[Dict]:
        result = await self._rpc("private/get_positions", {"subaccount_id": cfg.subaccount_id}, auth=True)
        return result if isinstance(result, list) else result.get("positions", [])

    async def close(self):
        await self._client.aclose()

class DeriveWSClient:
    """
    WS client with automatic reconnection and exponential backoff.
    Subscriptions are replayed on reconnect so callers don't need to care.
    """
    _BACKOFF_BASE = 1.0
    _BACKOFF_MAX  = 60.0
    _BACKOFF_EXP  = 2.0

    def __init__(self):
        self._ws              = None
        self._handlers:        Dict[str, Callable] = {}
        self._prefix_handlers: Dict[str, Callable] = {}  # prefix -> handler
        self._pending:         Dict[str, asyncio.Future] = {}
        self._subscribed:      List[str] = []
        self._running          = False
        self._backoff          = self._BACKOFF_BASE

    def on_channel(self, channel: str, handler: Callable) -> None:
        self._handlers[channel] = handler

    def on_channel_prefix(self, prefix: str, handler: Callable) -> None:
        """Route all channels starting with prefix to handler. Used for ticker.* feeds."""
        self._prefix_handlers[prefix] = handler

    async def connect(self) -> None:
        logger.info(f"WS connecting: {cfg.ws_url}")
        self._ws      = await websockets.connect(cfg.ws_url, ping_interval=20, ping_timeout=10)
        self._running = True
        self._backoff = self._BACKOFF_BASE  # reset on successful connect
        logger.info("WS connected")

    async def subscribe(self, channels: List[str]) -> None:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "public/subscribe",
            "params": {"channels": channels},
        }
        await self._ws.send(json.dumps(payload))
        # track for replay on reconnect
        for ch in channels:
            if ch not in self._subscribed:
                self._subscribed.append(ch)
        logger.debug(f"subscribed: {channels}")

    async def listen(self) -> None:
        """
        Main receive loop with reconnect. Keeps running until disconnect() is called.
        On any connection error, waits _backoff seconds then reconnects and replays subs.
        """
        while self._running:
            try:
                async for raw in self._ws:
                    self._dispatch(raw)
                # clean close from server side
                if self._running:
                    logger.warning("WS closed by server -- reconnecting")
                    await self._reconnect()
            except (websockets.ConnectionClosed, OSError) as e:
                if not self._running:
                    break
                logger.warning(f"WS error: {e} -- reconnecting in {self._backoff:.0f}s")
                await self._reconnect()
            except asyncio.CancelledError:
                break

    async def _reconnect(self) -> None:
        await asyncio.sleep(self._backoff)
        self._backoff = min(self._backoff * self._BACKOFF_EXP, self._BACKOFF_MAX)
        try:
            self._ws      = await websockets.connect(cfg.ws_url, ping_interval=20, ping_timeout=10)
            self._backoff = self._BACKOFF_BASE
            logger.info("WS reconnected")
            if self._subscribed:
                await self.subscribe(self._subscribed)
                logger.info(f"replayed {len(self._subscribed)} subscriptions")
        except Exception as e:
            logger.error(f"WS reconnect failed: {e}")

    def _dispatch(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"bad WS frame: {raw[:80]}")
            return

        if msg.get("method") == "subscription":
            channel = msg["params"]["channel"]
            data    = msg["params"]["data"]

            # exact match first, then prefix match
            handler = self._handlers.get(channel)
            if handler is None:
                for prefix, h in self._prefix_handlers.items():
                    if channel.startswith(prefix):
                        handler = h
                        break

            if handler:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(channel, data))
                else:
                    handler(channel, data)
        elif "id" in msg:
            fut = self._pending.pop(msg["id"], None)
            if fut and not fut.done():
                fut.set_result(msg.get("result"))

    async def disconnect(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info("WS disconnected")
