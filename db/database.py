import aiosqlite
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).parent / "derive_bot.db"

# no more asyncio.Lock -- aiosqlite serializes internally via its own thread pool,
# and sqlite WAL mode handles concurrent reads fine


async def _exec(sql: str, params: tuple = ()) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(sql, params)
        await db.commit()


async def _fetchall(sql: str, params: tuple = ()) -> List[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def _fetchone(sql: str, params: tuple = ()) -> Optional[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def _scalar(sql: str, params: tuple = ()) -> Any:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(sql, params) as cur:
            row = await cur.fetchone()
            return row[0] if row else None


def init_db() -> None:
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS positions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id     TEXT UNIQUE NOT NULL,
            strategy_type   TEXT NOT NULL,
            instrument_name TEXT NOT NULL,
            direction       TEXT NOT NULL,
            amount          REAL NOT NULL,
            entry_price     REAL NOT NULL,
            current_price   REAL,
            delta REAL, gamma REAL, theta REAL, vega REAL,
            iv REAL, dte INTEGER,
            opened_at    TEXT NOT NULL,
            closed_at    TEXT,
            close_price  REAL,
            realized_pnl REAL,
            status   TEXT DEFAULT 'open',
            mode     TEXT DEFAULT 'paper',
            raw_json TEXT
        );
        CREATE TABLE IF NOT EXISTS signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            underlying  TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            value       REAL NOT NULL,
            threshold   REAL,
            triggered   INTEGER DEFAULT 0,
            metadata    TEXT
        );
        CREATE TABLE IF NOT EXISTS pnl_snapshots (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ts             TEXT NOT NULL,
            total_pnl      REAL NOT NULL,
            unrealized_pnl REAL NOT NULL,
            realized_pnl   REAL NOT NULL,
            net_delta REAL, net_theta REAL, net_vega REAL,
            capital_used REAL,
            mode TEXT DEFAULT 'paper'
        );
        CREATE TABLE IF NOT EXISTS orders (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id        TEXT UNIQUE,
            instrument_name TEXT NOT NULL,
            direction       TEXT NOT NULL,
            amount          REAL NOT NULL,
            limit_price     REAL,
            fill_price      REAL,
            status          TEXT NOT NULL,
            strategy_leg    TEXT,
            ts              TEXT NOT NULL,
            mode            TEXT DEFAULT 'paper',
            raw_response    TEXT
        );
        CREATE TABLE IF NOT EXISTS vol_surface_snapshots (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         TEXT NOT NULL,
            underlying TEXT NOT NULL,
            spot_price REAL,
            rv_1h REAL, rv_4h REAL, rv_24h REAL,
            iv_atm_7d REAL, iv_atm_14d REAL, iv_atm_30d REAL,
            vrp_7d REAL, put_skew_30d REAL, call_skew_30d REAL
        );
    """)
    conn.commit()
    conn.close()


# positions

async def insert_position(pos: Dict[str, Any]) -> None:
    await _exec("""
        INSERT OR REPLACE INTO positions
        (position_id, strategy_type, instrument_name, direction, amount,
         entry_price, delta, gamma, theta, vega, iv, dte,
         opened_at, status, mode, raw_json)
        VALUES (:position_id, :strategy_type, :instrument_name, :direction, :amount,
                :entry_price, :delta, :gamma, :theta, :vega, :iv, :dte,
                :opened_at, :status, :mode, :raw_json)
    """, tuple(pos[k] for k in [
        "position_id", "strategy_type", "instrument_name", "direction", "amount",
        "entry_price", "delta", "gamma", "theta", "vega", "iv", "dte",
        "opened_at", "status", "mode", "raw_json",
    ]))


async def update_position_close(position_id: str, close_price: float, realized_pnl: float) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
            UPDATE positions
            SET status='closed', closed_at=?, close_price=?, realized_pnl=?
            WHERE position_id=? AND status='open'
        """, (datetime.utcnow().isoformat(), close_price, realized_pnl, position_id))
        await db.commit()
        return cur.rowcount > 0


async def update_position_greeks(
    position_id: str, current_price: float,
    delta: float, gamma: float, theta: float, vega: float, dte: int,
) -> None:
    await _exec("""
        UPDATE positions
        SET current_price=?, delta=?, gamma=?, theta=?, vega=?, dte=?
        WHERE position_id=? AND status='open'
    """, (current_price, delta, gamma, theta, vega, dte, position_id))


async def get_open_positions(mode: str = "paper") -> List[Dict]:
    return await _fetchall("SELECT * FROM positions WHERE status='open' AND mode=?", (mode,))


async def get_position_by_id(position_id: str) -> Optional[Dict]:
    return await _fetchone("SELECT * FROM positions WHERE position_id=? AND status='open'", (position_id,))


async def get_realized_pnl_total(mode: str = "paper") -> float:
    val = await _scalar(
        "SELECT COALESCE(SUM(realized_pnl), 0) FROM positions WHERE status='closed' AND mode=?", (mode,)
    )
    return float(val or 0)


# signals / snapshots

async def insert_signal(sig: Dict[str, Any]) -> None:
    await _exec("""
        INSERT INTO signals (ts, underlying, signal_type, value, threshold, triggered, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (sig["ts"], sig["underlying"], sig["signal_type"],
          sig["value"], sig["threshold"], sig["triggered"], sig["metadata"]))


async def insert_pnl_snapshot(snap: Dict[str, Any]) -> None:
    await _exec("""
        INSERT INTO pnl_snapshots
        (ts, total_pnl, unrealized_pnl, realized_pnl, net_delta, net_theta, net_vega, capital_used, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (snap["ts"], snap["total_pnl"], snap["unrealized_pnl"], snap["realized_pnl"],
          snap["net_delta"], snap["net_theta"], snap["net_vega"], snap["capital_used"], snap["mode"]))


async def insert_vol_snapshot(snap: Dict[str, Any]) -> None:
    await _exec("""
        INSERT INTO vol_surface_snapshots
        (ts, underlying, spot_price, rv_1h, rv_4h, rv_24h,
         iv_atm_7d, iv_atm_14d, iv_atm_30d, vrp_7d, put_skew_30d, call_skew_30d)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (snap["ts"], snap["underlying"], snap["spot_price"],
          snap["rv_1h"], snap["rv_4h"], snap["rv_24h"],
          snap["iv_atm_7d"], snap["iv_atm_14d"], snap["iv_atm_30d"],
          snap["vrp_7d"], snap["put_skew_30d"], snap["call_skew_30d"]))
