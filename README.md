# derive-bot

Automated options trading bot for [Derive.xyz](https://derive.xyz).

Strategy: short-dated Iron Condors (7–14 DTE) to harvest the volatility risk premium, with recycled premium funding long Calendar Spreads (30–45 DTE) as tail hedges. Paper/live mode are swappable via a single env var.

---

## Strategy overview

The vol risk premium (VRP) in crypto options is structurally persistent — short-dated IV consistently overestimates realized vol by 5–15 vol points. The bot captures this by:

1. **Credit leg** — Short Iron Condors at ~0.18 delta, 7–14 DTE. Closed at 55% of max profit to avoid gamma risk near expiry.
2. **Debit leg** — Long Calendar Spreads at the ATM strike, 30–45 DTE, financed by ~45% of the net credit received. Direction (call/put) determined by 25D risk-reversal skew.
3. **Delta hedge** — Net portfolio delta hedged via ETH-PERP when `|Δ_net| > 0.25`.

Signal trigger: `vrp_ratio = vrp_7d_pts / iv_7d > threshold` (normalized to avoid misfires across vol regimes). Both absolute floor (1.5 vol pts) and ratio threshold must be satisfied.

---

## Requirements

```
Python 3.11+
pip install -r requirements.txt
```

For live trading only:
```
pip install git+https://github.com/derivexyz/v2-action-signing-python
```

---

## Setup

```bash
cp .env.example .env
# edit .env — at minimum set UNDERLYING and MODE
```

For live mode, fill in `WALLET_PRIVATE_KEY`, `WALLET_ADDRESS`, and `SUBACCOUNT_ID`.

---

## Usage

```bash
# paper trading loop (default)
python main.py

# single signal scan — no trades, just prints surface state
python main.py --scan

# check open positions and realized PnL
python main.py --status

# backtest on synthetic vol data (or DB snapshots if available)
python main.py --backtest --days 90

# terminal dashboard
python main.py --dashboard

# live trading (also set MODE=live in .env)
python main.py --live
```

---

## Project structure

```
derive_bot/
├── config.py                  # env-based config, single cfg singleton
├── main.py                    # CLI entry point
│
├── core/
│   ├── bot.py                 # main loop orchestrator
│   └── derive_client.py       # REST + WebSocket client (JSON-RPC 2.0)
│
├── data/
│   └── vol_surface.py         # surface builder, BS pricer, RV estimator
│
├── signals/
│   └── signal_engine.py       # VRP signal, IC scanner, calendar scanner
│
├── risk/
│   └── risk_engine.py         # pre-trade checks, circuit breakers, close signals
│
├── paper_trading/
│   └── paper_engine.py        # simulated execution (taker fills, real bid/ask)
│
├── execution/
│   └── live_engine.py         # live order placement with Derive v2 signing
│
├── db/
│   └── database.py            # SQLite persistence (positions, signals, PnL)
│
└── utils/
    ├── backtester.py           # historical/synthetic backtest
    ├── dashboard.py            # rich terminal UI
    └── logger.py               # loguru setup
```

---

## Key config parameters

| Variable | Default | Description |
|---|---|---|
| `MODE` | `paper` | `paper` or `live` |
| `UNDERLYING` | `ETH` | `ETH` or `BTC` |
| `TOTAL_CAPITAL_USD` | `2000` | Total capital allocated |
| `CREDIT_DELTA_TARGET` | `0.18` | Delta of short strikes |
| `CREDIT_CLOSE_PCT` | `0.55` | Close credit legs at 55% of max profit |
| `DEBIT_PREMIUM_ALLOC` | `0.45` | Fraction of net credit recycled to debit leg |
| `VRP_THRESHOLD` | `3.0` | Minimum VRP in vol points (normalized internally) |
| `MAX_DRAWDOWN_PCT` | `0.15` | Circuit breaker threshold |
| `MARGIN_BUFFER_PCT` | `0.30` | Capital kept as margin buffer |

---

## Going live checklist

- [ ] Paper trade for at least 2–3 full IC cycles (10–14 days each)
- [ ] Install `derive-action-signing` SDK and verify signing on testnet with a $0 test order
- [ ] Verify API endpoints against `api.lyra.finance` (currently pointed at `api-demo.lyra.finance`)
- [ ] Set `SUBACCOUNT_ID` — create a dedicated subaccount on Derive for the bot
- [ ] Confirm margin engine behavior: open a small IC manually and check that `/private/get_margin` returns expected numbers before letting the bot size positions
- [ ] Set `MODE=live` in `.env`

---

## Known limitations

- Margin estimate uses a static 40% stress buffer over theoretical max loss. For production, replace `_margin_estimate()` in `risk_engine.py` with a live call to `/private/get_margin?simulated_positions=[...]`.
- Backtester uses an OU vol process that's too smooth — it doesn't produce enough vol spike events to stress-test the stop loss logic properly. Real historical snapshots (accumulated from paper trading) will give better results.
- WebSocket client has no reconnect logic. A dropped connection stops the feed silently until the next surface refresh cycle picks it up.
