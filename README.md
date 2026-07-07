# options-vrp-harvester

Automated options trading bot for [Derive.xyz](https://derive.xyz).

Strategy: short Iron Condors (7-14 DTE) to harvest the volatility risk premium, with recycled premium funding long Calendar Spreads (30-45 DTE) as tail hedges. Delta hedged via ETH-PERP. Paper and live mode are swappable via a single env var.

---

## Strategy

The vol risk premium (VRP) in crypto options is structurally persistent — short-dated IV consistently overestimates realized vol. The bot captures this by:

1. **Credit leg** — Short Iron Condors at ~0.18 delta, 7-14 DTE. Closed at 55% of max profit to avoid gamma risk near expiry.
2. **Debit leg** — Long Calendar Spreads at the ATM strike, 30-45 DTE, financed by ~45% of the net credit. Direction (call/put) set by 25D risk-reversal skew.
3. **Delta hedge** — Net portfolio delta hedged via ETH-PERP when `|Δ_net| > 0.25`.

Signal trigger: normalized VRP ratio (`vrp_pts / iv_7d`) must exceed threshold. Both absolute floor and ratio threshold required to avoid misfires across vol regimes.

Vol surface updated continuously via WebSocket (ticker feed), with a REST rebuild every 5 minutes to catch new listings.

---

## Requirements

```
Python 3.11+
pip install -r requirements.txt
```

---

## Setup

```bash
cp env.example .env
# edit .env
```

**Required for live trading** (in addition to the standard fields):

- `WALLET_PRIVATE_KEY` — session key private key
- `WALLET_ADDRESS` — session key address (EOA)
- `SUBACCOUNT_ID` — your Derive subaccount ID
- `DERIVE_WALLET_ADDRESS` — smart contract wallet on Derive Chain. **Not your EOA.** Find it at Home → Developers → "Derive Wallet" in the Derive UI.
- `DERIVE_DOMAIN_SEPARATOR` — from [docs.derive.xyz/reference/protocol-constants](https://docs.derive.xyz/reference/protocol-constants)

---

## Usage

```bash
# paper trading loop (default)
python main.py

# live trading
python main.py --live

# single signal scan, no trades
python main.py --scan

# check open positions and realized PnL
python main.py --status

# daily performance report
python main.py --report
python main.py --report --days 30

# backtest (uses DB snapshots if available, synthetic data otherwise)
python main.py --backtest --days 90

# terminal dashboard
python main.py --dashboard
```

---

## Going live checklist

- [ ] Run `python scripts/testnet_smoke.py` — verifies the full signing pipeline against testnet before touching mainnet. Must print `SMOKE TEST PASSED`.
- [ ] Paper trade for at least 2-3 full IC cycles (10-14 days each) and review `--report` output
- [ ] Set `DERIVE_BASE_URL=https://api.lyra.finance` and `DERIVE_WS_URL=wss://api.lyra.finance/ws` in `.env`
- [ ] Set `MODE=live`
- [ ] Confirm margin behavior: open a small IC manually on Derive and verify that `private/get_margin` returns the expected numbers before letting the bot size positions autonomously

---

## Project structure

```
thetavore/
├── config.py                  # env-based config, single cfg singleton
├── main.py                    # CLI entry point
│
├── core/
│   ├── bot.py                 # main loop orchestrator
│   └── derive_client.py       # REST + WebSocket client (JSON-RPC 2.0, auto-reconnect)
│
├── data/
│   └── vol_surface.py         # surface builder, BS pricer, RV estimator, WS feed handler
│
├── signals/
│   └── signal_engine.py       # VRP signal, IC scanner, calendar scanner
│
├── risk/
│   └── risk_engine.py         # pre-trade checks, live margin API, circuit breakers
│
├── paper_trading/
│   └── paper_engine.py        # simulated execution (taker fills, real bid/ask)
│
├── execution/
│   └── live_engine.py         # live order placement with Derive v2 action signing
│
├── db/
│   └── database.py            # aiosqlite persistence (positions, signals, PnL, vol snapshots)
│
├── scripts/
│   └── testnet_smoke.py       # signing pipeline verification against testnet
│
└── utils/
    ├── backtester.py           # historical/synthetic backtest with full BS revaluation
    ├── dashboard.py            # rich terminal UI
    ├── logger.py               # loguru setup
    └── report.py               # daily performance report
```

---

## Key config parameters

| Variable | Default | Description |
|---|---|---|
| `MODE` | `paper` | `paper` or `live` |
| `UNDERLYING` | `ETH` | `ETH` or `BTC` |
| `TOTAL_CAPITAL_USD` | `2000` | Total capital allocated |
| `CREDIT_DELTA_TARGET` | `0.18` | Delta of short IC strikes |
| `CREDIT_CLOSE_PCT` | `0.55` | Close credit legs at 55% of max profit |
| `DEBIT_PREMIUM_ALLOC` | `0.45` | Fraction of net credit recycled to debit leg |
| `VRP_THRESHOLD` | `3.0` | Minimum VRP in vol points (normalized internally by IV level) |
| `MAX_DRAWDOWN_PCT` | `0.15` | Circuit breaker — realized DD halts immediately, unrealized requires 3 consecutive checks |
| `MARGIN_BUFFER_PCT` | `0.30` | Capital kept as margin buffer |

---

## Known limitations

- Backtester synthetic vol generator (OU process) is too smooth — it rarely triggers the stop loss. Run `--backtest` after accumulating real DB snapshots for meaningful results.
- Dashboard has not been updated to reflect all recent changes — use `--report` for production monitoring instead.
