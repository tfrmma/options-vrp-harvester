"""
test_integration.py - Full integration test suite.
Tests all modules together without hitting the live API.
"""
import asyncio
import sys
sys.path.insert(0, '.')

async def test_paper_engine():
    from paper_trading.paper_engine import PaperTradingEngine
    from signals.signal_engine import Opportunity, StrategyType, VolRegime, Leg
    from risk.risk_engine import RiskDecision
    from data.vol_surface import VolSurface
    from core.derive_client import DeriveRESTClient

    rest = DeriveRESTClient()
    surface = VolSurface(rest)
    surface.spot = 3000.0

    engine = PaperTradingEngine(surface)

    def make_leg(name, direction, opt, strike, dte, iv, mid, bid, ask, delta):
        return Leg(
            instrument_name=name, direction=direction, option_type=opt,
            strike=strike, dte=dte, iv=iv, mid=mid, bid=bid, ask=ask,
            delta=delta, gamma=0.001, theta=-2.5, vega=0.04, spread_pct=0.05
        )

    # Use a future expiry date so parse_instrument returns positive DTE
    EXP = "20260617"   # ~10 DTE from test run date

    # IC with max_loss = $172 (USD-denominated, no spot multiply)
    # allocated = $2000 * 0.25 * 1.0 = $500 → 500/172 = 2.9 → floor = 2 contracts
    opp = Opportunity(
        strategy_type=StrategyType.IRON_CONDOR,
        legs=[
            make_leg(f"ETH-{EXP}-2600-P", "sell", "P", 2600, 10, 0.85, 15, 14, 16, -0.18),
            make_leg(f"ETH-{EXP}-2400-P", "buy",  "P", 2400, 10, 0.90, 8,  7,  9,  -0.08),
            make_leg(f"ETH-{EXP}-3400-C", "sell", "C", 3400, 10, 0.80, 14, 13, 15,  0.18),
            make_leg(f"ETH-{EXP}-3600-C", "buy",  "C", 3600, 10, 0.85, 7,  6,  8,   0.08),
        ],
        net_credit=0.028,
        net_delta=0.0,
        net_theta=0.006,
        net_vega=-0.04,
        max_profit=0.028,
        max_loss=172.0,   # USD, realistic for ETH IC
        score=68.5,
        vrp=5.2,
        regime=VolRegime.MEDIUM,
    )

    decision = RiskDecision(approved=True, reason="test", adjusted_size=1.0)
    pos_ids = await engine.open_position(opp, decision)

    assert len(pos_ids) == 4, f"Expected 4 legs, got {len(pos_ids)}"
    print(f"  Opened {len(pos_ids)} paper positions OK")

    # Verify sizing: max_loss=$172 → contracts=floor(500/172)=2
    first_pos = engine._positions[pos_ids[0]]
    assert float(first_pos["amount"]) == 2.0, \
        f"Expected 2 contracts, got {first_pos['amount']} (sizing bug: spot multiply?)"
    print(f"  Sizing correct: {first_pos['amount']:.0f} contracts (not inflated by spot)")

    # Verify Greeks update uses position_id key (not instrument_name)
    # Inject a fake surface with updated prices using the same future expiry
    fake_inst = f"ETH-{EXP}-2600-P"
    surface.surface = {
        EXP: {
            2600.0: {
                "put": {
                    "iv": 0.90, "mid": 18.0, "bid": 17.0, "ask": 19.0,
                    "spread_pct": 0.06, "delta": -0.20, "gamma": 0.0012,
                    "theta": -3.0, "vega": 0.05, "instrument_name": fake_inst, "dte": 9,
                }
            }
        }
    }
    await engine.update_position_greeks(surface)

    # Find the sell put position
    sell_put_pid = next(p for p in pos_ids if "2600-P" in p)
    updated = engine._positions.get(sell_put_pid)
    assert updated is not None, "Position not found in _open_positions after greeks update"
    assert updated["current_price"] == 18.0, \
        f"current_price not updated: {updated['current_price']} (key was wrong?)"
    assert updated["dte"] >= 1, f"DTE not updated or zero: {updated['dte']}"
    print(f"  Greeks update correct: price={updated['current_price']}, DTE={updated['dte']}")

    # Verify close uses self._surface as fallback (no surface arg needed)
    result = await engine.close_position(pos_ids[0], "test close")
    assert result is not None, "close_position returned None without surface arg"
    print(f"  Close with implicit surface fallback OK: PnL=${result['realized_pnl']:.4f}")

    print("Paper engine OK")
    await rest.close()

async def test_rv_estimator():
    from data.vol_surface import RVEstimator
    import time

    est = RVEstimator()
    base_ts = time.time() * 1000
    spot = 3000.0

    import numpy as np
    np.random.seed(99)
    for i in range(100):
        spot *= (1 + np.random.normal(0, 0.01))
        est.add_price(base_ts + i * 3600 * 1000, spot)

    rv = est.rv_annualized(24)
    assert rv is not None and 0.01 < rv < 5.0, f"RV out of range: {rv}"
    print(f"  RV 24h = {rv*100:.1f}%")

    ewma = est.ewma_rv()
    assert ewma is not None and 0.01 < ewma < 5.0
    print(f"  EWMA RV = {ewma*100:.1f}%")

    composite = est.composite_rv()
    assert composite is not None
    print(f"  Composite RV = {composite*100:.1f}%")
    print("RV estimator OK")

async def test_backtester():
    from utils.backtester import Backtester
    bt = Backtester(capital=2000, days=14)
    result = await bt.run()

    assert result.trades_opened >= 0
    assert len(result.equity_curve) > 0
    print(f"  Signals: {result.signals_fired}")
    print(f"  Trades:  {result.trades_opened}")
    print(f"  Win rate: {result.win_rate:.0%}")
    print(f"  Sharpe: {result.sharpe_ratio:.2f}")
    print("Backtester OK")

async def test_circuit_breakers():
    from risk.risk_engine import RiskEngine, PortfolioState

    engine = RiskEngine()

    # clean state -- no breach
    state = PortfolioState(net_delta=0.05, margin_available=1800, realized_pnl=0.0, unrealized_pnl=0.0)
    engine._peak_capital = 2000
    broken, _ = engine.check_circuit_breakers(state)
    assert not broken, "should not break at 0% drawdown"
    print("  No breaker at 0% drawdown OK")

    # realized DD >= threshold: halt immediately on first check
    state_realized = PortfolioState(net_delta=0.05, margin_available=1200, realized_pnl=-400.0, unrealized_pnl=0.0)
    engine._peak_capital = 2000
    broken, reason = engine.check_circuit_breakers(state_realized)
    assert broken and "REALIZED" in reason, f"expected realized DD halt, got: {reason}"
    print("  Realized DD halt OK")

    # unrealized DD: needs 3 consecutive checks, not just 1
    engine2 = RiskEngine()
    engine2._peak_capital = 2000
    state_unreal = PortfolioState(net_delta=0.05, margin_available=1200, realized_pnl=0.0, unrealized_pnl=-400.0)
    broken1, _ = engine2.check_circuit_breakers(state_unreal)
    assert not broken1, "should not halt on first unrealized DD breach"
    broken2, _ = engine2.check_circuit_breakers(state_unreal)
    assert not broken2, "should not halt on second unrealized DD breach"
    broken3, reason3 = engine2.check_circuit_breakers(state_unreal)
    assert broken3 and "TOTAL" in reason3, f"should halt on third consecutive breach, got: {reason3}"
    print("  Unrealized DD requires 3 consecutive checks OK")

    # excessive delta
    engine3 = RiskEngine()
    engine3._peak_capital = 2000
    state_delta = PortfolioState(net_delta=0.9, margin_available=1800, realized_pnl=0.0, unrealized_pnl=0.0)
    broken, reason = engine3.check_circuit_breakers(state_delta)
    assert broken and "DELTA" in reason
    print("  Breaker at excessive delta OK")

    print("Circuit breakers OK")

async def main():
    print("\n" + "="*50)
    print("  INTEGRATION TEST SUITE")
    print("="*50 + "\n")

    from db.database import init_db
    init_db()

    tests = [
        ("RV Estimator",     test_rv_estimator),
        ("Paper Engine",     test_paper_engine),
        ("Circuit Breakers", test_circuit_breakers),
        ("Backtester",       test_backtester),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"[{name}]")
        try:
            await test_fn()
            print(f"  PASS\n")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}\n")
            failed += 1

    print("="*50)
    print(f"  Results: {passed} passed / {failed} failed")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
