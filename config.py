import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

def _env(key: str, default: str) -> str:
    return os.getenv(key, default)

def _float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))

def _int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))

@dataclass
class Config:
    # endpoints
    base_url: str = field(default_factory=lambda: _env("DERIVE_BASE_URL", "https://api-demo.lyra.finance"))
    ws_url:   str = field(default_factory=lambda: _env("DERIVE_WS_URL",   "wss://api-demo.lyra.finance/ws"))

    # auth - all three required for live
    wallet_private_key: str = field(default_factory=lambda: _env("WALLET_PRIVATE_KEY", ""))
    wallet_address:     str = field(default_factory=lambda: _env("WALLET_ADDRESS", ""))
    subaccount_id:      int = field(default_factory=lambda: _int("SUBACCOUNT_ID", 0))

    underlying: str  = field(default_factory=lambda: _env("UNDERLYING", "ETH"))
    mode:       str  = field(default_factory=lambda: _env("MODE", "paper"))
    log_level:  str  = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # capital
    total_capital_usd:   float = field(default_factory=lambda: _float("TOTAL_CAPITAL_USD", 2000.0))
    max_position_pct:    float = field(default_factory=lambda: _float("MAX_POSITION_PCT", 0.25))
    max_portfolio_delta: float = field(default_factory=lambda: _float("MAX_PORTFOLIO_DELTA", 0.25))
    max_drawdown_pct:    float = field(default_factory=lambda: _float("MAX_DRAWDOWN_PCT", 0.15))
    margin_buffer_pct:   float = field(default_factory=lambda: _float("MARGIN_BUFFER_PCT", 0.30))

    # credit leg (IC)
    credit_dte_min:     int   = field(default_factory=lambda: _int("CREDIT_DTE_MIN", 7))
    credit_dte_max:     int   = field(default_factory=lambda: _int("CREDIT_DTE_MAX", 14))
    credit_delta_target: float = field(default_factory=lambda: _float("CREDIT_DELTA_TARGET", 0.18))
    credit_close_pct:   float = field(default_factory=lambda: _float("CREDIT_CLOSE_PCT", 0.55))

    # debit leg (calendar)
    debit_dte_min:       int   = field(default_factory=lambda: _int("DEBIT_DTE_MIN", 30))
    debit_dte_max:       int   = field(default_factory=lambda: _int("DEBIT_DTE_MAX", 45))
    debit_premium_alloc: float = field(default_factory=lambda: _float("DEBIT_PREMIUM_ALLOC", 0.45))

    # signal
    vrp_threshold: float = field(default_factory=lambda: _float("VRP_THRESHOLD", 3.0))

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def is_paper(self) -> bool:
        return self.mode == "paper"

    def validate(self) -> None:
        if self.is_live:
            assert self.wallet_private_key, "WALLET_PRIVATE_KEY required for live"
            assert self.wallet_address,     "WALLET_ADDRESS required for live"
            assert self.subaccount_id > 0,  "SUBACCOUNT_ID required for live"
        assert self.underlying in ("ETH", "BTC"), f"unsupported underlying: {self.underlying}"
        assert self.total_capital_usd > 0, "TOTAL_CAPITAL_USD must be positive"

cfg = Config()
