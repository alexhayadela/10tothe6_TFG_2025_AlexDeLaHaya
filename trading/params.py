# =========================
# IB CONNECTION
# =========================
IB_HOST = "127.0.0.1"
IB_PORT = 4002  # IB  | Paper trading
# IB_PORT = 4001 # IB  | Live trading
# IB_PORT = 7497 # TWS | Paper trading
# IB_PORT = 7496 # TWS | Live trading
CLIENT_ID = 1

# =========================
# MARKET SETTINGS
# =========================
MARKET = "SMART"
CURRENCY = "EUR"

# =========================
# PORTFOLIO RISK
# =========================
MAX_POSITION_PER_DAY = 0.1  # max % per day
# MAX_DAILY_LOSS = 500       # € stop trading if exceeded

STOP_LOSS_PCT = 0.03
# TAKE_PROFIT_PCT = 0.05

# =========================
# EXECUTION SETTINGS
# =========================
ORDER_TIMEOUT = 10  # seconds
RETRIES = 3

# =========================
# SCHEDULE (LOCAL TIME)
# =========================
MARKET_OPEN_HOUR = 9
MARKET_CLOSE_HOUR = 17

CLOSE_POSITIONS_BEFORE_MIN = 15
