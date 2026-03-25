import asyncio
import json
import subprocess
import sys
import datetime
from math import floor
from typing import Literal

from ib_async import IB, Stock, MarketOrder

import params
from config import DATA_PATH


class IBClient:
    def __init__(self):
        self.ib = IB()

    async def connect(self):
        await self.ib.connectAsync(
            params.IB_HOST,
            params.IB_PORT,
            clientId=params.CLIENT_ID
        )
        print("Connected to IB")

    def get_positions(self):
        return self.ib.positions()

    async def place_market_order(self, symbol, qty):
        contract = Stock(symbol, params.MARKET, params.CURRENCY)
        order_type = "BUY" if qty > 0 else "SELL"
        order = MarketOrder(order_type, abs(qty), stopLoss=params.STOP_LOSS_PCT)
        trade = self.ib.placeOrder(contract, order)

        start = asyncio.get_event_loop().time()
        while not trade.isDone():
            if asyncio.get_event_loop().time() - start > params.ORDER_TIMEOUT:
                raise TimeoutError(f"Order timeout for {symbol}")
            await asyncio.sleep(0.5)

        print(f"{symbol} {order_type} filled: {trade.orderStatus.status}")
        return trade
    
    async def get_total_balance(self) -> float:
            """
            Returns the total available cash for trading in EUR (or currency in params).
            """
            account_values = await self.ib.accountValues()
            # Available Funds: BuyingPower or CashBalance
            for av in account_values:
                if av.tag == "AvailableFunds" and av.currency == params.CURRENCY:
                    return float(av.value)
            print("AvailableFunds not found, returning 0")
            return 0.0

def git_pull():
    print("Pulling latest repo...")
    subprocess.run(["git", "pull"], check=False)


def load_signals(top_k: int = 3) -> tuple[bool, dict]:
    """Load today's signals, normalize, return dict {symbol: size}."""
    path = DATA_PATH / "predictions.json"
    today_str = datetime.date.today().isoformat()

    with open(path, "r") as f:
        raw_signals = json.load(f)

    today_signals = [s for s in raw_signals if s["date"] == today_str]
    if not today_signals:
        return False, {}

    today_signals.sort(key=lambda x: x["proba"], reverse=True)
    top_signals = today_signals[:top_k]

    # Convert pred to positive/negative size (later converted to buy or sell)
    signals = {}
    for s in top_signals:
        symbol = s["ticker"].replace(".MC", "")
        signals[symbol] = 1 if s["pred"] == 1 else -1

    # Normalize absolute sizes 
    total_abs = sum(abs(v) for v in signals.values())
    signals = {k: v / total_abs for k, v in signals.items()}

    return True, signals


async def execute_signals(client: IBClient, signals, total_balance):
    print("Executing signals...")

    for symbol, size in signals.items():
        if size != 0:
            # Get current market price
            contract = Stock(symbol, params.MARKET, params.CURRENCY)
            market_price = await client.ib.marketPrice(contract)
            
            # Calculate qty in shares
            euro_amount = size * params.MAX_POSITION_PER_DAY * total_balance
            qty = floor(abs(euro_amount) / market_price)
            
            if qty == 0:
                print(f"{symbol}: calculated qty=0, skipping")
                continue

            # Apply buy/sell sign
            qty = qty if size > 0 else -qty

            await client.place_market_order(symbol, qty)


async def close_positions(client: IBClient, profit: bool):
    print("Checking positions to close...")

    for p in client.get_positions():
        pnl = p.unrealizedPNL
        qty = p.position

        if qty != 0 and (not profit or pnl > 0):
            print(f"Closing {p.contract.symbol} | PnL: {pnl}")
            await client.place_market_order(p.contract.symbol, -qty)


def is_market_open():
    now = datetime.datetime.now()
    return params.MARKET_OPEN_HOUR <= now.hour < params.MARKET_CLOSE_HOUR


async def bot(action: Literal["open", "close"]):
    client = IBClient()
    await client.connect()

    total_balance = await client.get_total_balance()
    print(f"Total available balance: {total_balance} {params.CURRENCY}")

    if is_market_open():
        if action == "open":
            git_pull()
            updated, signals = load_signals()
            if updated:
                print("Executing signals.")
                await execute_signals(client, signals, total_balance)
            else:
                print("No valid signals for today.")
        elif action == "close":
            print("Closing profitable positions.")
            await close_positions(client, profit=True)
        else:
            print("Action must be 'open' or 'close'.")
    else:
        print("Market is closed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python execute.py [open|close]")
    else:
        asyncio.run(bot(sys.argv[1]))