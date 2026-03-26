import asyncio
from trading.execute import IBClient

async def test_all():
    client = IBClient()
    await client.connect()

    # Balance
    balance = await client.get_total_balance()
    print(f"Balance: {balance}")

    # Posiciones
    positions = client.get_positions()
    print(f"Positions: {positions}")

asyncio.run(test_all())