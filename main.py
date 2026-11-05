
import json
import math
import statistics
import urllib.request
from datetime import datetime, timezone


# ==========================================
# BITCOIN 15-MINUTE PREDICTION BOT
# ==========================================

PRODUCT = "BTC-USD"
API_URL = (
    "https://api.exchange.coinbase.com/products/"
    + PRODUCT
    + "/candles?granularity=60"
)


def get_bitcoin_data():
    """Download public Bitcoin 1-minute candles."""

    request = urllib.request.Request(
        API_URL,
        headers={"User-Agent": "BitcoinPredictionBot/1.0"}
    )

    with urllib.request.urlopen(request, timeout=15) as response:
        data = json.loads(response.read().decode())

    if not isinstance(data, list) or len(data) < 50:
        raise ValueError("Not enough Bitcoin price data received.")

    # Coinbase candle format:
    # [time, low, high, open, close, volume]

    data.sort(key=lambda candle: candle[0])

    # Remove the potentially incomplete current candle.
    data = data[:-1]

    closes = [float(candle[4]) for candle in data]

    if len(closes) < 40:
        raise ValueError("Not enough completed candles.")

    return closes


def calculate_rsi(prices, period=14):
    changes = [
        prices[i] - prices[i - 1]
        for i in range(1, len(prices))
    ]

    recent = changes[-period:]

    gains = [max(change, 0) for change in recent]
    losses = [max(-change, 0) for change in recent]

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0

    rs = average_gain / average_loss

    return 100 - (100 / (1 + rs))


def calculate_ema(prices, period):
    multiplier = 2 / (period + 1)

    ema = prices[-period]

    for price in prices[-period + 1:]:
        ema = (price * multiplier) + (
            ema * (1 - multiplier)
        )

    return ema


def calculate_momentum(prices, minutes):
    return (
        prices[-1] / prices[-minutes - 1] - 1
    ) * 100


def predict_bitcoin(prices):

    current_price = prices[-1]

    # Recent price momentum
    momentum_5 = calculate_momentum(prices, 5)
    momentum_15 = calculate_momentum(prices, 15)
    momentum_30 = calculate_momentum(prices, 30)

    # Technical indicators
    rsi = calculate_rsi(prices)
    ema_9 = calculate_ema(prices, 9)
    ema_21 = calculate_ema(prices, 21)

    # One-minute returns
    returns = [
        (prices[i] / prices[i - 1]) - 1
        for i in range(1, len(prices))
    ]

    volatility = statistics.stdev(returns[-30:])

    if volatility < 0.000001:
        volatility = 0.000001

    # Normalize momentum using recent volatility
    z5 = (momentum_5 / 100) / (
        volatility * math.sqrt(5)
    )

    z15 = (momentum_15 / 100) / (
        volatility * math.sqrt(15)
    )

    z30 = (momentum_30 / 100) / (
        volatility * math.sqrt(30)
    )

    trend = (ema_9 - ema_21) / current_price

    trend_score = trend / (
        volatility * math.sqrt(12)
    )

    # RSI adjustment
    rsi_score = (50 - rsi) / 25

    # Weighted prediction score
    score = (
        0.50 * z15
        + 0.25 * z5
        + 0.15 * z30
        + 0.10 * trend_score
        + 0.08 * rsi_score
    )

    # Prevent extreme probability estimates
    score = max(-2.5, min(2.5, score))

    # Convert score into estimated probability
    up_probability = (
        1 / (1 + math.exp(-1.15 * score))
    )

    down_probability = 1 - up_probability

    # Prediction
    if up_probability >= 0.55:
        prediction = "UP"
    elif down_probability >= 0.55:
        prediction = "DOWN"
    else:
        prediction = "UNCERTAIN"

    return {
        "price": round(current_price, 2),
        "prediction": prediction,
        "up_probability": round(
            up_probability * 100, 2
        ),
        "down_probability": round(
            down_probability * 100, 2
        ),
        "rsi": round(rsi, 2),
        "momentum_5m": round(momentum_5, 4),
        "momentum_15m": round(momentum_15, 4),
        "momentum_30m": round(momentum_30, 4),
        "ema_9": round(ema_9, 2),
        "ema_21": round(ema_21, 2),
        "volatility": round(volatility * 100, 5)
    }


def main():

    print("=" * 45)
    print("BITCOIN 15-MINUTE PREDICTION BOT")
    print("=" * 45)

    try:
        prices = get_bitcoin_data()

        result = predict_bitcoin(prices)

        print("\nTIME (UTC):")
        print(datetime.now(timezone.utc).isoformat())

        print("\nCURRENT BITCOIN PRICE:")
        print(f"${result['price']:,.2f}")

        print("\n15-MINUTE PREDICTION:")
        print(result["prediction"])

        print("\nESTIMATED PROBABILITIES:")
        print(
            f"UP:   {result['up_probability']}%"
        )
        print(
            f"DOWN: {result['down_probability']}%"
        )

        print("\nTECHNICAL INDICATORS:")
        print(f"RSI: {result['rsi']}")
        print(f"5-minute momentum: {result['momentum_5m']}%")
        print(f"15-minute momentum: {result['momentum_15m']}%")
        print(f"30-minute momentum: {result['momentum_30m']}%")
        print(f"EMA 9: ${result['ema_9']:,.2f}")
        print(f"EMA 21: ${result['ema_21']:,.2f}")

        print("\nFULL RESULTS:")
        print(json.dumps(result, indent=4))

        print("\n" + "=" * 45)
        print("PREDICTION COMPLETE")
        print("=" * 45)

    except Exception as error:
        print(f"\nERROR: {error}")
        raise


if __name__ == "__main__":
    main()
