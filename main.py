
import csv
import json
import math
import os
import statistics
import urllib.request
from datetime import datetime, timezone, timedelta

HISTORY_FILE = "predictions.csv"
BASE_URL = "https://api.exchange.coinbase.com"
PRODUCT = "BTC-USD"


def get_json(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BitcoinPredictionBot/2.0"}
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode())


def get_market_data():
    ticker = get_json(
        f"{BASE_URL}/products/{PRODUCT}/ticker"
    )

    candles = get_json(
        f"{BASE_URL}/products/{PRODUCT}/candles?granularity=60"
    )

    candles = sorted(candles, key=lambda x: x[0])

    now = datetime.now(timezone.utc).timestamp()

    # Remove unfinished candles
    candles = [
        c for c in candles
        if c[0] + 60 <= now
    ]

    if len(candles) < 100:
        raise RuntimeError("Not enough completed Bitcoin candles")

    prices = [float(c[4]) for c in candles]

    return float(ticker["price"]), candles, prices


def calculate_ema(prices, period):
    multiplier = 2 / (period + 1)
    ema = prices[0]

    for price in prices[1:]:
        ema = price * multiplier + ema * (1 - multiplier)

    return ema


def calculate_rsi(prices, period=14):
    changes = [
        prices[i] - prices[i - 1]
        for i in range(1, len(prices))
    ]

    recent = changes[-period:]

    gains = [max(change, 0) for change in recent]
    losses = [abs(min(change, 0)) for change in recent]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def momentum(prices, minutes):
    if len(prices) <= minutes:
        return 0.0

    old_price = prices[-(minutes + 1)]
    new_price = prices[-1]

    return ((new_price - old_price) / old_price) * 100


def calculate_volatility(prices, period=30):
    changes = [
        (prices[i] - prices[i - 1]) / prices[i - 1] * 100
        for i in range(len(prices) - period, len(prices))
    ]

    return statistics.pstdev(changes)


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r", newline="") as file:
        return list(csv.DictReader(file))


def save_history(records):
    fields = [
        "prediction_id",
        "created_at",
        "starting_price",
        "prediction",
        "up_probability",
        "down_probability",
        "target_at",
        "outcome_price",
        "outcome",
        "correct",
        "evaluated_at"
    ]

    with open(HISTORY_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def evaluate_predictions(records, candles, now):
    evaluated = 0
    current_time = now.timestamp()

    for record in records:
        if record["outcome"]:
            continue

        target = datetime.fromisoformat(
            record["target_at"]
        ).timestamp()

        eligible = [
            c for c in candles
            if c[0] >= target and c[0] + 60 <= current_time
        ]

        if not eligible:
            continue

        candle = min(eligible, key=lambda c: c[0])

        outcome_price = float(candle[4])
        starting_price = float(record["starting_price"])

        if outcome_price > starting_price:
            outcome = "UP"
        elif outcome_price < starting_price:
            outcome = "DOWN"
        else:
            outcome = "FLAT"

        correct = (
            "N/A" if outcome == "FLAT"
            else str(record["prediction"] == outcome)
        )

        record["outcome_price"] = round(outcome_price, 2)
        record["outcome"] = outcome
        record["correct"] = correct
        record["evaluated_at"] = now.isoformat()

        evaluated += 1

    return evaluated


def show_accuracy(records):
    completed = [
        r for r in records
        if r["correct"] in ("True", "False")
    ]

    correct = sum(
        r["correct"] == "True"
        for r in completed
    )

    total = len(completed)

    print("\nPREDICTION ACCURACY")
    print(f"Completed: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")

    if total:
        print(f"Accuracy: {correct / total * 100:.1f}%")
    else:
        print("Accuracy: Waiting for results")


def make_prediction(price, prices):
    rsi = calculate_rsi(prices)

    mom5 = momentum(prices, 5)
    mom15 = momentum(prices, 15)
    mom30 = momentum(prices, 30)
    mom60 = momentum(prices, 60)

    ema9 = calculate_ema(prices[-100:], 9)
    ema21 = calculate_ema(prices[-100:], 21)
    ema50 = calculate_ema(prices[-150:], 50)

    volatility = calculate_volatility(prices)

    ema_difference = ((ema9 - ema21) / price) * 100
    long_trend = ((ema21 - ema50) / price) * 100

    # Normalize market indicators
    score = (
        0.15 * math.tanh(mom5 / 0.10)
        + 0.20 * math.tanh(mom15 / 0.20)
        + 0.15 * math.tanh(mom30 / 0.30)
        + 0.10 * math.tanh(mom60 / 0.40)
        + 0.20 * math.tanh(ema_difference / 0.12)
        + 0.10 * math.tanh(long_trend / 0.20)
        + 0.10 * math.tanh((rsi - 50) / 12)
    )

    # Reduce confidence when market volatility is unusually high
    volatility_factor = 1 / (1 + volatility * 5)

    adjusted_score = score * volatility_factor

    up_probability = round(
        50 + 30 * math.tanh(adjusted_score * 2),
        1
    )

    down_probability = round(100 - up_probability, 1)

    prediction = (
        "UP" if up_probability >= 50 else "DOWN"
    )

    return {
        "price": round(price, 2),
        "prediction": prediction,
        "up_probability": up_probability,
        "down_probability": down_probability,
        "rsi": round(rsi, 2),
        "momentum_5m": round(mom5, 4),
        "momentum_15m": round(mom15, 4),
        "momentum_30m": round(mom30, 4),
        "momentum_60m": round(mom60, 4),
        "ema_9": round(ema9, 2),
        "ema_21": round(ema21, 2),
        "ema_50": round(ema50, 2),
        "volatility": round(volatility, 5)
    }


def main():
    now = datetime.now(timezone.utc)

    print("=" * 45)
    print("BITCOIN 15-MINUTE PREDICTION BOT V2")
    print("=" * 45)

    print("\nTIME (UTC):")
    print(now.isoformat())

    price, candles, prices = get_market_data()

    records = load_history()

    evaluated = evaluate_predictions(
        records, candles, now
    )

    result = make_prediction(price, prices)

    target_time = now + timedelta(minutes=15)

    records.append({
        "prediction_id": now.strftime("%Y%m%d%H%M%S"),
        "created_at": now.isoformat(),
        "starting_price": result["price"],
        "prediction": result["prediction"],
        "up_probability": result["up_probability"],
        "down_probability": result["down_probability"],
        "target_at": target_time.isoformat(),
        "outcome_price": "",
        "outcome": "",
        "correct": "",
        "evaluated_at": ""
    })

    save_history(records)

    print("\nCURRENT BITCOIN PRICE:")
    print(f"${result['price']:,.2f}")

    print("\n15-MINUTE PREDICTION:")
    print(result["prediction"])

    print("\nESTIMATED DIRECTIONAL SCORES:")
    print(f"UP: {result['up_probability']}%")
    print(f"DOWN: {result['down_probability']}%")

    print("\nTECHNICAL INDICATORS:")

    for key in [
        "rsi",
        "momentum_5m",
        "momentum_15m",
        "momentum_30m",
        "momentum_60m",
        "ema_9",
        "ema_21",
        "ema_50",
        "volatility"
    ]:
        print(f"{key}: {result[key]}")

    print("\nNEW PREDICTION SAVED")
    print(f"Target time: {target_time.isoformat()}")

    print(f"\nPreviously evaluated: {evaluated}")

    show_accuracy(records)

    print("\nPREDICTION COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"BOT ERROR: {error}")
        raise
