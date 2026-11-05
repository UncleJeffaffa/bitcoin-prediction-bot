import csv
import json
import math
import os
import urllib.request
from datetime import datetime, timezone, timedelta

HISTORY_FILE = "predictions.csv"
BASE_URL = "https://api.exchange.coinbase.com"
PRODUCT = "BTC-USD"


def get_json(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BitcoinPredictionBot/1.0"}
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

    if len(candles) < 40:
        raise RuntimeError("Not enough Bitcoin data")

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
    gains = [max(x, 0) for x in recent]
    losses = [abs(min(x, 0)) for x in recent]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def momentum(prices, minutes):
    old_price = prices[-(minutes + 1)]
    return ((prices[-1] - old_price) / old_price) * 100


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r", newline="") as file:
        return list(csv.DictReader(file))


def save_history(records):
    fields = [
        "prediction_id", "created_at", "starting_price",
        "prediction", "up_probability", "down_probability",
        "target_at", "outcome_price", "outcome",
        "correct", "evaluated_at"
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

    correct = sum(r["correct"] == "True" for r in completed)
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

    ema9 = calculate_ema(prices[-100:], 9)
    ema21 = calculate_ema(prices[-100:], 21)

    ema_difference = ((ema9 - ema21) / price) * 100

    score = (
        0.20 * (mom5 / 0.20)
        + 0.25 * (mom15 / 0.30)
        + 0.20 * (mom30 / 0.40)
        + 0.25 * (ema_difference / 0.15)
        + 0.10 * ((rsi - 50) / 10)
    )

    up_probability = round(
        50 + 30 * math.tanh(score / 2), 1
    )

    down_probability = round(100 - up_probability, 1)

    return {
        "price": round(price, 2),
        "prediction": "UP" if up_probability >= 50 else "DOWN",
        "up_probability": up_probability,
        "down_probability": down_probability,
        "rsi": round(rsi, 2),
        "momentum_5m": round(mom5, 4),
        "momentum_15m": round(mom15, 4),
        "momentum_30m": round(mom30, 4),
        "ema_9": round(ema9, 2),
        "ema_21": round(ema21, 2)
    }


def main():
    now = datetime.now(timezone.utc)

    print("=" * 45)
    print("BITCOIN 15-MINUTE PREDICTION BOT")
    print("=" * 45)

    print("\nTIME (UTC):")
    print(now.isoformat())

    price, candles, prices = get_market_data()
    records = load_history()

    evaluated = evaluate_predictions(records, candles, now)

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

    print("\nESTIMATED PROBABILITIES:")
    print(f"UP: {result['up_probability']}%")
    print(f"DOWN: {result['down_probability']}%")

    print("\nTECHNICAL INDICATORS:")
    print(f"RSI: {result['rsi']}")
    print(f"5-minute momentum: {result['momentum_5m']}%")
    print(f"15-minute momentum: {result['momentum_15m']}%")
    print(f"30-minute momentum: {result['momentum_30m']}%")
    print(f"EMA 9: ${result['ema_9']:,.2f}")
    print(f"EMA 21: ${result['ema_21']:,.2f}")

    print("\nNEW PREDICTION SAVED")
    print(f"Target time: {target_time.isoformat()}")

    print(f"\nPreviously evaluated: {evaluated}")

    show_accuracy(records)

    print("\nFULL RESULTS:")
    print(json.dumps(result, indent=4))

    print("\nPREDICTION COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"BOT ERROR: {error}")
        raise