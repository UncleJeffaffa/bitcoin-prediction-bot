import csv
import json
import math
import os
import urllib.request
from datetime import datetime, timezone, timedelta

# ==========================================
# BITCOIN 15-MINUTE PREDICTION BOT
# Prediction only - no automatic trading
# ==========================================

HISTORY_FILE = "predictions.csv"
PRODUCT = "BTC-USD"
BASE_URL = "https://api.exchange.coinbase.com"


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

    candles = sorted(candles, key=lambda candle: candle[0])

    if len(candles) < 40:
        raise RuntimeError("Not enough Bitcoin price data.")

    prices = [float(candle[4]) for candle in candles]

    return float(ticker["price"]), candles, prices


def calculate_ema(prices, period):
    multiplier = 2 / (period + 1)
    ema = prices[0]

    for price in prices[1:]:
        ema = (price * multiplier) + (
            ema * (1 - multiplier)
        )

    return ema


def calculate_rsi(prices, period=14):
    changes = [
        prices[i] - prices[i - 1]
        for i in range(1, len(prices))
    ]

    gains = [
        max(change, 0)
        for change in changes[-period:]
    ]

    losses = [
        abs(min(change, 0))
        for change in changes[-period:]
    ]

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100.0

    rs = average_gain / average_loss

    return 100 - (100 / (1 + rs))


def calculate_momentum(prices, minutes):
    old_price = prices[-(minutes + 1)]
    current_price = prices[-1]

    return ((current_price - old_price) / old_price) * 100


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
    now_epoch = now.timestamp()
    evaluated = 0

    for record in records:
        if record["outcome"] != "":
            continue

        target = datetime.fromisoformat(
            record["target_at"]
        ).timestamp()

        # Find the first completed one-minute candle
        # beginning at or after the 15-minute target.
        eligible = [
            candle for candle in candles
            if candle[0] >= target
            and candle[0] + 60 <= now_epoch
        ]

        if not eligible:
            continue

        target_candle = min(
            eligible,
            key=lambda candle: candle[0]
        )

        outcome_price = float(target_candle[4])
        starting_price = float(record["starting_price"])

        if outcome_price > starting_price:
            outcome = "UP"
        elif outcome_price < starting_price:
            outcome = "DOWN"
        else:
            outcome = "FLAT"

        if outcome == "FLAT":
            correct = "N/A"
        else:
            correct = str(
                record["prediction"] == outcome
            )

        record["outcome_price"] = round(
            outcome_price, 2
        )

        record["outcome"] = outcome
        record["correct"] = correct
        record["evaluated_at"] = now.isoformat()

        evaluated += 1

    return evaluated


def show_accuracy(records):
    completed = [
        record for record in records
        if record["correct"] in ("True", "False")
    ]

    correct = sum(
        record["correct"] == "True"
        for record in completed
    )

    total = len(completed)

    if total:
        accuracy = (correct / total) * 100
    else:
        accuracy = 0

    print("\nPREDICTION ACCURACY:")
    print(f"Completed predictions: {total}")
    print(f"Correct predictions:   {correct}")
    print(f"Incorrect predictions: {total - correct}")

    if total:
        print(f"Accuracy:              {accuracy:.1f}%")
    else:
        print("Accuracy:              Waiting for results")


def make_prediction(price, prices):
    rsi = calculate_rsi(prices)
    momentum_5 = calculate_momentum(prices, 5)
    momentum_15 = calculate_momentum(prices, 15)
    momentum_30 = calculate_momentum(prices, 30)

    ema_9 = calculate_ema(prices[-100:], 9)
    ema_21 = calculate_ema(prices[-100:], 21)

    ema_difference = (
        (ema_9 - ema_21) / price
    ) * 100

    # Combine technical indicators into a directional score.
    score = (
        0.20 * (momentum_5 / 0.20)
        + 0.25 * (momentum_15 / 0.30)
        + 0.20 * (momentum_30 / 0.40)
        + 0.25 * (ema_difference / 0.15)
        + 0.10 * ((rsi - 50) / 10)
    )

    # Convert score into a bounded model estimate.
    up_probability = 50 + (
        30 * math.tanh(score / 2)
    )

    up_probability = round(up_probability, 1)
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
        "momentum_5m": round(momentum_5, 4),
        "momentum_15m": round(momentum_15, 4),
        "momentum_30m": round(momentum_30, 4),
        "ema_9": round(ema_9, 2),
        "ema_21": round(ema_21, 2)
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

    # First, check predictions that have reached 15 minutes.
    evaluated = evaluate_predictions(
        records, candles, now
    )

    # Generate a new prediction.
    result = make_prediction(price, prices)

    target_time = now + timedelta(minutes=15)

    record = {
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
    }

    records.append(record)

    save_history(records)

    print("\nCURRENT BITCOIN PRICE:")
    print(f"${result['price']:,.2f}")

    print("\n15-MINUTE PREDICTION:")
    print(result["prediction"])

    print("\nESTIMATED PROBABILITIES:")
    print(f"UP:   {result['up_probability']}%")
    print(f"DOWN: {result['down_probability']}%")

    print("\nTECHNICAL INDICATORS:")
    print(f"RSI: {result['rsi']}")
    print(f"5-minute momentum: {result['momentum_5m']}%")
    print(f"15-minute momentum: {result['momentum_15m']}%")
    print(f"30-minute momentum: {result['momentum_30m']}%")
    print(f"EMA 9: ${result['ema_9']:,.2f}")
    print(f"EMA 21: ${result['ema_21']:,.2f}")

    print("\nNEW PREDICTION SAVED:")
    print(f"Target time: {target_time.isoformat()}")

    print(f"\nPreviously completed predictions: {evaluated}")

    show_accuracy(records)

    print("\nFULL RESULTS:")
    print(json.dumps(result, indent=4))

    print("\n" + "=" * 45)
    print("PREDICTION COMPLETE")
    print("=" * 45)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nBOT ERROR: {error}")
        raise