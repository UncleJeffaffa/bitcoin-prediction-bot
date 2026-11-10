
import csv
import json
import math
import os
import statistics
import urllib.request
from datetime import datetime, timezone, timedelta

BASE_URL = "https://api.exchange.coinbase.com"
PRODUCT = "BTC-USD"

HISTORY_FILE = "predictions.csv"
PAPER_FILE = "paper_mean_reversion.csv"

CURRENT_FIELDS = [
    "prediction_id", "created_at", "starting_price",
    "prediction", "up_probability", "down_probability",
    "target_at", "outcome_price", "outcome",
    "correct", "evaluated_at"
]

PAPER_FIELDS = [
    "prediction_id", "created_at", "starting_price",
    "prediction", "mean_reversion_score", "rsi",
    "momentum_15m", "target_at", "outcome_price",
    "outcome", "correct", "evaluated_at"
]


def get_json(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BitcoinPredictionBot/3.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_market_data():
    candles = get_json(
        f"{BASE_URL}/products/{PRODUCT}/candles?granularity=60"
    )

    candles = sorted(candles, key=lambda x: x[0])
    now = datetime.now(timezone.utc).timestamp()

    candles = [
        c for c in candles
        if c[0] + 60 <= now
    ]

    if len(candles) < 150:
        raise RuntimeError("Not enough completed Bitcoin candles.")

    prices = [float(c[4]) for c in candles]

    return candles, prices


def calculate_ema(prices, period):
    if len(prices) < period:
        raise RuntimeError("Not enough prices for EMA.")

    multiplier = 2 / (period + 1)
    value = prices[0]

    for price in prices[1:]:
        value = price * multiplier + value * (1 - multiplier)

    return value


def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        raise RuntimeError("Not enough prices for RSI.")

    changes = [
        prices[i] - prices[i - 1]
        for i in range(len(prices) - period, len(prices))
    ]

    gains = [max(change, 0) for change in changes]
    losses = [abs(min(change, 0)) for change in changes]

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0

    rs = average_gain / average_loss
    return 100 - (100 / (1 + rs))


def momentum(prices, minutes):
    if len(prices) <= minutes:
        return 0.0

    old_price = prices[-(minutes + 1)]
    new_price = prices[-1]

    return ((new_price / old_price) - 1) * 100


def calculate_volatility(prices, period=30):
    changes = [
        (prices[i] / prices[i - 1] - 1) * 100
        for i in range(len(prices) - period, len(prices))
    ]

    return statistics.pstdev(changes)


def load_history(filename):
    if not os.path.exists(filename):
        return []

    with open(filename, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_history(filename, records, fields):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def evaluate_predictions(records, candles, now):
    evaluated = 0
    current_time = now.timestamp()

    for record in records:
        if record.get("outcome"):
            continue

        target = datetime.fromisoformat(
            record["target_at"]
        ).timestamp()

        eligible = [
            candle for candle in candles
            if candle[0] >= target
            and candle[0] + 60 <= current_time
        ]

        if not eligible:
            continue

        candle = min(eligible, key=lambda item: item[0])
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


def show_accuracy(records, title):
    completed = [
        record for record in records
        if record.get("correct") in ("True", "False")
    ]

    wins = sum(
        record["correct"] == "True"
        for record in completed
    )

    total = len(completed)

    print(f"\n{title}")
    print(f"Completed predictions: {total}")
    print(f"Correct: {wins}")
    print(f"Incorrect: {total - wins}")

    if total:
        print(f"Accuracy: {wins / total * 100:.2f}%")
    else:
        print("Accuracy: Waiting for results")


def make_current_prediction(price, prices):
    rsi_value = calculate_rsi(prices)

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

    score = (
        0.15 * math.tanh(mom5 / 0.10)
        + 0.20 * math.tanh(mom15 / 0.20)
        + 0.15 * math.tanh(mom30 / 0.30)
        + 0.10 * math.tanh(mom60 / 0.40)
        + 0.20 * math.tanh(ema_difference / 0.12)
        + 0.10 * math.tanh(long_trend / 0.20)
        + 0.10 * math.tanh((rsi_value - 50) / 12)
    )

    adjusted_score = score / (1 + volatility * 5)

    up_probability = round(
        50 + 30 * math.tanh(adjusted_score * 2),
        1
    )

    return {
        "prediction": "UP" if up_probability >= 50 else "DOWN",
        "up_probability": up_probability,
        "down_probability": round(100 - up_probability, 1)
    }


def make_mean_reversion_prediction(prices):
    rsi_value = calculate_rsi(prices)
    mom15 = momentum(prices, 15)

    # Same mean-reversion scoring formula used in the backtest.
    score = (
        -0.60 * math.tanh((rsi_value - 50) / 12)
        -0.40 * math.tanh(mom15 / 0.20)
    )

    prediction = "UP" if score > 0 else "DOWN"

    return {
        "prediction": prediction,
        "score": round(score, 6),
        "rsi": round(rsi_value, 2),
        "momentum_15m": round(mom15, 5)
    }


def main():
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    print("=" * 55)
    print("BITCOIN 15-MINUTE PREDICTION BOT")
    print("CURRENT STRATEGY + MEAN-REVERSION PAPER TEST")
    print("=" * 55)

    print(f"\nUTC time: {now.isoformat()}")

    candles, prices = get_market_data()

    # Use the most recently completed candle close as the entry price.
    price = float(candles[-1][4])

    current_records = load_history(HISTORY_FILE)
    paper_records = load_history(PAPER_FILE)

    current_evaluated = evaluate_predictions(
        current_records, candles, now
    )

    paper_evaluated = evaluate_predictions(
        paper_records, candles, now
    )

    # Keep the target aligned to the scheduled 15-minute signal.
    target_time = now + timedelta(minutes=15)

    prediction_id = now.strftime("%Y%m%d%H%M")

    # Avoid duplicate entries if the workflow is manually rerun
    # during the same UTC minute.
    current_exists = any(
        record["prediction_id"] == prediction_id
        for record in current_records
    )

    paper_exists = any(
        record["prediction_id"] == prediction_id
        for record in paper_records
    )

    current = make_current_prediction(price, prices)
    mean_reversion = make_mean_reversion_prediction(prices)

    if not current_exists:
        current_records.append({
            "prediction_id": prediction_id,
            "created_at": now.isoformat(),
            "starting_price": round(price, 2),
            "prediction": current["prediction"],
            "up_probability": current["up_probability"],
            "down_probability": current["down_probability"],
            "target_at": target_time.isoformat(),
            "outcome_price": "",
            "outcome": "",
            "correct": "",
            "evaluated_at": ""
        })

    if not paper_exists:
        paper_records.append({
            "prediction_id": prediction_id,
            "created_at": now.isoformat(),
            "starting_price": round(price, 2),
            "prediction": mean_reversion["prediction"],
            "mean_reversion_score": mean_reversion["score"],
            "rsi": mean_reversion["rsi"],
            "momentum_15m": mean_reversion["momentum_15m"],
            "target_at": target_time.isoformat(),
            "outcome_price": "",
            "outcome": "",
            "correct": "",
            "evaluated_at": ""
        })

    save_history(
        HISTORY_FILE, current_records, CURRENT_FIELDS
    )

    save_history(
        PAPER_FILE, paper_records, PAPER_FIELDS
    )

    print(f"\nBITCOIN PRICE: ${price:,.2f}")

    print("\nCURRENT BOT PREDICTION:")
    print(current["prediction"])
    print(f"UP score: {current['up_probability']}%")
    print(f"DOWN score: {current['down_probability']}%")

    print("\nMEAN-REVERSION PAPER PREDICTION:")
    print(mean_reversion["prediction"])
    print(f"Mean-reversion score: {mean_reversion['score']}")
    print(f"RSI: {mean_reversion['rsi']}")
    print(f"15-minute momentum: {mean_reversion['momentum_15m']}")

    print(f"\nTarget time: {target_time.isoformat()}")

    print(f"\nCurrent predictions evaluated: {current_evaluated}")
    print(f"Paper predictions evaluated: {paper_evaluated}")

    show_accuracy(current_records, "CURRENT BOT RESULTS")
    show_accuracy(paper_records, "MEAN-REVERSION PAPER RESULTS")

    print("\nBoth prediction files saved.")
    print("PREDICTION RUN COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"BOT ERROR: {error}")
        raise
