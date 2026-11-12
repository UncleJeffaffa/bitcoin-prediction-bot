
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

SIGNAL_MINUTES = (7, 22, 37, 52)
HORIZON_MINUTES = 15

CURRENT_FIELDS = [
    "prediction_id", "created_at", "starting_price",
    "prediction", "up_probability", "down_probability",
    "target_at", "outcome_price", "outcome",
    "correct", "evaluated_at", "price_method"
]

PAPER_FIELDS = [
    "prediction_id", "created_at", "starting_price",
    "prediction", "mean_reversion_score", "rsi",
    "momentum_15m", "target_at", "outcome_price",
    "outcome", "correct", "evaluated_at", "price_method"
]


def get_json(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BitcoinPredictionBot/4.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_market_data():
    rows = get_json(
        f"{BASE_URL}/products/{PRODUCT}/candles?granularity=60"
    )

    rows = sorted(rows, key=lambda row: int(row[0]))

    candles = []
    for row in rows:
        if len(row) < 6:
            continue

        candles.append({
            "time": int(row[0]),
            "low": float(row[1]),
            "high": float(row[2]),
            "open": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5])
        })

    if len(candles) < 170:
        raise RuntimeError(
            f"Only {len(candles)} candles received from Coinbase."
        )

    return candles


def ema(values, period):
    if len(values) < period:
        return None

    alpha = 2.0 / (period + 1.0)
    result = values[0]

    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result

    return result


def rsi(values, period=14):
    if len(values) < period + 1:
        return None

    changes = [
        values[i] - values[i - 1]
        for i in range(len(values) - period, len(values))
    ]

    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0

    rs = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_features(candles):
    if len(candles) < 150:
        raise RuntimeError("Not enough completed candles for indicators.")

    window = candles[-150:]
    closes = [c["close"] for c in window]

    price = closes[-1]
    if price <= 0:
        raise RuntimeError("Invalid Bitcoin price.")

    def momentum(minutes):
        old = closes[-1 - minutes]
        return ((price / old) - 1.0) * 100.0 if old > 0 else 0.0

    mom5 = momentum(5)
    mom15 = momentum(15)
    mom30 = momentum(30)
    mom60 = momentum(60)

    ema9 = ema(closes[-100:], 9)
    ema21 = ema(closes[-100:], 21)
    ema50 = ema(closes[-150:], 50)
    current_rsi = rsi(closes, 14)

    if any(x is None for x in (ema9, ema21, ema50, current_rsi)):
        raise RuntimeError("Unable to calculate technical indicators.")

    returns = []
    for i in range(len(closes) - 30, len(closes)):
        previous = closes[i - 1]

        if previous <= 0:
            raise RuntimeError("Invalid price in volatility calculation.")

        returns.append((closes[i] / previous - 1.0) * 100.0)

    volatility = statistics.pstdev(returns)

    ema_difference = ((ema9 - ema21) / price) * 100.0
    long_trend = ((ema21 - ema50) / price) * 100.0

    current_score = (
        0.15 * math.tanh(mom5 / 0.10)
        + 0.20 * math.tanh(mom15 / 0.20)
        + 0.15 * math.tanh(mom30 / 0.30)
        + 0.10 * math.tanh(mom60 / 0.40)
        + 0.20 * math.tanh(ema_difference / 0.12)
        + 0.10 * math.tanh(long_trend / 0.20)
        + 0.10 * math.tanh((current_rsi - 50.0) / 12.0)
    )

    adjusted = current_score / (1.0 + volatility * 5.0)

    up_probability = round(
        50.0 + 30.0 * math.tanh(adjusted * 2.0), 1
    )

    mean_reversion_score = (
        -0.60 * math.tanh((current_rsi - 50.0) / 12.0)
        -0.40 * math.tanh(mom15 / 0.20)
    )

    return {
        "current_prediction": "UP" if up_probability >= 50.0 else "DOWN",
        "up_probability": up_probability,
        "down_probability": round(100.0 - up_probability, 1),
        "mean_reversion_prediction": (
            "UP" if mean_reversion_score > 0 else "DOWN"
        ),
        "mean_reversion_score": round(mean_reversion_score, 6),
        "rsi": round(current_rsi, 2),
        "momentum_15m": round(mom15, 5)
    }


def load_history(filename):
    if not os.path.exists(filename):
        return []

    with open(filename, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_history(filename, records, fields):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(records)


def evaluate_predictions(records, candles, now):
    evaluated = 0
    now_timestamp = now.timestamp()

    candle_by_time = {c["time"]: c for c in candles}

    for record in records:
        if record.get("outcome"):
            continue

        target = datetime.fromisoformat(
            record["target_at"]
        ).timestamp()

        method = record.get("price_method", "")

        if method == "scheduled_open":
            target_minute = int(target // 60)
            candle = candle_by_time.get(target_minute * 60)

            if candle is None or candle["time"] + 60 > now_timestamp:
                continue

            outcome_price = candle["open"]

        else:
            eligible = [
                c for c in candles
                if c["time"] >= target
                and c["time"] + 60 <= now_timestamp
            ]

            if not eligible:
                continue

            candle = min(eligible, key=lambda item: item["time"])
            outcome_price = candle["close"]

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
        r for r in records
        if r.get("correct") in ("True", "False")
    ]

    wins = sum(r["correct"] == "True" for r in completed)
    total = len(completed)

    print(f"\n{title}")
    print(f"Completed predictions: {total}")
    print(f"Correct: {wins}")
    print(f"Incorrect: {total - wins}")

    if total:
        print(f"Accuracy: {wins / total * 100.0:.2f}%")
    else:
        print("Accuracy: Waiting for results")


def get_scheduled_signal(now):
    current_minute = now.replace(second=0, microsecond=0)

    candidates = []

    for hour_offset in (0, -1):
        hour = current_minute.replace(
            minute=0
        ) + timedelta(hours=hour_offset)

        for minute in SIGNAL_MINUTES:
            candidate = hour + timedelta(minutes=minute)

            if candidate <= current_minute:
                candidates.append(candidate)

    if not candidates:
        return None

    signal_time = max(candidates)
    delay = (current_minute - signal_time).total_seconds()

    # Allow GitHub Actions to start late.
    # The most recent scheduled signal is still used,
    # never the current Bitcoin price as a replacement.
    if delay >= 15 * 60:
        return None

    return signal_time


def main():
    now = datetime.now(timezone.utc)

    print("=" * 60)
    print("BITCOIN 15-MINUTE PREDICTION BOT")
    print("SCHEDULED OPEN-TO-OPEN VERSION")
    print("=" * 60)
    print(f"\nUTC run time: {now.isoformat()}")

    candles = get_market_data()

    current_records = load_history(HISTORY_FILE)
    paper_records = load_history(PAPER_FILE)

    current_evaluated = evaluate_predictions(
        current_records, candles, now
    )

    paper_evaluated = evaluate_predictions(
        paper_records, candles, now
    )

    signal_time = get_scheduled_signal(now)

    if signal_time is None:
        print("\nNo scheduled signal is due.")
        print("Allowed UTC minutes: 07, 22, 37, 52.")
        print("Existing pending predictions were evaluated.")

    else:
        signal_timestamp = int(signal_time.timestamp())
        candle_by_time = {c["time"]: c for c in candles}

        entry_candle = candle_by_time.get(signal_timestamp)

        if entry_candle is None:
            print(
                "\nScheduled entry candle is not available yet. "
                "No prediction was created."
            )

        else:
            feature_candles = [
                c for c in candles
                if c["time"] + 60 <= signal_timestamp
            ]

            if len(feature_candles) < 150:
                raise RuntimeError(
                    "Not enough completed candles before entry time."
                )

            feature_candles = feature_candles[-150:]

            expected_times = [
                feature_candles[0]["time"] + i * 60
                for i in range(150)
            ]

            actual_times = [
                c["time"] for c in feature_candles
            ]

            if actual_times != expected_times:
                raise RuntimeError(
                    "Missing one-minute candles in indicator window. "
                    "Prediction skipped to avoid unreliable data."
                )

            if feature_candles[-1]["time"] + 60 != signal_timestamp:
                raise RuntimeError(
                    "Indicator window does not end immediately "
                    "before the scheduled entry."
                )

            indicators = calculate_features(feature_candles)

            entry_price = entry_candle["open"]
            target_time = signal_time + timedelta(
                minutes=HORIZON_MINUTES
            )

            prediction_id = signal_time.strftime("%Y%m%d%H%M")

            current_exists = any(
                r.get("prediction_id") == prediction_id
                for r in current_records
            )

            paper_exists = any(
                r.get("prediction_id") == prediction_id
                for r in paper_records
            )

            if not current_exists:
                current_records.append({
                    "prediction_id": prediction_id,
                    "created_at": signal_time.isoformat(),
                    "starting_price": round(entry_price, 2),
                    "prediction": indicators["current_prediction"],
                    "up_probability": indicators["up_probability"],
                    "down_probability": indicators["down_probability"],
                    "target_at": target_time.isoformat(),
                    "outcome_price": "",
                    "outcome": "",
                    "correct": "",
                    "evaluated_at": "",
                    "price_method": "scheduled_open"
                })

            if not paper_exists:
                paper_records.append({
                    "prediction_id": prediction_id,
                    "created_at": signal_time.isoformat(),
                    "starting_price": round(entry_price, 2),
                    "prediction": indicators["mean_reversion_prediction"],
                    "mean_reversion_score": indicators["mean_reversion_score"],
                    "rsi": indicators["rsi"],
                    "momentum_15m": indicators["momentum_15m"],
                    "target_at": target_time.isoformat(),
                    "outcome_price": "",
                    "outcome": "",
                    "correct": "",
                    "evaluated_at": "",
                    "price_method": "scheduled_open"
                })

            print(f"\nSIGNAL TIME: {signal_time.isoformat()}")
            print(f"ENTRY PRICE: ${entry_price:,.2f}")
            print(f"TARGET TIME: {target_time.isoformat()}")

            print("\nCURRENT BOT PREDICTION:")
            print(indicators["current_prediction"])
            print(f"UP probability: {indicators['up_probability']}%")
            print(f"DOWN probability: {indicators['down_probability']}%")

            print("\nMEAN-REVERSION PAPER PREDICTION:")
            print(indicators["mean_reversion_prediction"])
            print(f"Score: {indicators['mean_reversion_score']}")
            print(f"RSI: {indicators['rsi']}")
            print(f"15-minute momentum: {indicators['momentum_15m']}")

    save_history(HISTORY_FILE, current_records, CURRENT_FIELDS)
    save_history(PAPER_FILE, paper_records, PAPER_FIELDS)

    print(f"\nCurrent predictions evaluated: {current_evaluated}")
    print(f"Paper predictions evaluated: {paper_evaluated}")

    show_accuracy(current_records, "CURRENT BOT RESULTS")
    show_accuracy(paper_records, "MEAN-REVERSION PAPER RESULTS")

    print("\nPrediction files saved.")
    print("RUN COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nBOT ERROR: {error}")
        raise
