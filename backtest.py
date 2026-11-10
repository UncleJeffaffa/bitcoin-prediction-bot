
import csv
import json
import math
import time
import requests
from datetime import datetime, timedelta, timezone

API = "https://api.exchange.coinbase.com"
PRODUCT = "BTC-USD"
DAYS = 30
TRAIN_DAYS = 7
HORIZON = 15
HEADERS = {"User-Agent": "BTC-15m-Backtest/1.0"}

def ema(values, period):
    if len(values) < period:
        return None
    alpha = 2 / (period + 1)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1 - alpha) * result
    return result

def rsi(values, period=14):
    changes = [values[i] - values[i-1] for i in range(1, len(values))]
    changes = changes[-period:]
    if len(changes) < period:
        return None
    gains = [max(x, 0) for x in changes]
    losses = [max(-x, 0) for x in changes]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100 - (100 / (1 + avg_gain / avg_loss))

def tanh(x):
    return math.tanh(x)

def download_candles():
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=DAYS)
    rows = []
    cursor = start

    print("Downloading historical BTC/USD candles...")

    while cursor < end:
        batch_end = min(cursor + timedelta(minutes=300), end)
        params = {
            "granularity": 60,
            "start": cursor.isoformat(),
            "end": batch_end.isoformat()
        }

        response = None

        for attempt in range(5):
            try:
                response = requests.get(
                    f"{API}/products/{PRODUCT}/candles",
                    params=params,
                    headers=HEADERS,
                    timeout=30
                )
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

        data = response.json()
        rows.extend(data)
        cursor = batch_end
        time.sleep(0.25)

    unique = {}
    for row in rows:
        if len(row) >= 6:
            unique[int(row[0])] = {
                "time": int(row[0]),
                "low": float(row[1]),
                "high": float(row[2]),
                "open": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5])
            }

    candles = sorted(unique.values(), key=lambda x: x["time"])
    cutoff = int(end.timestamp())

    # Exclude the currently forming candle.
    candles = [c for c in candles if c["time"] + 60 <= cutoff]

    if len(candles) < 10000:
        raise RuntimeError(
            f"Only {len(candles)} candles downloaded. "
            "Not enough data for this backtest."
        )
    gaps = [
        candles[i]["time"] - candles[i-1]["time"]
        for i in range(1, len(candles))
        if candles[i]["time"] - candles[i-1]["time"] != 60
    ]

    if gaps:
        raise RuntimeError(
            f"Historical data has {len(gaps)} missing one-minute intervals. "
            "Backtest stopped to avoid misleading results."
        )
    
    print(f"Downloaded {len(candles)} unique completed candles.")
    return candles

def features(candles, i):
    closes = [c["close"] for c in candles[:i+1]]
    highs = [c["high"] for c in candles[:i+1]]
    lows = [c["low"] for c in candles[:i+1]]
    price = closes[-1]

    if len(closes) < 150:
        return None

    def momentum(minutes):
        old = closes[-1-minutes]
        return (price / old - 1) * 100 if old else 0

    mom5 = momentum(5)
    mom15 = momentum(15)
    mom30 = momentum(30)
    mom60 = momentum(60)

    e9 = ema(closes[-100:], 9)
    e21 = ema(closes[-100:], 21)
    e50 = ema(closes[-150:], 50)
    if not all(x is not None for x in [e9, e21, e50]):
        return None

    r = rsi(closes, 14)
    changes = [
        (closes[j] / closes[j-1] - 1) * 100
        for j in range(len(closes)-30, len(closes))
        if closes[j-1] != 0
    ]
    mean_change = sum(changes) / len(changes)
    volatility = math.sqrt(
        sum((x - mean_change) ** 2 for x in changes) / len(changes)
    )

    ema_difference = ((e9 - e21) / price) * 100
    long_trend = ((e21 - e50) / price) * 100

    score = (
        0.15 * tanh(mom5 / 0.10)
        + 0.20 * tanh(mom15 / 0.20)
        + 0.15 * tanh(mom30 / 0.30)
        + 0.10 * tanh(mom60 / 0.40)
        + 0.20 * tanh(ema_difference / 0.12)
        + 0.10 * tanh(long_trend / 0.20)
        + 0.10 * tanh((r - 50) / 12)
    )

    adjusted = score / (1 + volatility * 5)
    current_bot_probability = round(
        50 + 30 * tanh(adjusted * 2), 1
    )

    # Alternative 1: EMA trend.
    ema_score = tanh(ema_difference / 0.12)
    if (price > e50 and ema_score < 0) or (
        price < e50 and ema_score > 0
    ):
        ema_score = 0

    # Alternative 2: RSI plus short-term momentum.
    rsi_score = tanh((r - 50) / 12)
    rsi_score = 0.6 * rsi_score + 0.4 * tanh(mom15 / 0.20)

    # Alternative 3: breakout of the previous 30 candles.
    prior_high = max(highs[-31:-1])
    prior_low = min(lows[-31:-1])
    breakout_score = 0
    if price > prior_high and mom5 > 0:
        breakout_score = 1
    elif price < prior_low and mom5 < 0:
        breakout_score = -1

    # Alternative 4: combined trend and momentum.
    combined_score = (
        0.35 * tanh(ema_difference / 0.12)
        + 0.25 * tanh(long_trend / 0.20)
        + 0.25 * tanh(mom15 / 0.20)
        + 0.15 * tanh((r - 50) / 12)
    )
    combined_score *= 1 / (1 + volatility * 5)

    return {
        "current": 1 if current_bot_probability >= 50 else -1,
        "ema": ema_score,
        "rsi": rsi_score,
        "breakout": breakout_score,
        "combined": combined_score
    }

def choose_threshold(training, strategy):
    # Select a threshold using training data only.
    best_threshold = 0
    best_accuracy = -1
    best_count = 0

    for threshold in [0, .05, .10, .15, .20, .25, .30,
                      .35, .40, .45, .50, .60, .70, .80]:
        selected = [
            x for x in training
            if abs(x[strategy]) >= threshold and x["direction"] != 0
        ]
        if len(selected) < 30:
            continue

        wins = sum(1 for x in selected if (1 if x[strategy] > 0 else -1) == x["direction"])
        accuracy = wins / len(selected)

        if (accuracy > best_accuracy or
            (accuracy == best_accuracy and len(selected) > best_count)):
            best_accuracy = accuracy
            best_threshold = threshold
            best_count = len(selected)

    return best_threshold

def main():
    candles = download_candles()

    first_time = candles[0]["time"]
    training_end = first_time + TRAIN_DAYS * 86400
    strategies = ["current", "ema", "rsi", "breakout", "combined"]

    records = []
    print("Calculating 15-minute predictions...")

    for i in range(150, len(candles) - HORIZON - 2):
        signal_time = candles[i]["time"] + 60

        # Match a bot scheduled every 15 minutes.
       if signal_time % 3600 not in (7 * 60, 22 * 60, 37 * 60, 52 * 60):
    continue

        f = features(candles, i)
        if f is None:
            continue

        # Entry is the next candle's open, not the signal candle close.
        entry_index = i + 1
        entry_price = candles[entry_index]["open"]
        entry_time = candles[entry_index]["time"]

        # Match the first candle beginning at or after 15 minutes.
        target_time = entry_time + HORIZON * 60
        outcome_index = next(
            (j for j in range(entry_index + 1, len(candles))
             if candles[j]["time"] >= target_time),
            None
        )
        if outcome_index is None:
            continue

        outcome_price = candles[outcome_index]["close"]
        if outcome_price == entry_price:
            direction = 0
        else:
            direction = 1 if outcome_price > entry_price else -1

        record = {
            "time": signal_time,
            "direction": direction,
            "entry": entry_price,
            "outcome": outcome_price
        }
        record.update(f)
        records.append(record)

    training = [x for x in records if x["time"] < training_end]
    testing = [x for x in records if x["time"] >= training_end]

    if len(training) < 100 or len(testing) < 100:
        raise RuntimeError(
            "Not enough training or out-of-sample predictions. "
            "Check the downloaded historical data."
        )

    thresholds = {"current": 0}
    for strategy in strategies[1:]:
        thresholds[strategy] = choose_threshold(training, strategy)

    print("\nTRAINING PERIOD:", len(training), "signals")
    print("OUT-OF-SAMPLE PERIOD:", len(testing), "signals")
    print("\nOUT-OF-SAMPLE RESULTS")
    print("Strategy,Predictions,Wins,Losses,Accuracy,Threshold")

    results = []
    for strategy in strategies:
        threshold = thresholds[strategy]
        selected = [
            x for x in testing
            if abs(x[strategy]) >= threshold
            and x["direction"] != 0
        ]

        wins = sum(
            1 for x in selected
            if (1 if x[strategy] > 0 else -1) == x["direction"]
        )
        losses = len(selected) - wins
        accuracy = 100 * wins / len(selected) if selected else 0

        result = {
            "strategy": strategy,
            "predictions": len(selected),
            "wins": wins,
            "losses": losses,
            "accuracy_percent": round(accuracy, 2),
            "training_threshold": threshold
        }
        results.append(result)

        print(
            f"{strategy},{len(selected)},{wins},{losses},"
            f"{accuracy:.2f}%,{threshold}"
        )

    baseline = next(x for x in results if x["strategy"] == "current")
    print("\nPERFORMANCE VS CURRENT BOT")
    for result in results:
        difference = result["accuracy_percent"] - baseline["accuracy_percent"]
        print(
            result["strategy"],
            f"{difference:+.2f} percentage points"
        )

    with open("backtest_results.json", "w") as file:
        json.dump({
            "product": PRODUCT,
            "days": DAYS,
            "training_days": TRAIN_DAYS,
            "out_of_sample_days": DAYS - TRAIN_DAYS,
            "candles": len(candles),
            "results": results
        }, file, indent=2)

    with open("backtest_predictions.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "time", "entry", "outcome", "direction",
                "current", "ema", "rsi", "breakout", "combined"
            ]
        )
        writer.writeheader()
        for record in testing:
            writer.writerow(record)

    print("\nSaved backtest_results.json and backtest_predictions.csv")

if __name__ == "__main__":
    main()
