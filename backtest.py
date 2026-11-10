
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
MAX_FILLED_GAP_MINUTES = 5

HEADERS = {
    "User-Agent": "BTC-15m-Historical-Backtest/2.0"
}


def ema(values, period):
    if len(values) < period:
        return None

    alpha = 2 / (period + 1)
    result = values[0]

    for value in values[1:]:
        result = alpha * value + (1 - alpha) * result

    return result


def rsi(values, period=14):
    changes = [
        values[i] - values[i - 1]
        for i in range(1, len(values))
    ]

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


def repair_short_gaps(candles):
    """
    Fill short missing intervals with synthetic flat candles.

    Synthetic candles use the previous close for OHLC
    and zero volume. Gaps longer than the configured limit
    are rejected rather than silently filled.
    """

    if not candles:
        raise RuntimeError("No historical candles were downloaded.")

    repaired = [candles[0]]
    filled_count = 0
    gap_count = 0

    for current in candles[1:]:
        previous = repaired[-1]

        difference = current["time"] - previous["time"]

        if difference <= 0:
            continue

        if difference % 60 != 0:
            raise RuntimeError(
                "Historical candle timestamps are not aligned "
                "to one-minute intervals."
            )

        missing_minutes = difference // 60 - 1

        if missing_minutes > 0:
            gap_count += 1

            if missing_minutes > MAX_FILLED_GAP_MINUTES:
                raise RuntimeError(
                    f"Found a gap of {missing_minutes} minutes. "
                    "This is too large to reconstruct safely."
                )

            previous_close = previous["close"]

            for minute in range(1, missing_minutes + 1):
                synthetic_time = previous["time"] + minute * 60

                repaired.append({
                    "time": synthetic_time,
                    "open": previous_close,
                    "high": previous_close,
                    "low": previous_close,
                    "close": previous_close,
                    "volume": 0.0,
                    "synthetic": True
                })

                filled_count += 1

        repaired.append(current)

    print(f"Missing intervals found: {gap_count}")
    print(f"Synthetic candles inserted: {filled_count}")

    return repaired, filled_count


def download_candles():
    end = datetime.now(timezone.utc).replace(
        second=0,
        microsecond=0
    )

    start = end - timedelta(days=DAYS)

    rows = []
    cursor = start

    print("Downloading historical BTC/USD candles...")

    while cursor < end:
        batch_end = min(
            cursor + timedelta(minutes=300),
            end
        )

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
            timestamp = int(row[0])

            unique[timestamp] = {
                "time": timestamp,
                "low": float(row[1]),
                "high": float(row[2]),
                "open": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "synthetic": False
            }

    candles = sorted(
        unique.values(),
        key=lambda x: x["time"]
    )

    cutoff = int(end.timestamp())

    # Exclude the currently forming candle.
    candles = [
        c for c in candles
        if c["time"] + 60 <= cutoff
    ]

    if len(candles) < 10000:
        raise RuntimeError(
            f"Only {len(candles)} candles downloaded. "
            "Not enough data for this backtest."
        )

    original_count = len(candles)

    print(
        f"Downloaded {original_count} actual completed candles."
    )

    candles, synthetic_count = repair_short_gaps(candles)

    print(
        f"Total candles after reconstruction: {len(candles)}"
    )

    return candles, synthetic_count


def features(candles, i):
    history = candles[:i + 1]

    closes = [c["close"] for c in history]
    highs = [c["high"] for c in history]
    lows = [c["low"] for c in history]

    price = closes[-1]

    if len(closes) < 150:
        return None

    def momentum(minutes):
        old = closes[-1 - minutes]

        if old == 0:
            return 0.0

        return (price / old - 1) * 100

    mom5 = momentum(5)
    mom15 = momentum(15)
    mom30 = momentum(30)
    mom60 = momentum(60)

    # Match the live bot's EMA calculation windows.
    e9 = ema(closes[-100:], 9)
    e21 = ema(closes[-100:], 21)
    e50 = ema(closes[-150:], 50)

    if not all(
        value is not None
        for value in [e9, e21, e50]
    ):
        return None

    r = rsi(closes, 14)

    changes = [
        (closes[j] / closes[j - 1] - 1) * 100
        for j in range(len(closes) - 30, len(closes))
        if closes[j - 1] != 0
    ]

    if not changes:
        return None

    mean_change = sum(changes) / len(changes)

    volatility = math.sqrt(
        sum(
            (value - mean_change) ** 2
            for value in changes
        ) / len(changes)
    )

    ema_difference = ((e9 - e21) / price) * 100
    long_trend = ((e21 - e50) / price) * 100

    # Current bot calculation.
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

    current_probability = round(
        50 + 30 * tanh(adjusted * 2),
        1
    )

    # Alternative 1: EMA trend.
    ema_score = tanh(ema_difference / 0.12)

    if (
        (price > e50 and ema_score < 0)
        or (price < e50 and ema_score > 0)
    ):
        ema_score = 0

    # Alternative 2: RSI plus short-term momentum.
    rsi_score = tanh((r - 50) / 12)

    rsi_score = (
        0.6 * rsi_score
        + 0.4 * tanh(mom15 / 0.20)
    )

    # Alternative 3: breakout of previous 30 candles.
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
        "current": (
            1 if current_probability >= 50 else -1
        ),
        "ema": ema_score,
        "rsi": rsi_score,
        "breakout": breakout_score,
        "combined": combined_score
    }


def choose_threshold(training, strategy):
    # Threshold selection uses training data only.
    best_threshold = 0
    best_accuracy = -1
    best_count = 0

    threshold_values = [
        0, .05, .10, .15, .20, .25, .30,
        .35, .40, .45, .50, .60, .70, .80
    ]

    for threshold in threshold_values:
        selected = [
            record for record in training
            if abs(record[strategy]) >= threshold
            and record["direction"] != 0
        ]

        if len(selected) < 30:
            continue

        wins = sum(
            1 for record in selected
            if (
                1 if record[strategy] > 0 else -1
            ) == record["direction"]
        )

        accuracy = wins / len(selected)

        if (
            accuracy > best_accuracy
            or (
                accuracy == best_accuracy
                and len(selected) > best_count
            )
        ):
            best_accuracy = accuracy
            best_threshold = threshold
            best_count = len(selected)

    return best_threshold


def main():
    candles, synthetic_count = download_candles()

    first_time = candles[0]["time"]

    training_end = (
        first_time + TRAIN_DAYS * 86400
    )

    strategies = [
        "current",
        "ema",
        "rsi",
        "breakout",
        "combined"
    ]

    records = []

    print("\nCalculating 15-minute predictions...")

    for i in range(
        150,
        len(candles) - HORIZON - 2
    ):
        signal_time = candles[i]["time"] + 60

        # Match the live bot's scheduled prediction times.
        if signal_time % 3600 not in (
            7 * 60,
            22 * 60,
            37 * 60,
            52 * 60
        ):
            continue

        signal = features(candles, i)

        if signal is None:
            continue

        # Entry is the next candle's open.
        entry_index = i + 1

        entry_price = candles[entry_index]["open"]
        entry_time = candles[entry_index]["time"]

        target_time = entry_time + HORIZON * 60

        # The target candle begins at or after target_time.
        outcome_index = next(
            (
                j for j in range(
                    entry_index + 1,
                    len(candles)
                )
                if candles[j]["time"] >= target_time
            ),
            None
        )

        if outcome_index is None:
            continue

        outcome_candle = candles[outcome_index]
        outcome_price = outcome_candle["close"]

        if outcome_price == entry_price:
            direction = 0
        else:
            direction = (
                1 if outcome_price > entry_price else -1
            )

        record = {
            "time": signal_time,
            "entry_time": entry_time,
            "target_time": target_time,
            "outcome_time": outcome_candle["time"],
            "entry": entry_price,
            "outcome": outcome_price,
            "direction": direction,
            "entry_synthetic": candles[entry_index]["synthetic"],
            "outcome_synthetic": outcome_candle["synthetic"]
        }

        record.update(signal)
        records.append(record)

    # A training result must have fully completed before
    # the training period ends. This prevents label leakage.
    training = [
        record for record in records
        if (
            record["time"] < training_end
            and record["outcome_time"] + 60 <= training_end
        )
    ]

    # Out-of-sample predictions begin only after training ends.
    testing = [
        record for record in records
        if record["time"] >= training_end
    ]

    if len(training) < 100 or len(testing) < 100:
        raise RuntimeError(
            "Not enough training or out-of-sample predictions. "
            "Check the downloaded historical data."
        )

    thresholds = {
        "current": 0
    }

    for strategy in strategies[1:]:
        thresholds[strategy] = choose_threshold(
            training,
            strategy
        )

    print("\nDATA QUALITY")
    print(f"Actual candles: {len(candles) - synthetic_count}")
    print(f"Reconstructed candles: {synthetic_count}")
    print("Reconstructed candles use previous close and zero volume.")

    print("\nTRAINING PERIOD")
    print(f"Signals: {len(training)}")
    print("Training duration: 7 days")

    print("\nOUT-OF-SAMPLE PERIOD")
    print(f"Signals: {len(testing)}")
    print("Out-of-sample duration: approximately 23 days")

    print("\nTRAINED THRESHOLDS")

    for strategy in strategies:
        print(
            f"{strategy}: {thresholds[strategy]}"
        )

    print("\nOUT-OF-SAMPLE RESULTS")

    print(
        "Strategy,Predictions,Wins,Losses,"
        "Accuracy,Threshold"
    )

    results = []

    for strategy in strategies:
        threshold = thresholds[strategy]

        selected = [
            record for record in testing
            if (
                abs(record[strategy]) >= threshold
                and record["direction"] != 0
            )
        ]

        wins = sum(
            1 for record in selected
            if (
                1 if record[strategy] > 0 else -1
            ) == record["direction"]
        )

        losses = len(selected) - wins

        accuracy = (
            100 * wins / len(selected)
            if selected else 0
        )

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
            f"{strategy},{len(selected)},"
            f"{wins},{losses},"
            f"{accuracy:.2f}%,{threshold}"
        )

    baseline = next(
        result for result in results
        if result["strategy"] == "current"
    )

    print("\nPERFORMANCE VS CURRENT BOT")

    for result in results:
        difference = (
            result["accuracy_percent"]
            - baseline["accuracy_percent"]
        )

        print(
            f"{result['strategy']}: "
            f"{difference:+.2f} percentage points"
        )

    report = {
        "product": PRODUCT,
        "days": DAYS,
        "training_days": TRAIN_DAYS,
        "out_of_sample_days": DAYS - TRAIN_DAYS,
        "horizon_minutes": HORIZON,
        "actual_candles": len(candles) - synthetic_count,
        "synthetic_candles": synthetic_count,
        "synthetic_candle_method": (
            "Previous close used for OHLC; volume set to zero."
        ),
        "entry_method": "Next one-minute candle open",
        "outcome_method": (
            "Close of first candle beginning at or after "
            "the 15-minute target"
        ),
        "results": results
    }

    with open(
        "backtest_results.json",
        "w"
    ) as file:
        json.dump(
            report,
            file,
            indent=2
        )

    with open(
        "backtest_predictions.csv",
        "w",
        newline=""
    ) as file:
        fields = [
            "time",
            "entry_time",
            "target_time",
            "outcome_time",
            "entry",
            "outcome",
            "direction",
            "entry_synthetic",
            "outcome_synthetic",
            "current",
            "ema",
            "rsi",
            "breakout",
            "combined"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for record in testing:
            writer.writerow(record)

    print("\nSaved backtest_results.json")
    print("Saved backtest_predictions.csv")
    print("\nBACKTEST COMPLETE")


if __name__ == "__main__":
    main()
