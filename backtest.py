import csv
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError

# ============================================================
# BITCOIN 15-MINUTE WALK-FORWARD BACKTEST
# Matches the live bot's scheduled run times:
# :07, :22, :37, :52
#
# No third-party Python packages are required.
# ============================================================

API = "https://api.exchange.coinbase.com"
PRODUCT = "BTC-USD"

DAYS = 30
TRAIN_DAYS = 7
HORIZON_MINUTES = 15
SIGNAL_MINUTES = (7, 22, 37, 52)

# Only tiny data gaps may be filled. Any signal whose feature
# window, entry, or exit uses a synthetic candle is excluded.
MAX_FILLED_GAP_MINUTES = 5

HEADERS = {
    "User-Agent": "BTC-15m-WalkForward-Backtest/4.0"
}

STRATEGIES = ["current", "ema", "rsi", "breakout", "combined"]


def get_json(url):
    """Download JSON with retries for temporary API failures."""
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == 4:
                raise RuntimeError(f"Coinbase request failed: {error}")
            time.sleep(2 ** attempt)

    raise RuntimeError("Unable to download market data.")


def download_candles():
    """Download completed one-minute BTC-USD candles."""
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=DAYS)

    rows = []
    cursor = start

    print("Downloading historical BTC/USD one-minute candles...")
    print("This can take a few minutes.")

    while cursor < end:
        batch_end = min(cursor + timedelta(minutes=300), end)

        params = urllib.parse.urlencode({
            "granularity": 60,
            "start": cursor.isoformat(),
            "end": batch_end.isoformat()
        })

        url = f"{API}/products/{PRODUCT}/candles?{params}"
        data = get_json(url)
        rows.extend(data)

        cursor = batch_end
        time.sleep(0.25)

    unique = {}

    for row in rows:
        if len(row) < 6:
            continue

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

    candles = sorted(unique.values(), key=lambda item: item["time"])

    # Exclude any candle that had not fully closed at download time.
    cutoff = int(end.timestamp())
    candles = [
        candle for candle in candles
        if candle["time"] + 60 <= cutoff
    ]

    if len(candles) < 10000:
        raise RuntimeError(
            f"Only {len(candles)} actual candles downloaded. "
            "Not enough data for this backtest."
        )

    actual_count = len(candles)
    candles, synthetic_count = repair_gaps(candles)

    print(f"Actual completed candles: {actual_count}")
    print(f"Synthetic candles inserted: {synthetic_count}")

    return candles, actual_count, synthetic_count


def repair_gaps(candles):
    """Fill only small missing intervals with previous close."""
    if not candles:
        raise RuntimeError("No historical candles found.")

    repaired = [candles[0]]
    filled_count = 0

    for current in candles[1:]:
        previous = repaired[-1]
        difference = current["time"] - previous["time"]

        if difference <= 0:
            continue

        if difference % 60 != 0:
            raise RuntimeError(
                "Candle timestamps are not aligned to one-minute intervals."
            )

        missing = difference // 60 - 1

        if missing > MAX_FILLED_GAP_MINUTES:
            raise RuntimeError(
                f"Found a {missing}-minute data gap. "
                "Stopping rather than inventing market prices."
            )

        previous_close = previous["close"]

        for minute in range(1, missing + 1):
            repaired.append({
                "time": previous["time"] + minute * 60,
                "open": previous_close,
                "high": previous_close,
                "low": previous_close,
                "close": previous_close,
                "volume": 0.0,
                "synthetic": True
            })
            filled_count += 1

        repaired.append(current)

    return repaired, filled_count


def ema(values, period):
    """Exponential moving average, matching the live bot method."""
    if len(values) < period:
        return None

    alpha = 2.0 / (period + 1.0)
    result = values[0]

    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result

    return result


def rsi(values, period=14):
    """Simple-average RSI over the most recent period changes."""
    if len(values) < period + 1:
        return None

    changes = [
        values[index] - values[index - 1]
        for index in range(len(values) - period, len(values))
    ]

    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0

    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def calculate_features(candles, index):
    """Calculate all strategy scores using information available then."""
    first_index = index - 149
    window = candles[first_index:index + 1]

    if len(window) != 150:
        return None

    # Never generate a signal from a feature window containing
    # reconstructed candles.
    if any(candle["synthetic"] for candle in window):
        return None

    closes = [candle["close"] for candle in window]
    highs = [candle["high"] for candle in window]
    lows = [candle["low"] for candle in window]

    price = closes[-1]
    if price <= 0:
        return None

    def momentum(minutes):
        old_price = closes[-1 - minutes]
        if old_price <= 0:
            return 0.0
        return ((price / old_price) - 1.0) * 100.0

    mom5 = momentum(5)
    mom15 = momentum(15)
    mom30 = momentum(30)
    mom60 = momentum(60)

    ema9 = ema(closes[-100:], 9)
    ema21 = ema(closes[-100:], 21)
    ema50 = ema(closes[-150:], 50)
    current_rsi = rsi(closes, 14)

    if any(value is None for value in (ema9, ema21, ema50, current_rsi)):
        return None

    returns = []
    for position in range(len(closes) - 30, len(closes)):
        previous = closes[position - 1]
        if previous <= 0:
            return None
        returns.append(((closes[position] / previous) - 1.0) * 100.0)

    volatility = statistics.pstdev(returns)

    ema_difference = ((ema9 - ema21) / price) * 100.0
    long_trend = ((ema21 - ema50) / price) * 100.0

    # Existing live-bot score, retained as the baseline.
    current_score = (
        0.15 * math.tanh(mom5 / 0.10)
        + 0.20 * math.tanh(mom15 / 0.20)
        + 0.15 * math.tanh(mom30 / 0.30)
        + 0.10 * math.tanh(mom60 / 0.40)
        + 0.20 * math.tanh(ema_difference / 0.12)
        + 0.10 * math.tanh(long_trend / 0.20)
        + 0.10 * math.tanh((current_rsi - 50.0) / 12.0)
    )

    adjusted_current = current_score / (1.0 + volatility * 5.0)
    current_probability = 50.0 + 30.0 * math.tanh(adjusted_current * 2.0)
    current_signal = 1 if current_probability >= 50.0 else -1

    # Strategy 2: EMA trend alignment.
    ema_score = math.tanh(ema_difference / 0.12)

    if (
        (price > ema50 and ema_score < 0)
        or (price < ema50 and ema_score > 0)
    ):
        ema_score = 0.0

    # Strategy 3: RSI plus short-term momentum.
    rsi_score = (
        0.60 * math.tanh((current_rsi - 50.0) / 12.0)
        + 0.40 * math.tanh(mom15 / 0.20)
    )

    # Strategy 4: Breakout beyond the previous 30 candles.
    prior_high = max(highs[-31:-1])
    prior_low = min(lows[-31:-1])
    breakout_score = 0.0

    if price > prior_high and mom5 > 0:
        breakout_score = 1.0
    elif price < prior_low and mom5 < 0:
        breakout_score = -1.0

    # Strategy 5: Weighted trend and momentum combination.
    combined_score = (
        0.35 * math.tanh(ema_difference / 0.12)
        + 0.25 * math.tanh(long_trend / 0.20)
        + 0.25 * math.tanh(mom15 / 0.20)
        + 0.15 * math.tanh((current_rsi - 50.0) / 12.0)
    )
    combined_score /= (1.0 + volatility * 5.0)

    return {
        "current": float(current_signal),
        "ema": float(ema_score),
        "rsi": float(rsi_score),
        "breakout": float(breakout_score),
        "combined": float(combined_score),
        "rsi_value": float(current_rsi),
        "volatility": float(volatility)
    }


def build_records(candles):
    """
    Create signals at :07/:22/:37/:52.

    At a scheduled time, the most recently completed candle is used.
    Entry is the next candle's open. Exit is the open at the exact
    15-minute target timestamp, avoiding an extra minute of price data.
    """
    records = []
    print("\nCalculating historical signals...")

    for index in range(149, len(candles) - HORIZON_MINUTES - 1):
        signal_time = candles[index]["time"] + 60

        signal_minute = datetime.fromtimestamp(
            signal_time, timezone.utc
        ).minute

        if signal_minute not in SIGNAL_MINUTES:
            continue

        # Require exact schedule alignment, not just matching minute.
        if signal_time % 900 != 420:
            continue

        signals = calculate_features(candles, index)
        if signals is None:
            continue

        entry_index = index + 1
        entry_candle = candles[entry_index]

        if entry_candle["time"] != signal_time:
            continue
        if entry_candle["synthetic"]:
            continue

        entry_time = entry_candle["time"]
        entry_price = entry_candle["open"]
        target_time = entry_time + HORIZON_MINUTES * 60

        # Find the candle beginning exactly at the target time.
        outcome_index = next(
            (
                position for position in range(entry_index + 1, len(candles))
                if candles[position]["time"] == target_time
            ),
            None
        )

        if outcome_index is None:
            continue

        outcome_candle = candles[outcome_index]
        if outcome_candle["synthetic"]:
            continue

        # The target price is the open at the 15-minute timestamp.
        outcome_price = outcome_candle["open"]

        if outcome_price > entry_price:
            direction = 1
        elif outcome_price < entry_price:
            direction = -1
        else:
            direction = 0

        record = {
            "signal_time": signal_time,
            "entry_time": entry_time,
            "target_time": target_time,
            "entry_price": entry_price,
            "outcome_price": outcome_price,
            "direction": direction
        }
        record.update(signals)
        records.append(record)

    return records


def choose_threshold(training, strategy):
    """
    Choose a confidence threshold using training data only.
    Require a minimum sample size to reduce tiny-sample overfitting.
    """
    candidates = [
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80
    ]

    best_threshold = 0.0
    best_accuracy = -1.0
    best_count = 0

    for threshold in candidates:
        selected = [
            record for record in training
            if record["direction"] != 0
            and abs(record[strategy]) >= threshold
            and abs(record[strategy]) > 0
        ]

        if len(selected) < 40:
            continue

        wins = sum(
            1 for record in selected
            if (1 if record[strategy] > 0 else -1) == record["direction"]
        )
        accuracy = wins / len(selected)

        # Prefer higher training accuracy; on ties prefer more signals.
        if accuracy > best_accuracy or (
            math.isclose(accuracy, best_accuracy) and len(selected) > best_count
        ):
            best_accuracy = accuracy
            best_threshold = threshold
            best_count = len(selected)

    return best_threshold


def evaluate_strategy(records, strategy, threshold):
    """Evaluate one fixed threshold on the untouched test period."""
    selected = [
        record for record in records
        if record["direction"] != 0
        and abs(record[strategy]) > 0
        and abs(record[strategy]) >= threshold
    ]

    wins = sum(
        1 for record in selected
        if (1 if record[strategy] > 0 else -1) == record["direction"]
    )
    losses = len(selected) - wins
    accuracy = (100.0 * wins / len(selected)) if selected else 0.0

    return {
        "strategy": strategy,
        "predictions": len(selected),
        "wins": wins,
        "losses": losses,
        "accuracy_percent": round(accuracy, 2),
        "training_threshold": threshold
    }


def save_reports(test_records, results, report):
    """Save JSON summary and CSV detail files."""
    with open("backtest_results.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    detail_fields = [
        "signal_time",
        "entry_time",
        "target_time",
        "entry_price",
        "outcome_price",
        "direction",
        "current",
        "ema",
        "rsi",
        "breakout",
        "combined",
        "rsi_value",
        "volatility"
    ]

    with open(
        "backtest_predictions.csv", "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(test_records)

    result_fields = [
        "strategy",
        "predictions",
        "wins",
        "losses",
        "accuracy_percent",
        "training_threshold"
    ]

    with open(
        "backtest_strategy_results.csv", "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(results)


def main():
    print("=" * 65)
    print("BITCOIN 15-MINUTE WALK-FORWARD BACKTEST")
    print("=" * 65)

    candles, actual_count, synthetic_count = download_candles()
    records = build_records(candles)

    if len(records) < 200:
        raise RuntimeError(
            f"Only {len(records)} valid signals found. "
            "At least 200 are required."
        )

    first_time = candles[0]["time"]
    last_time = candles[-1]["time"]
    training_end = first_time + TRAIN_DAYS * 86400

    # Training outcomes must have fully occurred before test period.
    training = [
        record for record in records
        if record["signal_time"] < training_end
        and record["target_time"] <= training_end
    ]

    testing = [
        record for record in records
        if record["signal_time"] >= training_end
    ]

    if len(training) < 100 or len(testing) < 100:
        raise RuntimeError(
            f"Insufficient data: {len(training)} training signals and "
            f"{len(testing)} test signals."
        )

    actual_days = (last_time - first_time) / 86400.0
    testing_days = (last_time - training_end) / 86400.0

    print("\nDATA QUALITY")
    print(f"Actual candles: {actual_count}")
    print(f"Synthetic candles inserted: {synthetic_count}")
    print(f"Valid scheduled signals: {len(records)}")
    print(f"Actual historical period: {actual_days:.2f} days")

    print("\nDATA SPLIT")
    print(f"Training signals: {len(training)}")
    print(f"Out-of-sample test signals: {len(testing)}")
    print(f"Training period: {TRAIN_DAYS} days")
    print(f"Approximate test period: {testing_days:.2f} days")

    thresholds = {"current": 0.0}

    for strategy in STRATEGIES:
        if strategy == "current":
            continue
        thresholds[strategy] = choose_threshold(training, strategy)

    print("\nTHRESHOLDS SELECTED USING TRAINING DATA ONLY")
    for strategy in STRATEGIES:
        print(f"{strategy}: {thresholds[strategy]:.2f}")

    results = [
        evaluate_strategy(testing, strategy, thresholds[strategy])
        for strategy in STRATEGIES
    ]

    baseline = next(
        result for result in results if result["strategy"] == "current"
    )

    print("\nFINAL UNSEEN TEST RESULTS")
    print("-" * 68)
    print(
        f"{'Strategy':<12}{'Signals':>10}{'Wins':>9}"
        f"{'Losses':>9}{'Accuracy':>13}"
    )
    print("-" * 68)

    for result in results:
        print(
            f"{result['strategy']:<12}"
            f"{result['predictions']:>10}"
            f"{result['wins']:>9}"
            f"{result['losses']:>9}"
            f"{result['accuracy_percent']:>12.2f}%"
        )

    print("-" * 68)
    print("\nCHANGE VS CURRENT BOT BASELINE")
    for result in results:
        difference = result["accuracy_percent"] - baseline["accuracy_percent"]
        print(f"{result['strategy']}: {difference:+.2f} percentage points")

    # Report the strategy with the highest test accuracy, but do not
    # automatically deploy it. The test set is for evaluation, not tuning.
    eligible = [
        result for result in results
        if result["predictions"] >= 50
    ]
    best_test_strategy = (
        max(eligible, key=lambda result: result["accuracy_percent"])
        if eligible else None
    )

    report = {
        "product": PRODUCT,
        "requested_historical_days": DAYS,
        "actual_historical_days": round(actual_days, 2),
        "training_days": TRAIN_DAYS,
        "testing_days": round(testing_days, 2),
        "horizon_minutes": HORIZON_MINUTES,
        "signal_schedule_utc": [":07", ":22", ":37", ":52"],
        "entry_method": "Open of the one-minute candle at scheduled signal time",
        "exit_method": "Open of the candle exactly 15 minutes after entry",
        "actual_candles": actual_count,
        "synthetic_candles": synthetic_count,
        "synthetic_policy": (
            "Small gaps only are filled with previous close and zero volume. "
            "Signals are excluded if the feature window, entry, or exit "
            "contains a synthetic candle."
        ),
        "training_signals": len(training),
        "testing_signals": len(testing),
        "thresholds": thresholds,
        "results": results,
        "highest_test_accuracy_candidate": best_test_strategy,
        "warning": (
            "The test results are out-of-sample for this run, but selecting "
            "a winner after viewing test results can overfit future decisions. "
            "Confirm any candidate on a new, later holdout period before "
            "deploying it."
        )
    }

    save_reports(testing, results, report)

    print("\nREPORTS SAVED")
    print("backtest_results.json")
    print("backtest_predictions.csv")
    print("backtest_strategy_results.csv")
    print("\nBACKTEST COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nBACKTEST ERROR: {error}")
        raise
