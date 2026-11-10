
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
# BTC 15-MINUTE ADVANCED WALK-FORWARD BACKTEST
# Tests trend, momentum, mean reversion, breakouts, volume,
# combined signals, and the reverse of the current baseline.
#
# No third-party packages required.
# ============================================================

API = "https://api.exchange.coinbase.com"
PRODUCT = "BTC-USD"

DAYS = 30
TRAIN_DAYS = 7
HORIZON_MINUTES = 15
SIGNAL_MINUTES = (7, 22, 37, 52)
MAX_FILLED_GAP_MINUTES = 5
MIN_TRAINING_SIGNALS = 40
MIN_TEST_SIGNALS_FOR_CANDIDATE = 50

HEADERS = {
    "User-Agent": "BTC-Advanced-15m-Backtest/5.0"
}

STRATEGIES = [
    "current",
    "contrarian",
    "trend",
    "momentum",
    "mean_reversion",
    "breakout",
    "volume_breakout",
    "combined",
]


def get_json(url):
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
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=DAYS)

    rows = []
    cursor = start

    print("Downloading BTC/USD one-minute candles...")
    print("This may take a few minutes.")

    while cursor < end:
        batch_end = min(cursor + timedelta(minutes=300), end)

        params = urllib.parse.urlencode({
            "granularity": 60,
            "start": cursor.isoformat(),
            "end": batch_end.isoformat()
        })

        url = f"{API}/products/{PRODUCT}/candles?{params}"
        rows.extend(get_json(url))

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

    candles = sorted(unique.values(), key=lambda x: x["time"])

    cutoff = int(end.timestamp())
    candles = [
        candle for candle in candles
        if candle["time"] + 60 <= cutoff
    ]

    if len(candles) < 10000:
        raise RuntimeError(
            f"Only {len(candles)} actual candles downloaded. "
            "Not enough data for a reliable backtest."
        )

    actual_count = len(candles)
    candles, synthetic_count = repair_gaps(candles)

    print(f"Actual candles: {actual_count}")
    print(f"Inserted synthetic candles: {synthetic_count}")

    return candles, actual_count, synthetic_count


def repair_gaps(candles):
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
            raise RuntimeError("Candle timestamps are not minute-aligned.")

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

    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def calculate_features(candles, index):
    window = candles[index - 149:index + 1]

    if len(window) != 150:
        return None

    if any(candle["synthetic"] for candle in window):
        return None

    closes = [c["close"] for c in window]
    highs = [c["high"] for c in window]
    lows = [c["low"] for c in window]
    volumes = [c["volume"] for c in window]

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

    if any(x is None for x in (ema9, ema21, ema50, current_rsi)):
        return None

    returns = []

    for i in range(len(closes) - 30, len(closes)):
        previous = closes[i - 1]

        if previous <= 0:
            return None

        returns.append(
            ((closes[i] / previous) - 1.0) * 100.0
        )

    volatility = statistics.pstdev(returns)

    ema_difference = ((ema9 - ema21) / price) * 100.0
    long_trend = ((ema21 - ema50) / price) * 100.0

    average_recent_volume = sum(volumes[-5:]) / 5.0
    average_prior_volume = sum(volumes[-30:-5]) / 25.0

    if average_prior_volume > 0:
        volume_ratio = average_recent_volume / average_prior_volume
    else:
        volume_ratio = 1.0

    # Baseline: preserve the existing bot's directional method.
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
    current_probability = (
        50.0 + 30.0 * math.tanh(adjusted_current * 2.0)
    )

    current_signal = 1.0 if current_probability >= 50.0 else -1.0

    # Test the opposite of the current signal separately.
    contrarian_score = -current_signal

    # Trend-following: short and long EMA alignment.
    trend_score = (
        0.55 * math.tanh(ema_difference / 0.12)
        + 0.45 * math.tanh(long_trend / 0.20)
    )

    # Momentum: several time windows must agree.
    momentum_score = (
        0.35 * math.tanh(mom5 / 0.10)
        + 0.45 * math.tanh(mom15 / 0.20)
        + 0.20 * math.tanh(mom30 / 0.30)
    )

    # Mean reversion: tests whether extreme short-term movement
    # is more likely to reverse than continue.
    mean_reversion_score = (
        -0.60 * math.tanh((current_rsi - 50.0) / 12.0)
        -0.40 * math.tanh(mom15 / 0.20)
    )

    # Breakout beyond the previous 30 completed candles.
    prior_high = max(highs[-31:-1])
    prior_low = min(lows[-31:-1])

    breakout_score = 0.0

    if price > prior_high:
        breakout_score = 1.0
    elif price < prior_low:
        breakout_score = -1.0

    # Breakouts need confirmation from volume.
    volume_confirmation = math.tanh((volume_ratio - 1.0) / 0.75)
    volume_breakout_score = breakout_score * volume_confirmation

    # Combined model: trend + momentum + RSI.
    combined_score = (
        0.35 * math.tanh(ema_difference / 0.12)
        + 0.25 * math.tanh(long_trend / 0.20)
        + 0.25 * math.tanh(mom15 / 0.20)
        + 0.15 * math.tanh((current_rsi - 50.0) / 12.0)
    )

    combined_score /= (1.0 + volatility * 5.0)

    return {
        "current": current_signal,
        "contrarian": contrarian_score,
        "trend": trend_score,
        "momentum": momentum_score,
        "mean_reversion": mean_reversion_score,
        "breakout": breakout_score,
        "volume_breakout": volume_breakout_score,
        "combined": combined_score,
        "rsi_value": current_rsi,
        "volatility": volatility,
        "volume_ratio": volume_ratio
    }


def build_records(candles):
    print("\nCalculating scheduled signals...")

    records = []
    time_to_index = {
        candle["time"]: i
        for i, candle in enumerate(candles)
    }

    for index in range(149, len(candles) - HORIZON_MINUTES - 1):
        signal_time = candles[index]["time"] + 60

        signal_minute = datetime.fromtimestamp(
            signal_time, timezone.utc
        ).minute

        if signal_minute not in SIGNAL_MINUTES:
            continue

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
        target_time = entry_time + HORIZON_MINUTES * 60

        outcome_index = time_to_index.get(target_time)

        if outcome_index is None:
            continue

        outcome_candle = candles[outcome_index]

        if outcome_candle["synthetic"]:
            continue

        entry_price = entry_candle["open"]
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
    # Thresholds are selected from training data only.
    candidates = [
        0.00, 0.05, 0.10, 0.15, 0.20,
        0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.60, 0.70, 0.80, 0.90
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

        if len(selected) < MIN_TRAINING_SIGNALS:
            continue

        wins = sum(
            1 for record in selected
            if (1 if record[strategy] > 0 else -1)
            == record["direction"]
        )

        accuracy = wins / len(selected)

        if (
            accuracy > best_accuracy
            or (
                math.isclose(accuracy, best_accuracy)
                and len(selected) > best_count
            )
        ):
            best_accuracy = accuracy
            best_threshold = threshold
            best_count = len(selected)

    return best_threshold


def wilson_interval(wins, total):
    if total == 0:
        return 0.0, 0.0

    z = 1.96
    p = wins / total
    denominator = 1.0 + (z * z / total)

    center = (
        p + z * z / (2.0 * total)
    ) / denominator

    margin = (
        z * math.sqrt(
            (p * (1.0 - p) / total)
            + (z * z / (4.0 * total * total))
        )
    ) / denominator

    return (
        round(max(0.0, center - margin) * 100.0, 2),
        round(min(1.0, center + margin) * 100.0, 2)
    )


def evaluate_strategy(records, strategy, threshold):
    selected = [
        record for record in records
        if record["direction"] != 0
        and abs(record[strategy]) > 0
        and abs(record[strategy]) >= threshold
    ]

    wins = sum(
        1 for record in selected
        if (1 if record[strategy] > 0 else -1)
        == record["direction"]
    )

    total = len(selected)
    losses = total - wins

    accuracy = (
        100.0 * wins / total
        if total else 0.0
    )

    ci_low, ci_high = wilson_interval(wins, total)

    return {
        "strategy": strategy,
        "predictions": total,
        "wins": wins,
        "losses": losses,
        "accuracy_percent": round(accuracy, 2),
        "confidence_interval_low": ci_low,
        "confidence_interval_high": ci_high,
        "training_threshold": threshold
    }


def save_reports(test_records, results, report):
    with open(
        "backtest_results.json", "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    detail_fields = [
        "signal_time",
        "entry_time",
        "target_time",
        "entry_price",
        "outcome_price",
        "direction",
        "current",
        "contrarian",
        "trend",
        "momentum",
        "mean_reversion",
        "breakout",
        "volume_breakout",
        "combined",
        "rsi_value",
        "volatility",
        "volume_ratio"
    ]

    with open(
        "backtest_predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=detail_fields
        )
        writer.writeheader()
        writer.writerows(test_records)

    result_fields = [
        "strategy",
        "predictions",
        "wins",
        "losses",
        "accuracy_percent",
        "confidence_interval_low",
        "confidence_interval_high",
        "training_threshold"
    ]

    with open(
        "backtest_strategy_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=result_fields
        )
        writer.writeheader()
        writer.writerows(results)


def main():
    print("=" * 70)
    print("BITCOIN ADVANCED 15-MINUTE BACKTEST")
    print("=" * 70)

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

    training = [
        record for record in records
        if record["signal_time"] < training_end
        and record["target_time"] <= training_end
    ]

    testing = [
        record for record in records
        if record["signal_time"] >= training_end
    ]

    if (
        len(training) < 100
        or len(testing) < 100
    ):
        raise RuntimeError(
            f"Insufficient data: {len(training)} training signals "
            f"and {len(testing)} testing signals."
        )

    actual_days = (last_time - first_time) / 86400.0
    testing_days = (last_time - training_end) / 86400.0

    print("\nDATA QUALITY")
    print(f"Actual candles: {actual_count}")
    print(f"Synthetic candles: {synthetic_count}")
    print(f"Valid scheduled signals: {len(records)}")
    print(f"Historical days: {actual_days:.2f}")

    print("\nDATA SPLIT")
    print(f"Training signals: {len(training)}")
    print(f"Unseen testing signals: {len(testing)}")
    print(f"Training days: {TRAIN_DAYS}")
    print(f"Testing days: {testing_days:.2f}")

    thresholds = {}

    for strategy in STRATEGIES:
        if strategy == "current" or strategy == "contrarian":
            thresholds[strategy] = 0.0
        else:
            thresholds[strategy] = choose_threshold(
                training, strategy
            )

    print("\nTRAINING-SELECTED THRESHOLDS")

    for strategy in STRATEGIES:
        print(
            f"{strategy}: {thresholds[strategy]:.2f}"
        )

    results = [
        evaluate_strategy(
            testing,
            strategy,
            thresholds[strategy]
        )
        for strategy in STRATEGIES
    ]

    baseline = next(
        result for result in results
        if result["strategy"] == "current"
    )

    print("\nUNSEEN TEST RESULTS")
    print("-" * 85)
    print(
        f"{'Strategy':<19}"
        f"{'Signals':>9}"
        f"{'Wins':>8}"
        f"{'Losses':>9}"
        f"{'Accuracy':>12}"
        f"{'95% range':>19}"
    )
    print("-" * 85)

    for result in results:
        interval = (
            f"{result['confidence_interval_low']:.1f}-"
            f"{result['confidence_interval_high']:.1f}%"
        )

        print(
            f"{result['strategy']:<19}"
            f"{result['predictions']:>9}"
            f"{result['wins']:>8}"
            f"{result['losses']:>9}"
            f"{result['accuracy_percent']:>11.2f}%"
            f"{interval:>19}"
        )

    print("-" * 85)

    print("\nCHANGE VS CURRENT BOT")

    for result in results:
        difference = (
            result["accuracy_percent"]
            - baseline["accuracy_percent"]
        )

        print(
            f"{result['strategy']}: "
            f"{difference:+.2f} percentage points"
        )

    eligible = [
        result for result in results
        if result["predictions"] >= MIN_TEST_SIGNALS_FOR_CANDIDATE
    ]

    best_candidate = (
        max(
            eligible,
            key=lambda result: result["accuracy_percent"]
        )
        if eligible else None
    )

    report = {
        "product": PRODUCT,
        "requested_historical_days": DAYS,
        "actual_historical_days": round(actual_days, 2),
        "training_days": TRAIN_DAYS,
        "testing_days": round(testing_days, 2),
        "horizon_minutes": HORIZON_MINUTES,
        "signal_schedule_utc": [
            ":07", ":22", ":37", ":52"
        ],
        "entry_method": (
            "Open of the one-minute candle at scheduled signal time"
        ),
        "exit_method": (
            "Open of the candle exactly 15 minutes after entry"
        ),
        "actual_candles": actual_count,
        "synthetic_candles": synthetic_count,
        "synthetic_policy": (
            "Only small gaps are filled using previous close and "
            "zero volume. Signals using synthetic feature, entry, "
            "or exit candles are excluded."
        ),
        "training_signals": len(training),
        "testing_signals": len(testing),
        "thresholds": thresholds,
        "results": results,
        "highest_test_accuracy_candidate": best_candidate,
        "warning": (
            "The test period is separate from the training period. "
            "However, choosing a strategy after inspecting test results "
            "can still lead to overfitting across repeated experiments. "
            "Confirm a candidate on a new later holdout period before "
            "using it for live trading. Accuracy does not guarantee profit."
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
