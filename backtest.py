
import csv
import json
import math
import statistics
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError

API = "https://api.exchange.coinbase.com"
PRODUCT = "BTC-USD"

DAYS = 30
TRAIN_DAYS = 7
HORIZON = 15
MAX_FILLED_GAP_MINUTES = 5

HEADERS = {
    "User-Agent": "BTC-15m-Historical-Backtest/3.0"
}

STRATEGIES = [
    "current",
    "ema",
    "rsi",
    "breakout",
    "combined"
]


def get_json(url):
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                url,
                headers=HEADERS
            )

            with urllib.request.urlopen(
                request,
                timeout=30
            ) as response:
                return json.loads(
                    response.read().decode()
                )

        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == 4:
                raise RuntimeError(
                    f"Coinbase request failed: {error}"
                )

            time.sleep(2 ** attempt)

    raise RuntimeError("Unable to download market data.")


def ema(values, period):
    if len(values) < period:
        return None

    alpha = 2 / (period + 1)
    result = values[0]

    for value in values[1:]:
        result = (
            alpha * value
            + (1 - alpha) * result
        )

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

    return 100 - (
        100 / (1 + avg_gain / avg_loss)
    )


def download_candles():
    end = datetime.now(timezone.utc).replace(
        second=0,
        microsecond=0
    )

    start = end - timedelta(days=DAYS)

    rows = []
    cursor = start

    print("Downloading historical BTC/USD candles...")
    print("This may take a minute or two.")

    while cursor < end:
        batch_end = min(
            cursor + timedelta(minutes=300),
            end
        )

        params = urllib.parse.urlencode({
            "granularity": 60,
            "start": cursor.isoformat(),
            "end": batch_end.isoformat()
        })

        url = (
            f"{API}/products/{PRODUCT}/candles?"
            f"{params}"
        )

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

    candles = sorted(
        unique.values(),
        key=lambda x: x["time"]
    )

    cutoff = int(end.timestamp())

    candles = [
        c for c in candles
        if c["time"] + 60 <= cutoff
    ]

    if len(candles) < 10000:
        raise RuntimeError(
            f"Only {len(candles)} actual candles downloaded. "
            "Not enough data for this backtest."
        )

    actual_count = len(candles)

    print(
        f"Actual completed candles: {actual_count}"
    )

    candles, synthetic_count = repair_gaps(candles)

    print(
        f"Synthetic candles inserted: {synthetic_count}"
    )

    return candles, actual_count, synthetic_count


def repair_gaps(candles):
    if not candles:
        raise RuntimeError("No historical candles found.")

    repaired = [candles[0]]
    filled_count = 0

    for current in candles[1:]:
        previous = repaired[-1]

        difference = (
            current["time"] - previous["time"]
        )

        if difference <= 0:
            continue

        if difference % 60 != 0:
            raise RuntimeError(
                "Candle timestamps are not aligned "
                "to one-minute intervals."
            )

        missing = difference // 60 - 1

        if missing > MAX_FILLED_GAP_MINUTES:
            raise RuntimeError(
                f"Found a {missing}-minute data gap. "
                "Stopping rather than inventing prices."
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


def calculate_features(candles, i):
    history = candles[:i + 1]

    closes = [c["close"] for c in history]
    highs = [c["high"] for c in history]
    lows = [c["low"] for c in history]

    price = closes[-1]

    if len(closes) < 150:
        return None

    # Do not score signals whose indicator history
    # contains reconstructed candles.
    if any(
        c["synthetic"]
        for c in candles[i - 149:i + 1]
    ):
        return None

    def momentum(minutes):
        old_price = closes[-1 - minutes]

        if old_price == 0:
            return 0.0

        return (
            (price / old_price - 1) * 100
        )

    mom5 = momentum(5)
    mom15 = momentum(15)
    mom30 = momentum(30)
    mom60 = momentum(60)

    # Match the live bot's EMA windows.
    ema9 = ema(closes[-100:], 9)
    ema21 = ema(closes[-100:], 21)
    ema50 = ema(closes[-150:], 50)

    current_rsi = rsi(closes, 14)

    if any(
        x is None
        for x in [ema9, ema21, ema50, current_rsi]
    ):
        return None

    changes = [
        (
            (closes[j] - closes[j - 1])
            / closes[j - 1]
        ) * 100
        for j in range(
            len(closes) - 30,
            len(closes)
        )
    ]

    volatility = statistics.pstdev(changes)

    ema_difference = (
        (ema9 - ema21) / price
    ) * 100

    long_trend = (
        (ema21 - ema50) / price
    ) * 100

    # Current live bot strategy.
    score = (
        0.15 * math.tanh(mom5 / 0.10)
        + 0.20 * math.tanh(mom15 / 0.20)
        + 0.15 * math.tanh(mom30 / 0.30)
        + 0.10 * math.tanh(mom60 / 0.40)
        + 0.20 * math.tanh(ema_difference / 0.12)
        + 0.10 * math.tanh(long_trend / 0.20)
        + 0.10 * math.tanh(
            (current_rsi - 50) / 12
        )
    )

    adjusted = score / (1 + volatility * 5)

    current_probability = round(
        50 + 30 * math.tanh(adjusted * 2),
        1
    )

    current_signal = (
        1 if current_probability >= 50 else -1
    )

    # Strategy 2: EMA.
    ema_score = math.tanh(
        ema_difference / 0.12
    )

    if (
        (price > ema50 and ema_score < 0)
        or (price < ema50 and ema_score > 0)
    ):
        ema_score = 0

    # Strategy 3: RSI and momentum.
    rsi_score = math.tanh(
        (current_rsi - 50) / 12
    )

    rsi_score = (
        0.60 * rsi_score
        + 0.40 * math.tanh(mom15 / 0.20)
    )

    # Strategy 4: Breakout.
    prior_high = max(highs[-31:-1])
    prior_low = min(lows[-31:-1])

    breakout_score = 0

    if price > prior_high and mom5 > 0:
        breakout_score = 1

    elif price < prior_low and mom5 < 0:
        breakout_score = -1

    # Strategy 5: Combined.
    combined_score = (
        0.35 * math.tanh(ema_difference / 0.12)
        + 0.25 * math.tanh(long_trend / 0.20)
        + 0.25 * math.tanh(mom15 / 0.20)
        + 0.15 * math.tanh(
            (current_rsi - 50) / 12
        )
    )

    combined_score *= (
        1 / (1 + volatility * 5)
    )

    return {
        "current": current_signal,
        "ema": ema_score,
        "rsi": rsi_score,
        "breakout": breakout_score,
        "combined": combined_score
    }


def choose_threshold(training, strategy):
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
            if (
                abs(record[strategy]) > 0
                and abs(record[strategy]) >= threshold
                and record["direction"] != 0
            )
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


def build_records(candles):
    records = []

    print("\nCalculating historical signals...")

    # Signal is calculated after the previous candle closes.
    # Entry is the next candle's open.
    for i in range(149, len(candles) - HORIZON - 1):

        signal_time = candles[i]["time"] + 60

        # Match the live 15-minute schedule:
        # :00, :15, :30 and :45.
        if signal_time % 900 != 0:
            continue

        signals = calculate_features(candles, i)

        if signals is None:
            continue

        entry_index = i + 1
        entry_candle = candles[entry_index]

        if entry_candle["synthetic"]:
            continue

        entry_time = entry_candle["time"]
        entry_price = entry_candle["open"]

        target_time = entry_time + HORIZON * 60

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

        if outcome_candle["synthetic"]:
            continue

        outcome_price = outcome_candle["close"]

        if outcome_price > entry_price:
            direction = 1
        elif outcome_price < entry_price:
            direction = -1
        else:
            direction = 0

        record = {
            "time": signal_time,
            "entry_time": entry_time,
            "target_time": target_time,
            "outcome_time": outcome_candle["time"],
            "entry": entry_price,
            "outcome": outcome_price,
            "direction": direction,
            "entry_synthetic": False,
            "outcome_synthetic": False
        }

        record.update(signals)

        records.append(record)

    return records


def evaluate_strategy(records, strategy, threshold):
    selected = [
        record for record in records
        if (
            abs(record[strategy]) > 0
            and abs(record[strategy]) >= threshold
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

    return {
        "strategy": strategy,
        "predictions": len(selected),
        "wins": wins,
        "losses": losses,
        "accuracy_percent": round(accuracy, 2),
        "training_threshold": threshold
    }


def save_reports(testing, results, report):
    with open(
        "backtest_results.json",
        "w"
    ) as file:
        json.dump(
            report,
            file,
            indent=2
        )

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

    with open(
        "backtest_predictions.csv",
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(testing)

    with open(
        "backtest_strategy_results.csv",
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "strategy",
                "predictions",
                "wins",
                "losses",
                "accuracy_percent",
                "training_threshold"
            ]
        )

        writer.writeheader()
        writer.writerows(results)


def main():
    print("=" * 55)
    print("BITCOIN 15-MINUTE HISTORICAL BACKTEST")
    print("=" * 55)

    candles, actual_count, synthetic_count = (
        download_candles()
    )

    records = build_records(candles)

    if len(records) < 200:
        raise RuntimeError(
            f"Only {len(records)} valid signals found. "
            "Not enough data for reliable testing."
        )

    first_time = candles[0]["time"]

    training_end = (
        first_time + TRAIN_DAYS * 86400
    )

    training = [
        record for record in records
        if (
            record["time"] < training_end
            and record["outcome_time"] + 60 <= training_end
        )
    ]

    testing = [
        record for record in records
        if record["time"] >= training_end
    ]

    if len(training) < 100 or len(testing) < 100:
        raise RuntimeError(
            "Not enough training or testing signals."
        )

    print("\nDATA PERIOD")
    print("Total historical period: 30 days")
    print("Training period: 7 days")
    print("Out-of-sample testing: approximately 23 days")

    print("\nDATA QUALITY")
    print(f"Actual candles: {actual_count}")
    print(f"Synthetic candles inserted: {synthetic_count}")
    print(f"Valid historical signals: {len(records)}")

    print("\nTRAINING")
    print(f"Training signals: {len(training)}")

    print("\nOUT-OF-SAMPLE TEST")
    print(f"Testing signals: {len(testing)}")

    thresholds = {
        "current": 0
    }

    for strategy in STRATEGIES[1:]:
        thresholds[strategy] = choose_threshold(
            training,
            strategy
        )

    print("\nTRAINED THRESHOLDS")

    for strategy in STRATEGIES:
        print(
            f"{strategy}: {thresholds[strategy]}"
        )

    results = []

    for strategy in STRATEGIES:
        result = evaluate_strategy(
            testing,
            strategy,
            thresholds[strategy]
        )

        results.append(result)

    baseline = next(
        result for result in results
        if result["strategy"] == "current"
    )

    print("\nFINAL OUT-OF-SAMPLE RESULTS")
    print("-" * 65)

    print(
        f"{'Strategy':<12}"
        f"{'Signals':>10}"
        f"{'Wins':>9}"
        f"{'Losses':>9}"
        f"{'Accuracy':>12}"
    )

    print("-" * 65)

    for result in results:
        print(
            f"{result['strategy']:<12}"
            f"{result['predictions']:>10}"
            f"{result['wins']:>9}"
            f"{result['losses']:>9}"
            f"{result['accuracy_percent']:>11.2f}%"
        )

    print("-" * 65)

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
        "historical_days": DAYS,
        "training_days": TRAIN_DAYS,
        "testing_days": DAYS - TRAIN_DAYS,
        "horizon_minutes": HORIZON,
        "signal_schedule": [
            ":00", ":15", ":30", ":45"
        ],
        "actual_candles": actual_count,
        "synthetic_candles": synthetic_count,
        "synthetic_method": (
            "Previous close used for OHLC, zero volume. "
            "Signals with synthetic indicator history, "
            "entry or outcome candles are excluded."
        ),
        "entry_method": (
            "Next one-minute candle open"
        ),
        "outcome_method": (
            "Close of first candle beginning at or "
            "after the 15-minute target"
        ),
        "training_signals": len(training),
        "testing_signals": len(testing),
        "thresholds": thresholds,
        "results": results
    }

    save_reports(
        testing,
        results,
        report
    )

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
