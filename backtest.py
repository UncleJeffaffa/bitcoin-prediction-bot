
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
# BTC 15-MINUTE WALK-FORWARD BACKTEST
#
# 90 days of one-minute Coinbase BTC-USD candles.
# First 15 days: initial training period.
# Remaining history: consecutive 7-day test folds.
#
# Each fold uses ONLY earlier data to select thresholds.
# No future data is used to train an earlier prediction.
#
# Keeps the same five report filenames used by the workflow.
# ============================================================

API = "https://api.exchange.coinbase.com"
PRODUCT = "BTC-USD"

DAYS = 90
INITIAL_TRAIN_DAYS = 15
TEST_FOLD_DAYS = 7
HORIZON_MINUTES = 15

SIGNAL_MINUTES = (7, 22, 37, 52)
MAX_FILLED_GAP_MINUTES = 5
MIN_TRAINING_SIGNALS = 60
MIN_TEST_SIGNALS = 30

COSTS_BPS = (0, 5, 10, 25)

HEADERS = {
    "User-Agent": "BTC-15m-WalkForward-Backtest/1.0"
}

STRATEGIES = (
    "current",
    "contrarian",
    "mean_reversion",
    "trend",
    "momentum",
    "breakout",
    "volume_breakout",
    "combined",
)

THRESHOLDS = (
    0.00, 0.05, 0.10, 0.15, 0.20,
    0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.60, 0.70, 0.80, 0.90
)


def get_json(url):
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=40) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == 5:
                raise RuntimeError(f"Coinbase request failed: {error}")
            time.sleep(min(2 ** attempt, 20))

    raise RuntimeError("Unable to download market data.")


def repair_gaps(candles):
    if not candles:
        raise RuntimeError("No candles downloaded.")

    repaired = [candles[0]]
    filled = 0

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

        close = previous["close"]

        for minute in range(1, missing + 1):
            repaired.append({
                "time": previous["time"] + minute * 60,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 0.0,
                "synthetic": True
            })
            filled += 1

        repaired.append(current)

    return repaired, filled


def download_candles():
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=DAYS)

    rows = []
    cursor = start
    batches = 0

    print(f"Downloading {DAYS} days of BTC/USD one-minute candles.")
    print("This may take several minutes.")

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
        batches += 1

        if batches % 30 == 0:
            print(f"Downloaded {batches} batches.")

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
        c for c in candles
        if c["time"] + 60 <= cutoff
    ]

    if len(candles) < 50000:
        raise RuntimeError(
            f"Only {len(candles)} actual candles downloaded."
        )

    actual_count = len(candles)
    candles, synthetic_count = repair_gaps(candles)

    print(f"Actual candles: {actual_count}")
    print(f"Small gaps filled: {synthetic_count}")

    return candles, actual_count, synthetic_count


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

    gains = [max(x, 0.0) for x in changes]
    losses = [max(-x, 0.0) for x in changes]

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0

    rs = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_features(candles, index):
    window = candles[index - 149:index + 1]

    if len(window) != 150:
        return None

    if any(c["synthetic"] for c in window):
        return None

    closes = [c["close"] for c in window]
    highs = [c["high"] for c in window]
    lows = [c["low"] for c in window]
    volumes = [c["volume"] for c in window]

    price = closes[-1]

    if price <= 0:
        return None

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
        return None

    returns = []

    for i in range(len(closes) - 30, len(closes)):
        previous = closes[i - 1]

        if previous <= 0:
            return None

        returns.append((closes[i] / previous - 1.0) * 100.0)

    volatility = statistics.pstdev(returns)

    ema_difference = ((ema9 - ema21) / price) * 100.0
    long_trend = ((ema21 - ema50) / price) * 100.0

    recent_volume = sum(volumes[-5:]) / 5.0
    prior_volume = sum(volumes[-30:-5]) / 25.0

    volume_ratio = (
        recent_volume / prior_volume if prior_volume > 0 else 1.0
    )

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
    current_probability = 50.0 + 30.0 * math.tanh(adjusted * 2.0)
    current_signal = 1.0 if current_probability >= 50.0 else -1.0

    trend_score = (
        0.55 * math.tanh(ema_difference / 0.12)
        + 0.45 * math.tanh(long_trend / 0.20)
    )

    momentum_score = (
        0.35 * math.tanh(mom5 / 0.10)
        + 0.45 * math.tanh(mom15 / 0.20)
        + 0.20 * math.tanh(mom30 / 0.30)
    )

    mean_reversion_score = (
        -0.60 * math.tanh((current_rsi - 50.0) / 12.0)
        -0.40 * math.tanh(mom15 / 0.20)
    )

    prior_high = max(highs[-31:-1])
    prior_low = min(lows[-31:-1])

    breakout_score = 0.0

    if price > prior_high:
        breakout_score = 1.0
    elif price < prior_low:
        breakout_score = -1.0

    volume_confirmation = math.tanh((volume_ratio - 1.0) / 0.75)
    volume_breakout_score = breakout_score * volume_confirmation

    combined_score = (
        0.35 * math.tanh(ema_difference / 0.12)
        + 0.25 * math.tanh(long_trend / 0.20)
        + 0.25 * math.tanh(mom15 / 0.20)
        + 0.15 * math.tanh((current_rsi - 50.0) / 12.0)
    )

    combined_score /= (1.0 + volatility * 5.0)

    return {
        "current": current_signal,
        "contrarian": -current_signal,
        "mean_reversion": mean_reversion_score,
        "trend": trend_score,
        "momentum": momentum_score,
        "breakout": breakout_score,
        "volume_breakout": volume_breakout_score,
        "combined": combined_score,
        "rsi_value": current_rsi,
        "volatility": volatility,
        "ema_difference": ema_difference,
        "long_trend": long_trend,
        "volume_ratio": volume_ratio
    }


def build_records(candles):
    records = []
    time_index = {c["time"]: i for i, c in enumerate(candles)}

    print("Building scheduled 15-minute predictions...")

    for index in range(149, len(candles) - HORIZON_MINUTES - 1):
        signal_time = candles[index]["time"] + 60

        minute = datetime.fromtimestamp(
            signal_time, timezone.utc
        ).minute

        if minute not in SIGNAL_MINUTES:
            continue

        if signal_time % 900 != 420:
            continue

        features = calculate_features(candles, index)

        if features is None:
            continue

        entry = candles[index + 1]

        if entry["time"] != signal_time or entry["synthetic"]:
            continue

        target_time = signal_time + HORIZON_MINUTES * 60
        outcome_index = time_index.get(target_time)

        if outcome_index is None:
            continue

        outcome = candles[outcome_index]

        if outcome["synthetic"]:
            continue

        entry_price = entry["open"]
        exit_price = outcome["open"]

        if entry_price <= 0:
            continue

        direction = (
            1 if exit_price > entry_price
            else -1 if exit_price < entry_price
            else 0
        )

        record = {
            "signal_time": signal_time,
            "entry_time": signal_time,
            "target_time": target_time,
            "entry_price": entry_price,
            "outcome_price": exit_price,
            "direction": direction,
            "btc_return_percent": (
                (exit_price / entry_price - 1.0) * 100.0
            )
        }

        record.update(features)
        records.append(record)

    return records


def choose_threshold(training, strategy):
    best_threshold = 0.0
    best_accuracy = -1.0
    best_count = 0

    for threshold in THRESHOLDS:
        selected = [
            r for r in training
            if r["direction"] != 0
            and abs(r[strategy]) > 0
            and abs(r[strategy]) >= threshold
        ]

        if len(selected) < MIN_TRAINING_SIGNALS:
            continue

        wins = sum(
            1 for r in selected
            if (1 if r[strategy] > 0 else -1) == r["direction"]
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
    denominator = 1.0 + z * z / total

    center = (
        p + z * z / (2.0 * total)
    ) / denominator

    margin = (
        z * math.sqrt(
            p * (1.0 - p) / total
            + z * z / (4.0 * total * total)
        )
    ) / denominator

    return (
        round(max(0.0, center - margin) * 100.0, 2),
        round(min(1.0, center + margin) * 100.0, 2)
    )


def evaluate(records, strategy, threshold):
    selected = [
        r for r in records
        if r["direction"] != 0
        and abs(r[strategy]) > 0
        and abs(r[strategy]) >= threshold
    ]

    wins = sum(
        1 for r in selected
        if (1 if r[strategy] > 0 else -1) == r["direction"]
    )

    total = len(selected)
    low, high = wilson_interval(wins, total)

    return {
        "predictions": total,
        "wins": wins,
        "losses": total - wins,
        "accuracy_percent": round(
            100.0 * wins / total if total else 0.0, 2
        ),
        "confidence_interval_low": low,
        "confidence_interval_high": high
    }


def classify_regimes(training, testing):
    train_vol = [r["volatility"] for r in training]
    train_trend = [abs(r["ema_difference"]) for r in training]

    def percentile(values, fraction):
        ordered = sorted(values)
        if not ordered:
            return 0.0
        return ordered[int((len(ordered) - 1) * fraction)]

    low_vol = percentile(train_vol, 1 / 3)
    high_vol = percentile(train_vol, 2 / 3)
    trend_cutoff = percentile(train_trend, 2 / 3)

    for record in testing:
        if record["volatility"] <= low_vol:
            record["volatility_regime"] = "low"
        elif record["volatility"] <= high_vol:
            record["volatility_regime"] = "medium"
        else:
            record["volatility_regime"] = "high"

        if abs(record["ema_difference"]) >= trend_cutoff:
            record["trend_regime"] = (
                "uptrend" if record["ema_difference"] > 0 else "downtrend"
            )
        else:
            record["trend_regime"] = "sideways"

    return {
        "low_volatility_boundary": low_vol,
        "high_volatility_boundary": high_vol,
        "trend_strength_boundary": trend_cutoff
    }


def aggregate_results(records, strategy, threshold):
    result = evaluate(records, strategy, threshold)
    result["strategy"] = strategy
    result["training_threshold"] = threshold
    return result


def make_cost_report(records, threshold):
    selected = [
        r for r in records
        if r["direction"] != 0
        and abs(r["mean_reversion"]) > 0
        and abs(r["mean_reversion"]) >= threshold
    ]

    output = []

    for cost_bps in COSTS_BPS:
        returns = []

        for record in selected:
            predicted = 1 if record["mean_reversion"] > 0 else -1
            returns.append(record["btc_return_percent"] * predicted)

        gross = statistics.mean(returns) if returns else 0.0
        cost_percent = cost_bps / 100.0
        net = gross - cost_percent

        positive = sum(
            1 for value in returns
            if value - cost_percent > 0
        )

        output.append({
            "strategy": "mean_reversion",
            "predictions": len(selected),
            "assumed_round_trip_cost_bps": cost_bps,
            "average_gross_directional_return_percent": round(gross, 5),
            "average_net_directional_return_percent": round(net, 5),
            "positive_net_trade_percent": round(
                100.0 * positive / len(selected) if selected else 0.0,
                2
            )
        })

    return output


def write_csv(filename, rows, fields):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 72)
    print("BTC 15-MINUTE WALK-FORWARD BACKTEST")
    print("=" * 72)

    candles, actual_count, synthetic_count = download_candles()
    records = build_records(candles)

    if len(records) < 1000:
        raise RuntimeError(
            f"Only {len(records)} valid signals found. "
            "At least 1,000 are required."
        )

    first_time = candles[0]["time"]
    last_time = candles[-1]["time"]
    initial_train_end = first_time + INITIAL_TRAIN_DAYS * 86400

    folds = []
    fold_start = initial_train_end
    fold_number = 0

    while fold_start + TEST_FOLD_DAYS * 86400 <= last_time:
        fold_end = fold_start + TEST_FOLD_DAYS * 86400
        fold_number += 1

        training = [
            r for r in records
            if r["signal_time"] < fold_start
            and r["target_time"] <= fold_start
        ]

        testing = [
            r for r in records
            if r["signal_time"] >= fold_start
            and r["target_time"] <= fold_end
        ]

        if len(training) >= 300 and len(testing) >= MIN_TEST_SIGNALS:
            folds.append({
                "fold": fold_number,
                "start": fold_start,
                "end": fold_end,
                "training": training,
                "testing": testing
            })

        fold_start = fold_end

    if len(folds) < 5:
        raise RuntimeError(
            f"Only {len(folds)} usable test folds. "
            "At least 5 are required."
        )

    all_test_records = []
    fold_reports = []
    strategy_fold_rows = []
    regime_rows = []
    fold_thresholds = {strategy: [] for strategy in STRATEGIES}

    print(f"\nUsable walk-forward test folds: {len(folds)}")

    for fold in folds:
        number = fold["fold"]
        training = fold["training"]
        testing = fold["testing"]

        thresholds = {}

        for strategy in STRATEGIES:
            if strategy in ("current", "contrarian"):
                thresholds[strategy] = 0.0
            else:
                thresholds[strategy] = choose_threshold(
                    training, strategy
                )

            fold_thresholds[strategy].append(thresholds[strategy])

        boundaries = classify_regimes(training, testing)

        for record in testing:
            record["fold"] = number

        all_test_records.extend(testing)

        fold_summary = {
            "fold": number,
            "training_signals": len(training),
            "testing_signals": len(testing),
            "test_start_utc": datetime.fromtimestamp(
                fold["start"], timezone.utc
            ).isoformat(),
            "test_end_utc": datetime.fromtimestamp(
                fold["end"], timezone.utc
            ).isoformat(),
            "thresholds": thresholds,
            "regime_boundaries": boundaries
        }

        fold_reports.append(fold_summary)

        for strategy in STRATEGIES:
            result = evaluate(
                testing, strategy, thresholds[strategy]
            )

            strategy_fold_rows.append({
                "fold": number,
                "strategy": strategy,
                **result,
                "training_threshold": thresholds[strategy]
            })

        mr_threshold = thresholds["mean_reversion"]

        for regime_type in ("volatility_regime", "trend_regime"):
            labels = sorted(set(r[regime_type] for r in testing))

            for label in labels:
                subset = [
                    r for r in testing if r[regime_type] == label
                ]

                result = evaluate(
                    subset, "mean_reversion", mr_threshold
                )

                regime_rows.append({
                    "strategy": "mean_reversion",
                    "fold": number,
                    "regime_type": regime_type,
                    "regime": label,
                    **result,
                    "training_threshold": mr_threshold
                })

        print(
            f"Fold {number}: "
            f"{len(training)} training signals, "
            f"{len(testing)} test signals"
        )

    # Aggregate all genuinely out-of-sample predictions.
    aggregate = []

    for strategy in STRATEGIES:
        selected_thresholds = fold_thresholds[strategy]
        average_threshold = (
            statistics.mean(selected_thresholds)
            if selected_thresholds else 0.0
        )

        wins = 0
        total = 0

        for row in strategy_fold_rows:
            if row["strategy"] == strategy:
                wins += row["wins"]
                total += row["predictions"]

        losses = total - wins
        low, high = wilson_interval(wins, total)

        aggregate.append({
            "strategy": strategy,
            "predictions": total,
            "wins": wins,
            "losses": losses,
            "accuracy_percent": round(
                100.0 * wins / total if total else 0.0, 2
            ),
            "confidence_interval_low": low,
            "confidence_interval_high": high,
            "training_threshold": round(average_threshold, 4)
        })

    # Cost analysis uses each fold's own training-selected threshold.
    mr_selected = []

    for fold in folds:
        threshold = choose_threshold(
            fold["training"], "mean_reversion"
        )

        mr_selected.extend([
            r for r in fold["testing"]
            if r["direction"] != 0
            and abs(r["mean_reversion"]) > 0
            and abs(r["mean_reversion"]) >= threshold
        ])

    # Preserve the per-fold threshold logic for the cost report.
    cost_report = []
    for cost_bps in COSTS_BPS:
        signed_returns = []

        for record in mr_selected:
            predicted = 1 if record["mean_reversion"] > 0 else -1
            signed_returns.append(
                record["btc_return_percent"] * predicted
            )

        gross = (
            statistics.mean(signed_returns)
            if signed_returns else 0.0
        )
        cost_percent = cost_bps / 100.0
        positive = sum(
            1 for value in signed_returns
            if value - cost_percent > 0
        )

        cost_report.append({
            "strategy": "mean_reversion",
            "predictions": len(mr_selected),
            "assumed_round_trip_cost_bps": cost_bps,
            "average_gross_directional_return_percent": round(gross, 5),
            "average_net_directional_return_percent": round(
                gross - cost_percent, 5
            ),
            "positive_net_trade_percent": round(
                100.0 * positive / len(signed_returns)
                if signed_returns else 0.0, 2
            )
        })

    detail_fields = [
        "fold", "signal_time", "entry_time", "target_time",
        "entry_price", "outcome_price", "direction",
        "btc_return_percent", "current", "contrarian",
        "mean_reversion", "trend", "momentum",
        "breakout", "volume_breakout", "combined",
        "rsi_value", "volatility", "ema_difference",
        "long_trend", "volume_ratio",
        "volatility_regime", "trend_regime"
    ]

    strategy_fields = [
        "strategy", "predictions", "wins", "losses",
        "accuracy_percent", "confidence_interval_low",
        "confidence_interval_high", "training_threshold"
    ]

    regime_fields = [
        "strategy", "fold", "regime_type", "regime",
        "predictions", "wins", "losses", "accuracy_percent",
        "confidence_interval_low", "confidence_interval_high",
        "training_threshold"
    ]

    cost_fields = [
        "strategy", "predictions", "assumed_round_trip_cost_bps",
        "average_gross_directional_return_percent",
        "average_net_directional_return_percent",
        "positive_net_trade_percent"
    ]

    write_csv(
        "backtest_predictions.csv",
        all_test_records,
        detail_fields
    )

    # Aggregate results first, then include fold rows for transparency.
    strategy_output = aggregate + [
        {
            "strategy": f"fold_{row['fold']}_{row['strategy']}",
            "predictions": row["predictions"],
            "wins": row["wins"],
            "losses": row["losses"],
            "accuracy_percent": row["accuracy_percent"],
            "confidence_interval_low": row["confidence_interval_low"],
            "confidence_interval_high": row["confidence_interval_high"],
            "training_threshold": row["training_threshold"]
        }
        for row in strategy_fold_rows
    ]

    write_csv(
        "backtest_strategy_results.csv",
        strategy_output,
        strategy_fields
    )

    write_csv(
        "backtest_regime_results.csv",
        regime_rows,
        regime_fields
    )

    write_csv(
        "backtest_cost_results.csv",
        cost_report,
        cost_fields
    )

    best = max(
        aggregate,
        key=lambda x: x["accuracy_percent"]
    )

    report = {
        "product": PRODUCT,
        "backtest_type": "walk_forward",
        "requested_historical_days": DAYS,
        "actual_historical_days": round(
            (last_time - first_time) / 86400.0, 2
        ),
        "initial_training_days": INITIAL_TRAIN_DAYS,
        "test_fold_days": TEST_FOLD_DAYS,
        "number_of_test_folds": len(folds),
        "horizon_minutes": HORIZON_MINUTES,
        "signal_schedule_utc": [":07", ":22", ":37", ":52"],
        "entry": "Scheduled one-minute candle open",
        "exit": "One-minute candle open exactly 15 minutes later",
        "actual_candles": actual_count,
        "synthetic_candles": synthetic_count,
        "out_of_sample_signals": len(all_test_records),
        "folds": fold_reports,
        "aggregate_strategy_results": aggregate,
        "fold_strategy_results": strategy_fold_rows,
        "mean_reversion_regime_results": regime_rows,
        "mean_reversion_hypothetical_cost_results": cost_report,
        "highest_accuracy_candidate": best,
        "important_limitations": [
            "Thresholds are selected using only data before each test fold.",
            "The reported aggregate combines multiple historical test periods.",
            "The strategy with the highest historical accuracy is not guaranteed to remain best.",
            "Confidence intervals do not fully account for time dependence or strategy selection.",
            "Hypothetical BTC directional costs are not Polymarket fees.",
            "Directional BTC return is not the same as Polymarket contract profit.",
            "Accuracy does not guarantee profitability.",
            "A live paper-trading period is still needed before risking money."
        ]
    }

    with open("backtest_results.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nAGGREGATE WALK-FORWARD RESULTS")
    print("-" * 88)

    for result in aggregate:
        print(
            f"{result['strategy']:<19}"
            f"{result['predictions']:>8} signals  "
            f"{result['accuracy_percent']:>6.2f}%  "
            f"({result['wins']} wins / {result['losses']} losses)"
        )

    print("\nREPORTS SAVED")
    print("backtest_results.json")
    print("backtest_predictions.csv")
    print("backtest_strategy_results.csv")
    print("backtest_regime_results.csv")
    print("backtest_cost_results.csv")
    print("\nWALK-FORWARD BACKTEST COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nBACKTEST ERROR: {error}")
        raise
