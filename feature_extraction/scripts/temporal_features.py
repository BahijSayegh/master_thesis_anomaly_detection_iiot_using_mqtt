import math
import numpy as np
from numba import jit
from typing import Dict, List, Any, Tuple


@jit(nopython=True)
def calculate_linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Calculate linear regression slope and intercept using optimized numba implementation.
    Uses the formula: slope = (n∑xy - ∑x∑y) / (n∑x² - (∑x)²)

    Args:
        x: Independent variable array (typically time points)
        y: Dependent variable array (metric values)

    Returns:
        Tuple[float, float]: (slope, intercept)
    """

    n = len(x)
    if n < 2:
        return 0.0, 0.0

    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0

    # Single pass calculation
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
        sum_xy += x[i] * y[i]
        sum_x2 += x[i] * x[i]

    # Calculate slope
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.0, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denominator

    # Calculate intercept
    intercept = (sum_y - slope * sum_x) / n

    return float(slope), float(intercept)


@jit(nopython=True)
def calculate_ema(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Calculate Exponential Moving Average with numba optimization.
    Uses the formula: EMA_t = α * value_t + (1 - α) * EMA_{t-1}

    Args:
        values: Array of values to calculate EMA
        alpha: Smoothing factor (default: 0.1)

    Returns:
        np.ndarray: Array of EMA values
    """

    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.float64)  # Return empty array with explicit type

    ema = np.zeros(n, dtype=np.float64)
    ema[0] = values[0]  # Initialize with first value

    # Single pass EMA calculation
    for i in range(1, n):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]

    return ema


@jit(nopython=True)
def calculate_rolling_variance(values: np.ndarray, window_size: int = 20) -> np.ndarray:
    """
    Calculate rolling variance using parallel processing with numba.
    Implements optimized rolling variance calculation using Welford's online algorithm.

    Args:
        values: Array of values
        window_size: Size of rolling window (default: 20)

    Returns:
        np.ndarray: Array of rolling variances
    """

    n = len(values)
    variances = np.zeros(n, dtype=np.float64)

    if n < 2:
        return variances

    # Parallel processing for window calculations
    for i in range(window_size - 1, n):
        window = values[max(0, i - window_size + 1):i + 1]
        mean = 0.0
        m2 = 0.0
        count = 0

        # Welford's online variance algorithm
        for x in window:
            count += 1
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            m2 += delta * delta2

        if count > 1:
            variances[i] = m2 / (count - 1)

    return variances


@jit(nopython=True)
def calculate_peak_to_average(values: np.ndarray, window_size: int = 20) -> np.ndarray:
    """
    Calculate Peak-to-Average Ratio (PAR) within rolling windows.
    PAR = max_value / mean_value for each window

    Args:
        values: Array of values
        window_size: Size of rolling window (default: 20)

    Returns:
        np.ndarray: Array of PAR values
    """

    n = len(values)
    par = np.zeros(n, dtype=np.float64)

    if n < window_size:
        return par

    # Using prange for parallel processing
    for i in range(window_size - 1, n):
        window = values[i - window_size + 1:i + 1]
        window_sum = 0.0
        window_max = window[0]

        # Manual calculation for thread safety
        for j in range(len(window)):
            window_sum += window[j]
            if window[j] > window_max:
                window_max = window[j]

        window_mean = window_sum / len(window)

        if window_mean > 0:
            par[i] = window_max / window_mean

    return par


@jit(nopython=True)
def calculate_packet_arrival_rate(timestamps: np.ndarray, window_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and maximum packet arrival rates within rolling windows.
    """

    n = len(timestamps)
    mean_rates = np.zeros(n, dtype=np.float64)
    max_rates = np.zeros(n, dtype=np.float64)

    if n < window_size:
        return mean_rates, max_rates

    # Use prange for parallel processing of windows
    for i in range(window_size - 1, n):
        window = timestamps[i - window_size + 1:i + 1]
        duration = window[-1] - window[0]

        if duration > 0:
            sub_rates = np.zeros(len(window) - 1, dtype=np.float64)
            sub_rates_count = 0

            # This inner loop remains serial as it has dependencies
            for j in range(len(window) - 1):
                sub_duration = window[j + 1] - window[j]
                if sub_duration > 0:
                    sub_rates[sub_rates_count] = 1.0 / sub_duration
                    sub_rates_count += 1

            if sub_rates_count > 0:
                valid_rates = sub_rates[:sub_rates_count]
                mean_rates[i] = np.sum(valid_rates) / sub_rates_count
                max_rates[i] = np.max(valid_rates)

    return mean_rates, max_rates


@jit(nopython=True)
def calculate_time_gaps(timestamps: np.ndarray, window_size: int = 20) -> np.ndarray:
    """
    Calculate statistical measure of time gaps between events.
    Uses coefficient of variation of inter-arrival times as a measure of timing regularity.

    Args:
        timestamps: Array of packet timestamps
        window_size: Size of rolling window (default: 20)

    Returns:
        np.ndarray: Array of time gap metrics
    """

    n = len(timestamps)
    gaps = np.zeros(n, dtype=np.float64)
    if n < window_size:
        return gaps

    # Using prange for parallel processing
    for i in range(window_size - 1, n):
        window = timestamps[i - window_size + 1:i + 1]
        inter_arrival = np.zeros(len(window) - 1, dtype=np.float64)

        # Calculate inter-arrival times
        for j in range(len(window) - 1):
            inter_arrival[j] = window[j + 1] - window[j]
        if len(inter_arrival) > 0:

            # Manual calculations for thread safety
            sum_val = 0.0
            for val in inter_arrival:
                sum_val += val
            mean = sum_val / len(inter_arrival)

            if mean > 0:
                # Calculate variance manually
                squared_diff_sum = 0.0
                for val in inter_arrival:
                    diff = val - mean
                    squared_diff_sum += diff * diff
                std = np.sqrt(squared_diff_sum / len(inter_arrival))
                gaps[i] = std / mean  # Coefficient of variation

    return gaps


def extract_temporal_arrays(window: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract arrays needed for temporal analysis from window.

    Args:
        window: List of packet dictionaries

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (timestamps, sizes, inter_arrival_times)
    """

    if not window:
        return (np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64))
    n = len(window)
    timestamps = np.zeros(n, dtype=np.float64)
    sizes = np.zeros(n, dtype=np.float64)

    for i in range(n):
        timestamps[i] = float(window[i]['time'])
        sizes[i] = float(window[i]['length'])

    if n > 1:
        inter_arrival = np.zeros(n - 1, dtype=np.float64)
        for i in range(n - 1):
            inter_arrival[i] = timestamps[i + 1] - timestamps[i]

    else:
        inter_arrival = np.zeros(0, dtype=np.float64)

    return timestamps, sizes, inter_arrival


def calculate_temporal_features(window: List[Dict[str, Any]], window_size: int) -> Dict[str, Any]:
    """
    Calculate all temporal features for a given window.

    Args:
        window: List of packet dictionaries
        window_size: Size of window (100 or 500)

    Returns:
        Dict[str, Any]: Dictionary of calculated temporal features
    """

    if not window:
        return {}
    suffix = f"_{window_size}"
    try:
        # Extract arrays for calculations
        timestamps, sizes, inter_arrival = extract_temporal_arrays(window)
        if len(timestamps) == 0:
            return {
                f"moving_linear_regression_slope{suffix}": 0.0,
                f"exponential_moving_average{suffix}": 0.0,
                f"rolling_variance_inter_arrival_times{suffix}": 0.0,
                f"peak_to_average_ratio{suffix}": 0.0,
                f"packet_arrival_rate_mean{suffix}": 0.0,
                f"packet_arrival_rate_max{suffix}": 0.0,
                f"time_gaps_between_events{suffix}": 0.0
            }

        # Time points for regression (normalized to start at 0)
        time_points = timestamps - timestamps[0]

        # Calculate basic temporal features
        slope, _ = calculate_linear_regression(time_points, sizes)
        ema_values = calculate_ema(sizes)
        rolling_var = calculate_rolling_variance(inter_arrival)
        par_values = calculate_peak_to_average(sizes)
        mean_rates, max_rates = calculate_packet_arrival_rate(timestamps)
        time_gap_metrics = calculate_time_gaps(timestamps)

        # Use safe mean calculation for arrays
        features = {
            f"moving_linear_regression_slope{suffix}": float(slope),
            f"exponential_moving_average{suffix}": float(ema_values[-1]) if len(ema_values) > 0 else 0.0,
            f"rolling_variance_inter_arrival_times{suffix}": float(np.sum(rolling_var) / len(rolling_var)) if len(
                rolling_var) > 0 else 0.0,
            f"peak_to_average_ratio{suffix}": float(np.sum(par_values) / len(par_values)) if len(
                par_values) > 0 else 0.0,
            f"packet_arrival_rate_mean{suffix}": float(np.sum(mean_rates) / len(mean_rates)) if len(
                mean_rates) > 0 else 0.0,
            f"packet_arrival_rate_max{suffix}": float(np.max(max_rates)) if len(max_rates) > 0 else 0.0,
            f"time_gaps_between_events{suffix}": float(np.sum(time_gap_metrics) / len(time_gap_metrics)) if len(
                time_gap_metrics) > 0 else 0.0
        }

        return features

    except Exception as e:
        print(f"Error in temporal feature extraction: {e}")
        return {}


def process_windows(small_window: List[Dict[str, Any]],
                    large_window: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process both windows to extract temporal features.

    Args:
        small_window: 100-packet window
        large_window: 500-packet window

    Returns:
        Dict[str, Any]: Combined features from both windows
    """

    small_features = calculate_temporal_features(small_window, 100)
    large_features = calculate_temporal_features(large_window, 500)
    return {**small_features, **large_features}


def verify_temporal_features(features: Dict[str, Any]) -> bool:
    """
    Verify all required temporal features are present and valid.

    Args:
        features: Dictionary of extracted features

    Returns:
        bool: True if all features are valid
    """

    required_suffixes = ['_100', '_500']
    required_base_features = [
        'moving_linear_regression_slope',
        'exponential_moving_average',
        'rolling_variance_inter_arrival_times',
        'peak_to_average_ratio',
        'packet_arrival_rate_mean',
        'packet_arrival_rate_max',
        'time_gaps_between_events'
    ]

    for suffix in required_suffixes:
        for base in required_base_features:
            feature_name = f"{base}{suffix}"
            if feature_name not in features:
                print(f"Missing feature: {feature_name}")
                return False

            value = features[feature_name]

            if not isinstance(value, (int, float)):
                print(f"Invalid feature type for {feature_name}: {type(value)}")
                return False

            if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
                print(f"Invalid feature value for {feature_name}: {value}")
                return False

    return True
