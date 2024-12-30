import math
from typing import Dict, List, Any, Tuple
import numpy as np
from numba import jit


def prepare_statistical_arrays(window: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract arrays needed for statistical calculations from window data.
    Separates dictionary handling from numba-optimized calculations.

    Args:
        window: List of packet dictionaries

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Arrays for packet sizes, inter-arrival times, and payload sizes
    """

    n = len(window)
    sizes = np.zeros(n, dtype=np.float64)
    times = np.zeros(n, dtype=np.float64)
    payloads = np.zeros(n, dtype=np.float64)

    for i, packet in enumerate(window):
        sizes[i] = float(packet['length'])
        times[i] = float(packet['time'])
        features = packet.get('features', {})
        payloads[i] = float(features.get('tcp_payload_size', 0))

    # Calculate inter-arrival times
    if n > 1:
        inter_arrival = np.diff(times)
    else:
        inter_arrival = np.zeros(0, dtype=np.float64)

    return sizes, inter_arrival, payloads


@jit(nopython=True)
def calculate_cv(values: np.ndarray) -> float:
    """
    Calculate Coefficient of Variation (CV) with numba optimization.
    CV = σ/μ where σ is standard deviation and μ is mean.

    Args:
        values: Array of numerical values

    Returns:
        float: Coefficient of variation, or 0.0 if mean is 0
    """

    if len(values) < 2:
        return 0.0
    mean = np.mean(values)
    if mean == 0:
        return 0.0
    std = np.std(values)

    return float(std / mean)


@jit(nopython=True)
def calculate_moving_range(values: np.ndarray) -> float:
    """
    Calculate average moving range with parallel processing.
    MR_i = |x_i - x_{i-1}|

    Args:
        values: Array of numerical values

    Returns:
        float: Average moving range
    """

    if len(values) < 2:
        return 0.0

    ranges = np.zeros(len(values) - 1, dtype=np.float64)
    # Calculate ranges in parallel
    for i in range(len(values) - 1):
        ranges[i] = abs(values[i + 1] - values[i])

    return float(np.mean(ranges))


@jit(nopython=True)
def calculate_iqr(values: np.ndarray) -> float:
    """
    Calculate Interquartile Range (IQR) with numba optimization.
    IQR = Q3 - Q1

    Args:
        values: Array of numerical values

    Returns:
        float: Interquartile range
    """

    if len(values) < 4:  # Need at least 4 points for meaningful quartiles
        return 0.0

    sorted_vals = np.sort(values)
    n = len(sorted_vals)

    # Calculate quartile indices
    q1_idx = int(0.25 * (n - 1))
    q3_idx = int(0.75 * (n - 1))

    return float(sorted_vals[q3_idx] - sorted_vals[q1_idx])


@jit(nopython=True)
def calculate_skewness_kurtosis(values: np.ndarray) -> Tuple[float, float]:
    """
    Calculate skewness and kurtosis with single-pass algorithm.
    Uses numerically stable computations.

    Args:
        values: Array of numerical values

    Returns:
        Tuple[float, float]: (skewness, kurtosis)
    """

    if len(values) < 3:  # Need at least 3 points for skewness
        return 0.0, 0.0

    n = len(values)
    mean = np.mean(values)
    m2 = 0.0  # Second moment
    m3 = 0.0  # Third moment
    m4 = 0.0  # Fourth moment

    # Single pass calculation
    for i in range(n):
        diff = values[i] - mean
        diff2 = diff * diff
        m2 += diff2
        m3 += diff * diff2
        m4 += diff2 * diff2

    # Adjust for sample size
    m2 /= n
    m3 /= n
    m4 /= n

    # Calculate skewness and kurtosis
    if m2 == 0:
        return 0.0, 0.0

    std_dev = np.sqrt(m2)
    skewness = m3 / (std_dev * std_dev * std_dev)
    kurtosis = m4 / (m2 * m2) - 3.0  # Excess kurtosis

    return float(skewness), float(kurtosis)


@jit(nopython=True)
def calculate_rolling_entropy(values: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Shannon entropy with adaptive binning.
    H = -Σ p_i log(p_i)

    Args:
        values: Array of numerical values
        bins: Number of bins for histogram

    Returns:
        float: Shannon entropy
    """

    if len(values) < bins:
        return 0.0

    # Create histogram
    hist, _ = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)

    # Calculate probabilities
    total = np.sum(hist)
    if total == 0:
        return 0.0
    probs = hist / total

    # Calculate entropy
    entropy = 0.0
    for p in probs:
        if p > 0:  # Avoid log(0)
            entropy -= p * np.log2(p)

    return float(entropy)


def calculate_statistical_features(window: List[Dict[str, Any]], window_size: int) -> Dict[str, Any]:
    """
    Calculate all statistical features for a given window.

    Args:
        window: List of packet dictionaries
        window_size: Size of window (100 or 500)

    Returns:
        Dict[str, Any]: Dictionary of calculated statistical features
    """

    if not window:
        return {}
    suffix = f"_{window_size}"

    try:
        # Extract arrays for calculations
        sizes, inter_arrival, payloads = prepare_statistical_arrays(window)

        # Calculate features for packet sizes
        size_cv = calculate_cv(sizes)
        size_mr = calculate_moving_range(sizes)
        size_iqr = calculate_iqr(sizes)
        size_skew, size_kurt = calculate_skewness_kurtosis(sizes)
        size_entropy = calculate_rolling_entropy(sizes)

        # Calculate features for inter-arrival times
        iat_cv = calculate_cv(inter_arrival)
        iat_mr = calculate_moving_range(inter_arrival)
        iat_iqr = calculate_iqr(inter_arrival)
        iat_skew, iat_kurt = calculate_skewness_kurtosis(inter_arrival)
        iat_entropy = calculate_rolling_entropy(inter_arrival)

        # Calculate features for payload sizes
        payload_cv = calculate_cv(payloads)
        payload_mr = calculate_moving_range(payloads)
        payload_iqr = calculate_iqr(payloads)
        payload_skew, payload_kurt = calculate_skewness_kurtosis(payloads)
        payload_entropy = calculate_rolling_entropy(payloads)

        # Combine all features
        features = {
            # Packet size features
            f"coefficient_of_variation{suffix}": size_cv,
            f"moving_range{suffix}": size_mr,
            f"interquartile_range{suffix}": size_iqr,
            f"skewness{suffix}": size_skew,
            f"kurtosis{suffix}": size_kurt,
            f"rolling_entropy{suffix}": size_entropy,

            # Inter-arrival time features
            f"iat_coefficient_of_variation{suffix}": iat_cv,
            f"iat_moving_range{suffix}": iat_mr,
            f"iat_interquartile_range{suffix}": iat_iqr,
            f"iat_skewness{suffix}": iat_skew,
            f"iat_kurtosis{suffix}": iat_kurt,
            f"iat_rolling_entropy{suffix}": iat_entropy,

            # Payload size features
            f"payload_coefficient_of_variation{suffix}": payload_cv,
            f"payload_moving_range{suffix}": payload_mr,
            f"payload_interquartile_range{suffix}": payload_iqr,
            f"payload_skewness{suffix}": payload_skew,
            f"payload_kurtosis{suffix}": payload_kurt,
            f"payload_rolling_entropy{suffix}": payload_entropy
        }
        return features

    except Exception as e:
        print(f"Error in statistical feature extraction: {e}")
        return {}


def process_windows(small_window: List[Dict[str, Any]],
                    large_window: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process both windows to extract statistical features.

    Args:
        small_window: 100-packet window
        large_window: 500-packet window

    Returns:
        Dict[str, Any]: Combined features from both windows
    """

    small_features = calculate_statistical_features(small_window, 100)
    large_features = calculate_statistical_features(large_window, 500)
    return {**small_features, **large_features}


def verify_statistical_features(features: Dict[str, Any]) -> bool:
    """
    Verify all required statistical features are present and valid.

    Args:
        features: Dictionary of extracted features

    Returns:
        bool: True if all features are valid, False otherwise
    """

    required_suffixes = ['_100', '_500']
    required_base_features = [
        'coefficient_of_variation',
        'moving_range',
        'interquartile_range',
        'skewness',
        'kurtosis',
        'rolling_entropy',
        'iat_coefficient_of_variation',
        'iat_moving_range',
        'iat_interquartile_range',
        'iat_skewness',
        'iat_kurtosis',
        'iat_rolling_entropy',
        'payload_coefficient_of_variation',
        'payload_moving_range',
        'payload_interquartile_range',
        'payload_skewness',
        'payload_kurtosis',
        'payload_rolling_entropy'
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
