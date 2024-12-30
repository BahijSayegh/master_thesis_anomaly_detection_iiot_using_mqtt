import math
from typing import Dict, List, Any, Tuple
import numpy as np
from numba import jit
from scipy import stats
from config.extractor_config import SILENCE_THRESHOLD, BURST_THRESHOLD

@jit(nopython=True)
def extract_window_arrays(window: List[Dict[str, Any]]
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract all arrays in a single pass through the window.

    Args:
        window: List of packet dictionaries

    Returns:
        Tuple of arrays: (packet_sizes, timestamps, inter_arrival_times, tcp_flags, mqtt_flags)
    """

    n = len(window)
    sizes = np.empty(n, dtype=np.float64)
    times = np.empty(n, dtype=np.float64)
    tcp_flags = np.zeros((n, 8), dtype=np.int32)  # For 8 TCP flags
    mqtt_flags = np.zeros(n, dtype=np.int32)
    for i in range(n):
        packet = window[i]
        features = packet.get('features', {})
        sizes[i] = packet['length']
        times[i] = packet['time']

        # TCP flags in single array
        tcp_flags[i, 0] = features.get('tcp_syn_flag', 0)
        tcp_flags[i, 1] = features.get('tcp_ack_flag', 0)
        tcp_flags[i, 2] = features.get('tcp_fin_flag', 0)
        tcp_flags[i, 3] = features.get('tcp_rst_flag', 0)
        tcp_flags[i, 4] = features.get('tcp_psh_flag', 0)
        tcp_flags[i, 5] = features.get('tcp_urg_flag', 0)
        tcp_flags[i, 6] = features.get('tcp_ece_flag', 0)
        tcp_flags[i, 7] = features.get('tcp_cwr_flag', 0)
        mqtt_flags[i] = features.get('is_mqtt', 0)

    # Vectorized inter-arrival time calculation
    inter_arrival = np.diff(times)
    return sizes, times, inter_arrival, tcp_flags, mqtt_flags


@jit(nopython=True)
def calculate_basic_stats(values: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calculate basic statistical measures using parallel processing.

    Args:
        values: Array of values

    Returns:
        (mean, variance, std_dev, range)
    """

    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0

    total = 0.0
    for i in range(len(values)):
        total += values[i]
    mean = total / len(values)

    squared_diff_sum = 0.0
    for i in range(len(values)):
        diff = values[i] - mean
        squared_diff_sum += diff * diff

    variance = squared_diff_sum / len(values)
    std_dev = np.sqrt(variance)
    value_range = np.ptp(values)

    return mean, variance, std_dev, value_range


@jit(nopython=True)
def calculate_percentiles(values: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate percentiles using optimized algorithm.

    Args:
        values: Array of values

    Returns:
        (25th, 50th, 75th percentiles)
    """

    if len(values) == 0:
        return 0.0, 0.0, 0.0

    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    idx_25 = max(0, min(n - 1, int(0.25 * (n - 1))))
    idx_50 = max(0, min(n - 1, int(0.50 * (n - 1))))
    idx_75 = max(0, min(n - 1, int(0.75 * (n - 1))))

    # Explicit conversion to float
    return float(sorted_vals[idx_25]), float(sorted_vals[idx_50]), float(sorted_vals[idx_75])


@jit(nopython=True)
def calculate_entropy(values: np.ndarray) -> float:
    """
    Calculate entropy using pre-binned data.

    Args:
        values: Array of values

    Returns:
        Entropy value
    """

    if len(values) == 0:
        return 0.0

    hist, _ = np.histogram(values, bins=50)
    probs = hist / hist.sum()

    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy


def generate_flow_id(packet: Dict[str, Any]) -> str:
    """
    Generate unique flow identifier.

    Args:
        packet: Packet dictionary

    Returns:
        Flow identifier string
    """

    features = packet.get('features', {})
    src_ip = features.get('source_ip_address', '')
    dst_ip = features.get('destination_ip_address', '')
    src_port = features.get('source_port', 0)
    dst_port = features.get('destination_port', 0)
    is_mqtt = features.get('is_mqtt', 0)

    if f"{src_ip}:{src_port}" < f"{dst_ip}:{dst_port}":
        flow_tuple = (src_ip, dst_ip, src_port, dst_port, is_mqtt)

    else:
        flow_tuple = (dst_ip, src_ip, dst_port, src_port, is_mqtt)

    return f"flow_{hash(flow_tuple)}"


def prepare_window_arrays(window: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract arrays from window dictionaries before numba processing.
    Not decorated with @jit because it handles dictionaries.

    Args:
        window: List of packet dictionaries

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (packet_sizes, timestamps, inter_arrival_times)
    """

    n = len(window)

    # Explicit array types
    sizes = np.zeros(n, dtype=np.float64)
    times = np.zeros(n, dtype=np.float64)

    for i in range(n):
        packet = window[i]
        sizes[i] = float(packet['length'])
        times[i] = float(packet['time'])

    # Calculate inter-arrival times
    if n > 1:
        inter_arrival = np.diff(times)
    else:
        inter_arrival = np.zeros(0, dtype=np.float64)
    # Ensure consistent types

    return (sizes.astype(np.float64),
            times.astype(np.float64),
            inter_arrival.astype(np.float64))


@jit(nopython=True)
def process_arrays(sizes: np.ndarray, times: np.ndarray,
                   inter_arrival: np.ndarray) -> Tuple[float, float, float, float,
                                                       float, float, float, float,
                                                       float, float, float, float,
                                                       float, float, float, float,
                                                       float, int, int]:
    """
    Process numeric arrays with numba optimization.
    Only handles numpy arrays, no dictionaries.

    Returns:
        Tuple of:
        - mean_size, var_size, std_size, range_size
        - size_25th, size_50th, size_75th
        - mean_iat, var_iat, std_iat, range_iat
        - iat_25th, iat_50th, iat_75th
        - duration, packet_rate, byte_rate
        - bursts, silences
    """

    # Calculate statistics with explicit type conversion
    mean_size, var_size, std_size, range_size = calculate_basic_stats(sizes)
    size_25th, size_50th, size_75th = calculate_percentiles(sizes)
    mean_iat, var_iat, std_iat, range_iat = calculate_basic_stats(inter_arrival)
    iat_25th, iat_50th, iat_75th = calculate_percentiles(inter_arrival)

    # Flow metrics with explicit type conversion
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    packet_rate = float(len(sizes)) / duration if duration > 0 else 0.0
    byte_rate = float(np.sum(sizes)) / duration if duration > 0 else 0.0

    # Count bursts and silences with explicit type conversion
    bursts = int(np.sum(inter_arrival < BURST_THRESHOLD))
    silences = int(np.sum(inter_arrival > SILENCE_THRESHOLD))

    return (float(mean_size), float(var_size), float(std_size), float(range_size),
            float(size_25th), float(size_50th), float(size_75th),
            float(mean_iat), float(var_iat), float(std_iat), float(range_iat),
            float(iat_25th), float(iat_50th), float(iat_75th),
            float(duration), float(packet_rate), float(byte_rate),
            int(bursts), int(silences))


def extract_flow_features(window: List[Dict[str, Any]],
                          window_size: int) -> Dict[str, Any]:
    """
    Main feature extraction function - handles dictionaries and calls optimized functions.
    """

    if not window:
        return {}

    suffix = f"_{window_size}"
    flow_id = generate_flow_id(window[-1])

    # Ensure correct unpacking of arrays
    try:
        arrays = prepare_window_arrays(window)
        sizes, times, inter_arrival = arrays

        # Process arrays
        results = process_arrays(sizes, times, inter_arrival)
        (mean_size, var_size, std_size, range_size,
         size_25th, size_50th, size_75th,
         mean_iat, var_iat, std_iat, range_iat,
         iat_25th, iat_50th, iat_75th,
         duration, packet_rate, byte_rate,
         bursts, silences) = results

        features = {
            f"flow_id{suffix}": flow_id,
            f"flow_duration{suffix}": duration,
            f"total_packets{suffix}": int(len(window)),
            f"total_bytes{suffix}": int(np.sum(sizes)),
            f"packet_rate{suffix}": packet_rate,
            f"byte_rate{suffix}": byte_rate,
            f"average_packet_size{suffix}": mean_size,
            f"packet_size_variance{suffix}": var_size,
            f"packet_size_std_dev{suffix}": std_size,
            f"packet_size_range{suffix}": range_size,
            f"packet_size_25th_percentile{suffix}": size_25th,
            f"packet_size_50th_percentile{suffix}": size_50th,
            f"packet_size_75th_percentile{suffix}": size_75th,
            f"mean_inter_arrival_time{suffix}": mean_iat,
            f"inter_arrival_time_variance{suffix}": var_iat,
            f"inter_arrival_time_std_dev{suffix}": std_iat,
            f"inter_arrival_time_range{suffix}": range_iat,
            f"inter_arrival_time_25th_percentile{suffix}": iat_25th,
            f"inter_arrival_time_50th_percentile{suffix}": iat_50th,
            f"inter_arrival_time_75th_percentile{suffix}": iat_75th,
            f"burst_count{suffix}": bursts,
            f"silence_count{suffix}": silences,
            f"packet_size_entropy{suffix}": float(calculate_entropy(sizes)),
            f"packet_size_z_score{suffix}": float(np.mean(stats.zscore(sizes)) if len(sizes) > 1 else 0.0),
            f"inter_arrival_time_z_score{suffix}": float(
                np.mean(stats.zscore(inter_arrival)) if len(inter_arrival) > 1 else 0.0)
        }
        return features

    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return {}


def process_windows(small_window: List[Dict[str, Any]],
                    large_window: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process both windows efficiently.

    Args:
        small_window: 100-packet window
        large_window: 500-packet window

    Returns:
        Combined features from both windows
    """

    small_features = extract_flow_features(small_window, 100)
    large_features = extract_flow_features(large_window, 500)
    return {**small_features, **large_features}


def verify_feature_extraction(features: Dict[str, Any]) -> bool:
    """
    Verify all required features are present and valid.

    Args:
        features: Dictionary of extracted features

    Returns:
        True if all features are valid
    """

    required_suffixes = ['_100', '_500']
    required_base_features = [
        'flow_id',
        'flow_duration',
        'total_packets',
        'total_bytes',
        'packet_rate',
        'byte_rate',
        'average_packet_size',
        'packet_size_entropy',
        'mean_inter_arrival_time',
        'burst_count',
        'silence_count'
    ]

    for suffix in required_suffixes:
        for base in required_base_features:
            feature_name = f"{base}{suffix}"

            if feature_name not in features:
                print(f"Missing feature: {feature_name}")
                return False

            if base != 'flow_id':
                value = features[feature_name]
                if not isinstance(value, (int, float)):
                    print(f"Invalid feature type for {feature_name}: {type(value)}")
                    return False
                if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
                    print(f"Invalid feature value for {feature_name}: {value}")
                    return False

    return True
