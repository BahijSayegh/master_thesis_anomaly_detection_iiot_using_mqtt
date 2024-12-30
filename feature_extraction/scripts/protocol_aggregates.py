from typing import Dict, List, Any, Tuple
import numpy as np
from numba import jit


def prepare_protocol_arrays(window: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare protocol-specific arrays from window data.
    This function handles dictionary unpacking before numba optimization.

    Args:
        window: List of packet dictionaries containing protocol information

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays for TCP flags, MQTT types, and QoS levels
    """

    n = len(window)
    tcp_flags = np.zeros((n, 2), dtype=np.int32)  # SYN and FIN flags
    mqtt_types = np.zeros(n, dtype=np.int32)  # PUBLISH and SUBSCRIBE
    qos_levels = np.zeros(n, dtype=np.int32)  # QoS 0,1,2

    for i, packet in enumerate(window):
        features = packet.get('features', {})

        # TCP flags
        tcp_flags[i, 0] = features.get('tcp_syn_flag', 0)
        tcp_flags[i, 1] = features.get('tcp_fin_flag', 0)

        # MQTT message types
        if features.get('is_mqtt', 0):
            msg_type = features.get('mqtt_message_type', '')
            if msg_type == 'PUBLISH':
                mqtt_types[i] = 1
            elif msg_type == 'SUBSCRIBE':
                mqtt_types[i] = 2
            # QoS level
            qos_levels[i] = features.get('mqtt_qos_level', 0)

    return tcp_flags, mqtt_types, qos_levels


@jit(nopython=True)
def calculate_tcp_ratios(tcp_flags: np.ndarray) -> float:
    """
    Calculate TCP SYN to FIN ratio with safety checks.

    Args:
        tcp_flags: Array of TCP flags [SYN, FIN]

    Returns:
        float: SYN to FIN ratio, or 0.0 if no FIN flags
    """

    syn_count = np.sum(tcp_flags[:, 0])
    fin_count = np.sum(tcp_flags[:, 1])

    if fin_count == 0:
        return 0.0

    return float(syn_count) / float(fin_count)


@jit(nopython=True)
def calculate_mqtt_ratios(mqtt_types: np.ndarray) -> float:
    """
    Calculate MQTT PUBLISH to SUBSCRIBE ratio with safety checks.

    Args:
        mqtt_types: Array of MQTT message types

    Returns:
        float: PUBLISH to SUBSCRIBE ratio, or 0.0 if no SUBSCRIBE messages
    """

    publish_count = np.sum(mqtt_types == 1)
    subscribe_count = np.sum(mqtt_types == 2)

    if subscribe_count == 0:
        return 0.0

    return float(publish_count) / float(subscribe_count)


@jit(nopython=True)
def calculate_qos_frequency(qos_levels: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate frequency distribution of QoS levels.

    Args:
        qos_levels: Array of QoS levels

    Returns:
        Tuple[float, float, float]: Frequencies of QoS 0, 1, and 2
    """

    total = len(qos_levels)
    if total == 0:
        return 0.0, 0.0, 0.0

    qos0 = float(np.sum(qos_levels == 0)) / total
    qos1 = float(np.sum(qos_levels == 1)) / total
    qos2 = float(np.sum(qos_levels == 2)) / total

    return qos0, qos1, qos2


def calculate_protocol_features(window: List[Dict[str, Any]], window_size: int) -> Dict[str, Any]:
    """
    Calculate all protocol-specific features for a given window.

    Args:
        window: List of packet dictionaries
        window_size: Size of window (100 or 500)

    Returns:
        Dict[str, Any]: Dictionary of calculated protocol features
    """

    if not window:
        return {}

    suffix = f"_{window_size}"

    try:
        # Prepare arrays for calculations (handle dictionary unpacking here)
        tcp_flags, mqtt_types, qos_levels = prepare_protocol_arrays(window)

        # Calculate protocol ratios using numba-optimized functions
        tcp_ratio = calculate_tcp_ratios(tcp_flags)
        mqtt_ratio = calculate_mqtt_ratios(mqtt_types)
        qos0_freq, qos1_freq, qos2_freq = calculate_qos_frequency(qos_levels)

        # Combine features
        features = {
            f"tcp_syn_to_fin_ratio{suffix}": float(tcp_ratio),
            f"mqtt_publish_subscribe_ratio{suffix}": float(mqtt_ratio),
            f"qos_level_frequency{suffix}": float(qos0_freq + qos1_freq + qos2_freq)
        }

        return features

    except Exception as e:
        print(f"Error in protocol feature extraction: {e}")
        return {}


def process_windows(small_window: List[Dict[str, Any]],
                    large_window: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process both windows to extract protocol features.

    Args:
        small_window: 100-packet window
        large_window: 500-packet window

    Returns:
        Dict[str, Any]: Combined features from both windows
    """

    small_features = calculate_protocol_features(small_window, 100)
    large_features = calculate_protocol_features(large_window, 500)

    return {**small_features, **large_features}


def verify_protocol_features(features: Dict[str, Any]) -> bool:
    """
    Verify all required protocol features are present and valid.

    Args:
        features: Dictionary of extracted features

    Returns:
        bool: True if all features are valid
    """

    required_suffixes = ['_100', '_500']
    required_base_features = [
        'tcp_syn_to_fin_ratio',
        'mqtt_publish_subscribe_ratio',
        'qos_level_frequency'
    ]

    for suffix in required_suffixes:
        for base in required_base_features:
            feature_name = f"{base}{suffix}"

            if feature_name not in features:
                print(f"Missing feature: {feature_name}")
                return False

            value = features[feature_name]
            if not isinstance(value, float):
                print(f"Invalid feature type for {feature_name}: {type(value)}")
                return False

            if np.isnan(value) or np.isinf(value):
                print(f"Invalid feature value for {feature_name}: {value}")
                return False

    return True
