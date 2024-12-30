import os
import re
from typing import Dict, Any, Optional, Tuple

# Label mapping dictionaries
LABEL_MAPS = {
    "label_1_map": {"Normal": 0, "Anomalous": 1},
    "label_2_map": {"Stock": 0, "Order": 1},
    "label_3_map": {"Red": 0, "White": 1, "Blue": 2},
    "label_4_map": {
        "No-Anomaly": 0,
        "inject-bme": 1,
        "inject-ldr": 2,
        "move-camera": 3,
        "stock-how": 4,
        "change-state-dsi": 5,
        "change-state-dso": 6,
        "change-state-mpo-2": 7,
        "change-state-mpo-4": 8,
        "change-state-hbw-2": 9,
        "change-state-hbw-4": 10,
        "change-state-sld-2": 11,
        "change-state-sld-4": 12,
        "change-state-vgr-2": 13,
        "change-state-vgr-4": 14
    }
}

# Anomaly type ranges mapping
ANOMALY_TYPES = {
    range(1, 6): "inject-bme",
    range(6, 11): "inject-ldr",
    range(11, 16): "move-camera",
    range(16, 21): "stock-how",
    range(21, 26): "change-state-dsi",
    range(26, 31): "change-state-dso",
    range(31, 34): "change-state-mpo-2",
    range(34, 37): "change-state-mpo-4",
    range(37, 40): "change-state-hbw-2",
    range(40, 43): "change-state-hbw-4",
    range(43, 46): "change-state-sld-2",
    range(46, 49): "change-state-sld-4",
    range(49, 55): "change-state-vgr-2",
    range(55, 61): "change-state-vgr-4"
}


def parse_filename(filename: str) -> Tuple[str, str, str, int]:
    """
    Parse a pcapng filename to extract label components.

    Args:
        filename (str): Name of the pcapng file (e.g., 'Anomalous_Order_Blue-009.pcapng')

    Returns:
        Tuple[str, str, str, int]: Tuple containing (class_type, process_type, color, file_id)

    Raises:
        ValueError: If filename doesn't match expected format
    """
    # Remove file extension
    base_name = os.path.splitext(filename)[0]

    # Split into components
    pattern = r"(\w+)_(\w+)_(\w+)-(\d{3})"
    match = re.match(pattern, base_name)

    if not match:
        raise ValueError(f"Invalid filename format: {filename}")

    class_type, process_type, color, file_number = match.groups()
    file_number = int(file_number)

    # Calculate actual file_id based on file type
    base_ranges = {
        ('Normal', 'Order', 'Blue'): 0,  # 001-100
        ('Normal', 'Order', 'Red'): 100,  # 101-200
        ('Normal', 'Order', 'White'): 200,  # 201-300
        ('Normal', 'Stock', 'Blue'): 300,  # 301-400
        ('Normal', 'Stock', 'Red'): 400,  # 401-500
        ('Normal', 'Stock', 'White'): 500,  # 501-600
        ('Anomalous', 'Order', 'Blue'): 600,  # 601-660
        ('Anomalous', 'Order', 'Red'): 660,  # 661-720
        ('Anomalous', 'Order', 'White'): 720,  # 721-780
        ('Anomalous', 'Stock', 'Blue'): 780,  # 781-840
        ('Anomalous', 'Stock', 'Red'): 840,  # 841-900
        ('Anomalous', 'Stock', 'White'): 900  # 901-960
    }

    base_id = base_ranges.get((class_type, process_type, color), 0)
    file_id = base_id + file_number

    return class_type, process_type, color, file_id


def determine_anomaly_type(class_type: str, file_id: int) -> str:
    """
    Determine specific anomaly type based on file classification and ID ranges.

    Args:
        class_type (str): Classification of the file ('Normal' or 'Anomalous')
        file_id (int): Numerical identifier from filename

    Returns:
        str: Corresponding anomaly type or "No-Anomaly" for normal cases
    """
    # If it's a normal case, return immediately
    if class_type == "Normal":
        return "No-Anomaly"

    # For anomalous cases, calculate the relative position within the 60-file blocks
    base_id = ((file_id - 1) % 60) + 1

    # Find matching range for anomaly type
    for id_range, anomaly_type in ANOMALY_TYPES.items():
        if base_id in id_range:
            return anomaly_type

    return "No-Anomaly"


def create_label_dict(filename: str) -> Dict[str, Any]:
    """
    Create a complete label dictionary for a given file.

    This function implements a comprehensive labeling scheme for the anomaly detection
    system, incorporating multiple dimensions of classification.

    Args:
        filename (str): Name of the pcapng file

    Returns:
        Dict[str, Any]: Dictionary containing all label dimensions and file ID
    """
    # Parse filename components
    class_type, process_type, color, file_id = parse_filename(filename)

    # Create label dictionary
    labels = {
        'label_1': LABEL_MAPS['label_1_map'][class_type],
        'label_2': LABEL_MAPS['label_2_map'][process_type],
        'label_3': LABEL_MAPS['label_3_map'][color],
        'label_4': LABEL_MAPS['label_4_map'][determine_anomaly_type(class_type, file_id)],
        'file_id': file_id
    }

    return labels

def validate_labels(labels: Dict[str, Any]) -> bool:
    """
    Validate label dictionary for consistency and completeness.

    Args:
        labels (Dict[str, Any]): Label dictionary to validate

    Returns:
        bool: True if labels are valid, False otherwise
    """
    required_keys = {'label_1', 'label_2', 'label_3', 'label_4', 'file_id'}

    # Check for required keys
    if not all(key in labels for key in required_keys):
        return False

    # Validate value ranges
    validations = [
        0 <= labels['label_1'] <= 1,
        0 <= labels['label_2'] <= 1,
        0 <= labels['label_3'] <= 2,
        0 <= labels['label_4'] <= 14,
        1 <= labels['file_id'] <= 960
    ]

    return all(validations)

def process_file_labels(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Process and validate labels for a given file path.

    This function serves as the main entry point for label processing,
    combining parsing, creation, and validation of labels.

    Args:
        filepath (str): Full path to the pcapng file

    Returns:
        Optional[Dict[str, Any]]: Validated label dictionary or None if invalid
    """
    try:
        filename = os.path.basename(filepath)
        labels = create_label_dict(filename)

        if validate_labels(labels):
            return labels
        return None

    except (ValueError, KeyError) as e:
        print(f"Error processing labels for {filepath}: {str(e)}")
        return None
