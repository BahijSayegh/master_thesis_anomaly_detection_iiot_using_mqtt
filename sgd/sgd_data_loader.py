import os
import gc
import sys
import json
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import zstandard as zstd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Tuple, Dict, List, Optional, Union

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import config.sgd_config as cfg

# Set up logging
logger = logging.getLogger(__name__)

# Type aliases
BatchType = Tuple[np.ndarray, np.ndarray]
PathType = Path | str


class DataLoadingError(Exception):
    """Custom exception for data loading errors."""
    pass


def validate_data_paths() -> None:
    """
    Validate existence of all required data paths.

    Raises:
        DataLoadingError: If any required path is missing
    """
    try:
        for category in cfg.DATA_PATHS:
            for subcategory in cfg.DATA_PATHS[category]:
                path = cfg.DATA_PATHS[category][subcategory]
                if not path.exists():
                    raise DataLoadingError(f"Required path does not exist: {path}")

                # Check for JSONL.ZST files
                files = list(path.glob("*.jsonl.zst"))
                if not files:
                    raise DataLoadingError(f"No JSONL.ZST files found in {path}")

        logger.info("All data paths validated successfully")
    except Exception as e:
        raise DataLoadingError(f"Data path validation failed: {str(e)}")


def get_file_paths() -> Dict[str, List[Path]]:
    """
    Get paths of data files based on sampling configuration.

    Returns:
        Dictionary mapping categories to lists of file paths
    """
    all_files = {
        "normal": {
            "order": list(cfg.DATA_PATHS["normal"]["order"].glob("*.jsonl.zst")),
            "stock": list(cfg.DATA_PATHS["normal"]["stock"].glob("*.jsonl.zst"))
        },
        "anomalous": {
            "order": list(cfg.DATA_PATHS["anomalous"]["order"].glob("*.jsonl.zst")),
            "stock": list(cfg.DATA_PATHS["anomalous"]["stock"].glob("*.jsonl.zst"))
        }
    }

    if not cfg.DATA_CONFIG["use_sampling"]:
        return {
            "normal": all_files["normal"]["order"] + all_files["normal"]["stock"],
            "anomalous": all_files["anomalous"]["order"] + all_files["anomalous"]["stock"]
        }

    sampled_files = {
        "normal": [],
        "anomalous": []
    }

    total_files = cfg.DATA_CONFIG["total_files"]
    for category in ["normal", "anomalous"]:
        for subcategory in ["order", "stock"]:
            n_files = int(total_files * cfg.DATA_CONFIG["sampling_ratios"][category][subcategory])
            available_files = all_files[category][subcategory]

            if len(available_files) < n_files:
                logger.warning(
                    f"Requested {n_files} files from {category}/{subcategory} "
                    f"but only {len(available_files)} available"
                )
                n_files = len(available_files)

            # Random sampling
            sampled = np.random.choice(
                available_files,
                size=n_files,
                replace=False
            ).tolist()

            sampled_files[category].extend(sampled)

            logger.info(
                f"Sampled {len(sampled)} files from {category}/{subcategory} "
                f"(requested: {n_files})"
            )

    return sampled_files


def create_memmap_array(shape: Tuple[int, int], dtype: str = 'float32') -> np.ndarray:
    """
    Create a memory-mapped array for storing features.
    Cross-platform compatible (Windows and macOS).

    Args:
        shape: Shape of the array (n_samples, n_features)
        dtype: Data type for the array

    Returns:
        Memory-mapped numpy array
    """
    # Create a valid filename without special characters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory if it doesn't exist
    cfg.TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Use platform-specific path handling
        if sys.platform == 'win32':
            # For Windows, use tempfile
            import tempfile
            temp_dir = tempfile.gettempdir()
            filename = os.path.join(temp_dir, f'memmap_{timestamp}.dat')
        else:
            # For macOS/Linux, use the config path
            filename = str(cfg.TEMP_DIR / f'memmap_{timestamp}.dat')

        logger.info(f"Attempting to create memory-mapped array at: {filename}")

        memmap_array = np.memmap(
            filename,
            dtype=dtype,
            mode='w+',
            shape=shape,
            order='C'  # Use C-order for better performance
        )

        logger.info(f"Successfully created memory-mapped array with shape {shape}")
        return memmap_array

    except Exception as e:
        logger.error(f"Memory mapping failed, using regular array: {str(e)}")
        return np.zeros(shape, dtype=dtype)


def get_feature_dimensions(files: Dict[str, List[Path]]) -> int:
    """
    Get the maximum number of features across all files.

    Args:
        files: Dictionary of file paths by category

    Returns:
        Number of features
    """
    first_sample = None

    # Get first sample to establish baseline feature set
    for category, file_list in files.items():
        for file_path in file_list:
            with zstd.open(file_path, 'rt') as f:
                first_sample = json.loads(f.readline())
                break
        if first_sample:
            break

    if not first_sample:
        raise DataLoadingError("No data files found")

    # Use the same function to get feature names
    feature_names = get_feature_names(first_sample)
    logger.debug(f"Feature names: {feature_names}")  # Add this for debugging

    return len(feature_names)


def extract_file_id(file_path: Path) -> int:
    """
    Extract file ID from filename.

    Args:
        file_path: Path object of the file

    Returns:
        Extracted file ID as integer
    """
    try:
        # Get filename without extension
        name = file_path.stem
        if name.endswith('.jsonl'):
            name = name[:-6]  # Remove '.jsonl' if present

        # Split by '-' and get the numeric part
        parts = name.split('-')
        if len(parts) >= 2:
            # Explicitly type the filter function and cast to list of characters
            numeric_chars = [char for char in parts[-1] if char.isdigit()]
            numeric_part = ''.join(numeric_chars)
            if numeric_part:
                return int(numeric_part)

        # If we couldn't extract a numeric ID, create a deterministic hash
        hash_value = hash(str(file_path.stem))
        return abs(hash_value) % (10 ** 6)  # Keep it to 6 digits

    except Exception as e:
        logger.warning(f"Error extracting file ID from {file_path.name}: {str(e)}")
        return abs(hash(str(file_path))) % (10 ** 6)


def _process_file_chunk(args: Tuple[Path, List[str], Optional[MinMaxScaler], int, int, int, int]
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single file and return its features, labels, and file IDs.

    Args:
        args: Tuple containing:
            - file_path: Path to the data file
            - feature_names: List of feature names to extract
            - scaler: Optional scaler for feature normalization
            - start_idx: Starting index in the dataset
            - chunk_size: Size of chunks to process
            - file_id: Mapped file ID for consistent tracking
            - label: Label for the file (0 for normal, 1 for anomalous)

    Returns:
        Tuple of (features, labels, file_ids) arrays:
            - features: Array of shape (n_samples, n_features)
            - labels: Array of shape (n_samples,)
            - file_ids: Array of shape (n_samples)

    Notes:
        The function processes the file in chunks to manage memory efficiently
        and applies MQTT filtering if configured.
    """
    file_path, feature_names, scaler, start_idx, chunk_size, file_id, label = args

    try:
        features_list = []
        labels_list = []
        file_ids_list = []

        with zstd.open(file_path, 'rt') as f:
            # Initialize chunk buffer
            current_chunk: List[dict] = []

            for line in f:
                try:
                    packet = json.loads(line)

                    # Apply MQTT filtering if configured
                    if cfg.DATA_CONFIG["process_mqtt_only"]:
                        if packet.get(cfg.DATA_CONFIG["mqtt_feature_name"], 0) != 1:
                            continue

                    # Add file_id to the packet
                    packet['file_id'] = file_id
                    # Add label to the packet
                    packet[cfg.DATA_SPLIT_CONFIG['target_column']] = label

                    current_chunk.append(packet)

                    # Process chunk when it reaches the specified size
                    if len(current_chunk) >= chunk_size:
                        features_chunk, labels_chunk, file_ids_chunk = load_and_preprocess_chunk(
                            current_chunk, feature_names, scaler
                        )

                        if len(features_chunk) > 0:
                            features_list.append(features_chunk)
                            labels_list.append(labels_chunk)
                            file_ids_list.append(file_ids_chunk)

                        current_chunk = []

                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding JSON in {file_path}: {str(e)}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing packet in {file_path}: {str(e)}")
                    continue

            # Process remaining packets in the last chunk
            if current_chunk:
                features_chunk, labels_chunk, file_ids_chunk = load_and_preprocess_chunk(
                    current_chunk, feature_names, scaler
                )
                if len(features_chunk) > 0:
                    features_list.append(features_chunk)
                    labels_list.append(labels_chunk)
                    file_ids_list.append(file_ids_chunk)

        # Combine all chunks
        if not features_list:
            logger.warning(f"No valid features extracted from {file_path}")
            return np.array([]), np.array([]), np.array([])

        try:
            features = np.vstack(features_list)
            labels = np.concatenate(labels_list)
            file_ids = np.concatenate(file_ids_list)

            if len(features) == 0:
                logger.warning(f"No features extracted from {file_path}")
                return np.array([]), np.array([]), np.array([])

            return features, labels, file_ids

        except ValueError as e:
            logger.error(f"Error combining chunks from {file_path}: {str(e)}")
            return np.array([]), np.array([]), np.array([])

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return np.array([]), np.array([]), np.array([])


def _count_mqtt_packets(file_path: Path) -> int:
    """
    Count MQTT packets in a file.

    Args:
        file_path: Path to the data file

    Returns:
        Number of MQTT packets in the file
    """
    try:
        count = 0
        with zstd.open(file_path, 'rt') as f:
            for line in f:
                packet = json.loads(line)
                if packet.get(cfg.DATA_CONFIG["mqtt_feature_name"], 0) == 1:
                    count += 1
        return count
    except Exception as e:
        logger.warning(f"Error counting MQTT packets in {file_path}: {str(e)}")
        return 0


def _collect_scaling_samples(file_path: Path,
                             sample_percentage: float,
                             feature_names: List[str],
                             random_seed: Optional[int] = None) -> np.ndarray:
    """
    Collect samples from a single file for scaling.

    Args:
        file_path: Path to data file
        sample_percentage: Percentage of MQTT packets to sample (0.0 to 1.0)
        feature_names: List of feature names to extract
        random_seed: Random seed for reproducibility

    Returns:
        Array of samples for scaling
    """
    try:
        # First count MQTT packets in the file
        total_mqtt_packets = _count_mqtt_packets(file_path)
        if total_mqtt_packets == 0:
            return np.array([])

        # Calculate number of samples to collect
        n_samples = int(total_mqtt_packets * sample_percentage)
        if n_samples == 0:
            return np.array([])

        # Initialize random state
        rng = np.random.RandomState(random_seed)

        # Read all MQTT packets first
        mqtt_packets = []
        with zstd.open(file_path, 'rt') as f:
            for line in f:
                try:
                    packet = json.loads(line)
                    if packet.get(cfg.DATA_CONFIG["mqtt_feature_name"], 0) == 1:
                        # Add file_id if missing
                        if 'file_id' not in packet:
                            packet['file_id'] = extract_file_id(file_path)
                        # Add label_1 if missing
                        if 'label_1' not in packet:
                            # Determine label from file path
                            packet['label_1'] = 1 if 'Anomalous' in str(file_path) else 0
                        mqtt_packets.append(packet)
                except json.JSONDecodeError:
                    continue

        if not mqtt_packets:
            return np.array([])

        # Randomly sample MQTT packets
        sampled_indices = rng.choice(
            len(mqtt_packets),
            size=min(n_samples, len(mqtt_packets)),
            replace=False
        )
        sampled_packets = [mqtt_packets[i] for i in sampled_indices]

        # Process samples in chunks
        scaling_samples = []
        chunk = []
        for packet in sampled_packets:
            chunk.append(packet)
            if len(chunk) >= cfg.DATA_CONFIG["chunk_size"]:
                try:
                    features_chunk, _, _ = load_and_preprocess_chunk(chunk, feature_names)
                    if len(features_chunk) > 0:
                        scaling_samples.append(features_chunk)
                except Exception as e:
                    logger.debug(f"Error processing chunk in {file_path}: {str(e)}")
                chunk = []

        if chunk:  # Process remaining samples
            try:
                features_chunk, _, _ = load_and_preprocess_chunk(chunk, feature_names)
                if len(features_chunk) > 0:
                    scaling_samples.append(features_chunk)
            except Exception as e:
                logger.debug(f"Error processing final chunk in {file_path}: {str(e)}")

        if scaling_samples:
            return np.vstack(scaling_samples)
        return np.array([])

    except Exception as e:
        logger.warning(f"Error collecting samples from {file_path}: {str(e)}")
        return np.array([])


def process_files_with_memmap(files: Dict[str, List[Path]],
                              feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process files using memory mapping with parallel processing.

    Args:
        files: Dictionary mapping categories to lists of file paths
        feature_names: List of feature names to extract

    Returns:
        Tuple of (features, labels, file_ids) arrays

    Raises:
        DataLoadingError: If processing fails
    """
    try:
        n_features = len(feature_names)
        logger.info(f"Processing {n_features} features")
        n_jobs = max(1, cpu_count() - 1)

        # Count total MQTT packets first
        n_samples = 0
        logger.info("Counting MQTT packets...")

        # Create file ID mapping
        file_id_map = {}
        current_file_id = 0

        for category, file_list in files.items():
            for file_path in file_list:
                # Extract and map file ID
                original_id = extract_file_id(file_path)
                if original_id not in file_id_map:
                    file_id_map[original_id] = current_file_id
                    current_file_id += 1

                # Count packets
                with zstd.open(file_path, 'rt') as f:
                    if cfg.DATA_CONFIG["process_mqtt_only"]:
                        for line in f:
                            packet = json.loads(line)
                            if packet.get(cfg.DATA_CONFIG["mqtt_feature_name"], 0) == 1:
                                n_samples += 1
                    else:
                        for _ in f:
                            n_samples += 1

        logger.info(f"Total MQTT packets: {n_samples}")
        logger.info(f"Unique files mapped: {len(file_id_map)}")

        # Initialize arrays
        features = create_memmap_array((n_samples, n_features), dtype='float32')
        labels = np.zeros(n_samples, dtype='float32')
        file_ids = np.zeros(n_samples, dtype='int32')

        # Parallel sample collection for scaling
        logger.info(f"Collecting samples for scaling in parallel "
                    f"({cfg.DATA_CONFIG['scaling']['sample_percentage'] * 100:.1f}% of MQTT packets)...")

        sample_percentage = cfg.DATA_CONFIG["scaling"]["sample_percentage"]
        random_seed = cfg.DATA_CONFIG["random_state"]

        all_files = [(file_path, sample_percentage, feature_names, random_seed)
                     for category, file_list in files.items()
                     for file_path in file_list]

        with Pool(n_jobs) as pool:
            scaling_results = list(tqdm(
                pool.starmap(_collect_scaling_samples, all_files),
                total=len(all_files),
                desc="Collecting scaling samples"
            ))

        # Combine scaling samples and fit scaler
        scaling_samples = np.vstack([result for result in scaling_results if result.size > 0])
        n_collected = len(scaling_samples)
        logger.info(f"Collected {n_collected} samples for scaling "
                    f"({(n_collected / n_samples) * 100:.2f}% of total MQTT packets)")

        if n_collected < cfg.DATA_CONFIG["scaling"]["min_samples"]:
            logger.warning(f"Collected samples ({n_collected}) is below minimum threshold "
                           f"({cfg.DATA_CONFIG['scaling']['min_samples']})")

        logger.info("Fitting scaler...")
        scaler = MinMaxScaler()
        scaler.fit(scaling_samples)
        del scaling_samples
        gc.collect()

        # Process files in parallel
        logger.info("Processing files with parallel processing...")
        current_index = 0
        chunk_size = cfg.DATA_CONFIG["chunk_size"]

        for category, file_list in files.items():
            logger.info(f"Processing {category} files...")

            # Prepare arguments for parallel processing
            file_args = []
            for file_path in file_list:
                original_id = extract_file_id(file_path)
                mapped_id = file_id_map[original_id]
                file_args.append((
                    file_path,
                    feature_names,
                    scaler,
                    current_index,
                    chunk_size,
                    mapped_id,
                    1 if category == 'anomalous' else 0  # Label based on category
                ))

            with Pool(n_jobs) as pool:
                file_results = list(tqdm(
                    pool.imap(_process_file_chunk, file_args),
                    total=len(file_list),
                    desc=f"Processing {category} files"
                ))

            # Process results
            for features_chunk, labels_chunk, file_ids_chunk in file_results:
                if len(features_chunk) > 0:
                    chunk_size_actual = len(features_chunk)

                    if current_index + chunk_size_actual > n_samples:
                        logger.warning(f"Buffer overflow detected. Truncating chunk.")
                        chunk_size_actual = n_samples - current_index
                        features_chunk = features_chunk[:chunk_size_actual]
                        labels_chunk = labels_chunk[:chunk_size_actual]
                        file_ids_chunk = file_ids_chunk[:chunk_size_actual]

                    features[current_index:current_index + chunk_size_actual] = features_chunk
                    labels[current_index:current_index + chunk_size_actual] = labels_chunk
                    file_ids[current_index:current_index + chunk_size_actual] = file_ids_chunk

                    current_index += chunk_size_actual

                    if current_index >= n_samples:
                        logger.warning(f"Sample limit reached. Stopping at {current_index} samples.")
                        break

            if current_index >= n_samples:
                break

            # Clean up after each category
            if cfg.DATA_CONFIG["clear_cache"]:
                gc.collect()

        logger.info(f"Processed {current_index} samples")

        # Verify data and trim if necessary
        if current_index != n_samples:
            logger.warning(f"Sample count mismatch. Expected: {n_samples}, Got: {current_index}")
            features = features[:current_index]
            labels = labels[:current_index]
            file_ids = file_ids[:current_index]

        # Log final statistics
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        label_dist = {int(label): int(count) for label, count in zip(unique_labels, label_counts)}
        logger.info(f"Label distribution: {label_dist}")

        unique_files = np.unique(file_ids)
        logger.info(f"Number of unique file IDs: {len(unique_files)}")

        return features, labels, file_ids

    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        raise DataLoadingError(f"File processing failed: {str(e)}")


def load_and_preprocess_chunk(chunk: List[dict],
                              feature_names: List[str],
                              scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess a chunk of data, applying MQTT filtering and feature scaling.

    Args:
        chunk: List of dictionaries containing packet data
        feature_names: List of feature names to extract
        scaler: Optional scaler for feature normalization

    Returns:
        Tuple of (features, labels, file_ids)
    """
    try:
        if not chunk:
            return (np.empty((0, len(feature_names)), dtype='float32'),
                    np.empty(0, dtype='float32'),
                    np.empty(0, dtype='int32'))

        # Convert to DataFrame
        df = pd.DataFrame(chunk)

        # Handle required columns
        required_cols = {cfg.DATA_SPLIT_CONFIG['target_column'], 'file_id'}
        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            logger.debug(f"Adding missing required columns: {missing_cols}")
            for col in missing_cols:
                if col == 'file_id':
                    # Extract file_id from the first packet's metadata if available
                    df['file_id'] = 0  # Default value
                elif col == cfg.DATA_SPLIT_CONFIG['target_column']:
                    df[col] = 0  # Default to normal class

        # Apply MQTT filtering if configured
        if cfg.DATA_CONFIG["process_mqtt_only"]:
            mqtt_feature = cfg.DATA_CONFIG["mqtt_feature_name"]
            if mqtt_feature not in df.columns:
                return (np.empty((0, len(feature_names)), dtype='float32'),
                        np.empty(0, dtype='float32'),
                        np.empty(0, dtype='int32'))

            before_count = len(df)
            df = df[df[mqtt_feature] == 1].copy()
            after_count = len(df)

            if after_count == 0:
                return (np.empty((0, len(feature_names)), dtype='float32'),
                        np.empty(0, dtype='float32'),
                        np.empty(0, dtype='int32'))

        # Extract labels and file_ids
        labels = df[cfg.DATA_SPLIT_CONFIG['target_column']].values.astype('float32')
        file_ids = df['file_id'].values.astype('int32')

        # Add missing features with default values
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            for feature in missing_features:
                df[feature] = 0.0  # Default value for missing features

        # Ensure feature order matches feature_names
        df = df[feature_names]

        # Handle categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = pd.Categorical(df[col]).codes

        # Handle missing values
        df = df.fillna(0)  # Simple imputation with 0

        # Convert to float32
        features = df.astype('float32').values

        # Scale features if scaler provided
        if scaler is not None:
            features = scaler.transform(features)

        return features, labels, file_ids

    except Exception as e:
        logger.debug(f"Chunk preprocessing failed: {str(e)}")
        return (np.empty((0, len(feature_names)), dtype='float32'),
                np.empty(0, dtype='float32'),
                np.empty(0, dtype='int32'))


def get_feature_names(sample: dict) -> List[str]:
    """
    Extract feature names from a sample, handling required columns properly.

    Args:
        sample: Dictionary containing a data sample

    Returns:
        List of feature names
    """
    # Get all column names except excluded ones
    feature_names = [
        col for col in sample.keys()
        if col not in cfg.EXCLUDE_COLUMNS and
           col != 'packet_count' and
           col != 'timestamp' and
           not col.startswith('label_')
    ]

    # Sort to ensure consistent ordering
    feature_names.sort()

    return feature_names


def prepare_dataset(test_size: float = 0.2,
                    validation_size: float = 0.2,
                    random_state: int = 42
                    ) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[
        np.ndarray, np.ndarray, np.ndarray], List[str]]:
    """
    Prepare dataset with file-based stratification and dimension validation.

    Args:
        test_size: Proportion of data to use for testing
        validation_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple containing:
        - Training data: (X_train, y_train, train_file_ids)
        - Validation data: (X_val, y_val, val_file_ids)
        - Test data: (X_test, y_test, test_file_ids)
        - List of feature names

    Raises:
        DataLoadingError: If data loading or preprocessing fails
    """
    try:
        # Validate paths
        validate_data_paths()

        # Get file paths
        files = get_file_paths()

        # Get feature dimensions and validate
        n_features = get_feature_dimensions(files)
        logger.info(f"Detected {n_features} features in dataset")

        if n_features == 0:
            raise DataLoadingError("No features detected in dataset")

        # Log MQTT processing mode
        if cfg.DATA_CONFIG["process_mqtt_only"]:
            logger.info("Dataset preparation: Processing MQTT packets only")
            logger.info(f"Using MQTT feature identifier: {cfg.DATA_CONFIG['mqtt_feature_name']}")

        # Get feature names from first file
        first_file = next(iter(files['normal'] + files['anomalous']))
        with zstd.open(first_file, 'rt') as f:
            sample = json.loads(f.readline())
            feature_names = get_feature_names(sample)

        # Validate feature names against detected dimensions
        if len(feature_names) != n_features:
            logger.warning(f"Feature name count ({len(feature_names)}) does not match "
                           f"detected dimensions ({n_features})")
            # Ensure we're not duplicating or missing any features
            feature_set = set(feature_names)
            if len(feature_names) < n_features:
                # Add generic names for missing features
                for i in range(n_features - len(feature_names)):
                    new_name = f"feature_{len(feature_names) + i}"
                    while new_name in feature_set:
                        new_name = f"feature_{len(feature_names) + i}_alt"
                    feature_names.append(new_name)
                    feature_set.add(new_name)
            else:
                # Truncate excess features
                feature_names = feature_names[:n_features]

        logger.info(f"Using {len(feature_names)} feature names")

        # Process files and get features, labels, and file IDs
        X, y, file_ids = process_files_with_memmap(files, feature_names)

        # Get unique file IDs
        unique_file_ids = np.unique(file_ids)
        logger.info(f"Total unique files: {len(unique_file_ids)}")

        # First split: train+val vs test
        train_val_files, test_files = train_test_split(
            unique_file_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=None  # Can't stratify by file IDs
        )

        # Second split: train vs val
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=validation_size,
            random_state=random_state,
            stratify=None
        )

        # Create masks for each split
        train_mask = np.isin(file_ids, train_files)
        val_mask = np.isin(file_ids, val_files)
        test_mask = np.isin(file_ids, test_files)

        # Apply masks
        X_train, y_train = X[train_mask], y[train_mask]
        train_file_ids = file_ids[train_mask]

        X_val, y_val = X[val_mask], y[val_mask]
        val_file_ids = file_ids[val_mask]

        X_test, y_test = X[test_mask], y[test_mask]
        test_file_ids = file_ids[test_mask]

        # Log split sizes and class distributions
        logger.info("\nDataset split statistics:")
        logger.info(f"Training set:")
        logger.info(f"  Samples: {len(X_train)}")
        logger.info(f"  Files: {len(np.unique(train_file_ids))}")
        logger.info(f"  Class distribution: {np.bincount(y_train.astype(int))}")

        logger.info(f"\nValidation set:")
        logger.info(f"  Samples: {len(X_val)}")
        logger.info(f"  Files: {len(np.unique(val_file_ids))}")
        logger.info(f"  Class distribution: {np.bincount(y_val.astype(int))}")

        logger.info(f"\nTest set:")
        logger.info(f"  Samples: {len(X_test)}")
        logger.info(f"  Files: {len(np.unique(test_file_ids))}")
        logger.info(f"  Class distribution: {np.bincount(y_test.astype(int))}")

        return (
            (X_train, y_train, train_file_ids),
            (X_val, y_val, val_file_ids),
            (X_test, y_test, test_file_ids),
            feature_names
        )

    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
        raise DataLoadingError(f"Dataset preparation failed: {str(e)}")
