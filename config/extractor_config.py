from pathlib import Path

# Project root and main directories
BASE_DIR = Path(__file__).parent.parent  # Move up two levels: config -> anomaly_detection
DATA_DIR = BASE_DIR / "data"
RAW_DIR = Path(r"C:\Users\BahijSayegh\Master Thesis Data Capture\FT-24v Traffic Capture")

# Data directories
PROCESSED_DIR = DATA_DIR / "Processed"
OUTPUT_DIR = DATA_DIR / "Output"

# Model specific directories
RANDOM_FOREST_DIR = BASE_DIR / "random_forest"
SVM_DIR = BASE_DIR / "sgd"
FEATURE_ANALYSIS_DIR = BASE_DIR / "feature_analysis"

# Feature extractor directories
FEATURE_EXTRACTOR_DIR = BASE_DIR / "feature_extractor"
SCRIPTS_DIR = FEATURE_EXTRACTOR_DIR / "scripts"


# Ensure all required directories exist
def create_directory_structure():
    """Create the complete directory structure for the project."""
    # Data directories
    for category in ["Normal", "Anomalous"]:
        for process in ["Order2", "Stock"]:
            (PROCESSED_DIR / category / process).mkdir(parents=True, exist_ok=True)

    # Output directories for different models
    for model in ["random_forest", "sgd"]:
        (OUTPUT_DIR / model).mkdir(parents=True, exist_ok=True)


# Create directories
create_directory_structure()

# Processing control
TARGET_CATEGORY = "Anomalous"  # "Normal" or "Anomalous"
TARGET_PROCESS = "Order2"  # "ALL", "Order", or "Stock"
FILES_TO_PROCESS = -1  # -1 for all files, or specific number like 2,3,...
MAX_WORKERS = 8  # Number of parallel processes, or -1 for CPU count

# Processing parameters
CHUNK_SIZE = 1000  # Number of packets to process at once
LARGE_WINDOW = 500  # Size of larger sliding window
SMALL_WINDOW = 100  # Size of smaller sliding window

# Flow feature extraction parameters
SILENCE_THRESHOLD = 0.1  # seconds, threshold for silence detection
BURST_THRESHOLD = 0.01  # seconds, threshold for burst detection
Z_SCORE_THRESHOLD = 3.0  # threshold for anomaly detection in z-scores

# Protocol-specific parameters
MQTT_PORTS = {1883, 8883}  # Standard MQTT ports

# Compression settings
COMPRESSION_ENABLED = True
COMPRESSION_LEVEL = 3  # Range: 1 (fastest) to 22 (highest compression)
COMPRESSION_THREADS = -1  # Number of compression threads
COMPRESSION_CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming compression
