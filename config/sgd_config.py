import logging
from pathlib import Path

import numpy as np

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "Processed"
OUTPUT_DIR = BASE_DIR / "data" / "Output" / "sgd"
TEMP_DIR = BASE_DIR / "data" / "temp"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Data Paths Structure
DATA_PATHS = {
    "normal": {
        "order": DATA_DIR / "Normal" / "Order",
        "stock": DATA_DIR / "Normal" / "Stock"
    },
    "anomalous": {
        "order": DATA_DIR / "Anomalous" / "Order",
        "stock": DATA_DIR / "Anomalous" / "Stock"
    }
}

MQTT_CONFIG = {
    "PROCESS_MQTT_ONLY": True,  # Set to True/False as needed
    "MQTT_FEATURE_NAME": "is_mqtt"
}

# Data Processing Configuration
DATA_CONFIG = {
    # MQTT Processing
    "process_mqtt_only": MQTT_CONFIG["PROCESS_MQTT_ONLY"],
    "mqtt_feature_name": MQTT_CONFIG["MQTT_FEATURE_NAME"],

    # Basic Processing Parameters
    "batch_size": 500,           # Number of samples to process at once
    "test_size": 0.2,           # 20% of data used for testing
    "validation_size": 0.2,      # 20% of training data used for validation
    "random_state": 42,         # Random seed for reproducibility
    "n_jobs": -1,               # Number of CPU cores to use (-1 = all cores)

    # Scaling Configuration
    "scaling": {
        "sample_percentage": 0.30,  # Percentage of MQTT packets to sample for scaling
        "min_samples": 10000,       # Minimum number of samples to collect
    },

    # File Sampling Configuration
    "use_sampling": True,       # Whether to use file sampling
    "total_files": 120,          # Total number of files to process
    "sampling_ratios": {        # Distribution of files across categories
        "normal": {
            "order": 0.3,       # 30% normal order files
            "stock": 0.3        # 30% normal stock files
        },
        "anomalous": {
            "order": 0.2,       # 20% anomalous order files
            "stock": 0.2        # 20% anomalous stock files
        }
    },

    # Memory Management
    "chunk_size": 500,          # Size of chunks for file reading
    "clear_cache": True,        # Whether to clear memory cache periodically
}

# Model Configuration
MODEL_CONFIG = {
    "model_type": "sgd",  # Model type: 'sgd' or 'sgd'

    # SVM Parameters
    "sgd": {
        "kernel": "linear",  # Kernel type: 'linear' or 'rbf'
        "C": 1.0,  # Regularization parameter
        "class_weight": "balanced",  # Adjust weights inversely proportional to class frequencies
        "probability": True,  # Enable probability estimates
        "cache_size": 2000,  # Kernel cache size in MB
        "max_iter": 1000,  # Maximum iterations (-1 for no limit)
        "tol": 1e-3  # Tolerance for stopping criterion
    },

    # SGD Classifier Parameters
    "sgd": {
        "loss": "modified_huber",  # Loss function: supports probability estimation
        "penalty": "l2",  # Regularization type
        "alpha": 0.0001,  # Regularization strength
        "max_iter": 1000,  # Maximum number of iterations
        "tol": 1e-3,  # Stopping criterion tolerance
        "shuffle": True,  # Shuffle samples in each iteration
        "learning_rate": "constant",  # Learning rate schedule
        "eta0": 0.1,  # Initial learning rate
        "power_t": 0.5,  # Power for inv-scaling learning rate
        "class_weight": "balanced",  # Adjust weights inversely proportional to class frequencies
        "warm_start": False,  # Reuse previous solution
        "early_stopping": False,  # Whether to use early stopping
        "validation_fraction": 0.1,  # Fraction of training data for early stopping
        "n_iter_no_change": 5,  # Number of iterations with no improvement to wait before early stopping
        "average": False  # Whether to average SGD weights
    }
}

# Hyperparameter Optimization Configuration
OPTIMIZATION_CONFIG = {
    "n_trials": 25,  # Number of optimization trials
    "cv_folds": 5,  # Number of cross-validation folds

    # Parameter Search Spaces
    "param_distributions": {
        # SVM hyperparameters
        "sgd": {
            "C": {
                "min": 0.1,  # Minimum regularization strength
                "max": 10,  # Maximum regularization strength
                "log": True  # Use log-uniform distribution
            },
            "kernel": ["linear", "rbf"],  # Kernel options
            "gamma": {  # Only for rbf kernel
                "min": 1e-6,
                "max": 1e1,
                "log": True
            }
        },
        # SGD hyperparameters
        "sgd": {
            "alpha": {
                "min": 1e-5,
                "max": 1e-1,
                "log": True
            },
            "loss": [
                "modified_huber",  # Smooth hinge loss
                "log_loss"  # Logistic regression loss
            ],
            "penalty": [
                "l2",  # Standard L2 regularization
                "l1",  # L1 regularization for sparsity
                "elasticnet"  # Combination of L1 and L2
            ],
            "learning_rate": ["constant"],  # Fixed learning rate
            "eta0": {
                "min": 0.01,
                "max": 1.0,
                "log": True
            }
        }
    }
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    # Performance Metrics to Calculate
    "metrics": [
        "accuracy",              # Overall accuracy
        "precision",            # Precision (true positives / predicted positives)
        "recall",               # Recall (true positives / actual positives)
        "f1",                   # F1 score (harmonic mean of precision and recall)
        "roc_auc",             # Area under ROC curve
        "average_precision"     # Average precision score
    ],

    # Statistical Analysis
    "confidence_level": 0.95,    # Confidence level for intervals
    "n_bootstrap": 1000,         # Number of bootstrap iterations

    # Visualization Settings
    "generate_plots": True,      # Whether to generate evaluation plots
    "plot_formats": ["png"],     # Output formats for plots
    "fig_dpi": 300,             # DPI for saved figures
    "fig_size": (10, 6)         # Default figure size (width, height) in inches
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": [
        logging.FileHandler(OUTPUT_DIR / "anomaly_detection.log"),
        logging.StreamHandler()
    ]
}

# Feature Processing
FEATURE_CONFIG = {
    "scaling": "standard",       # Feature scaling method
    "handle_missing": "mean",    # Strategy for missing values
    "handle_infinite": "clip",   # Strategy for infinite values
    "dtype": "float32"          # Data type for features
}

EXCLUDE_COLUMNS = {
    'timestamp', 'packet_id',  # Irrelevant metadata
    'label_2', 'label_3', 'label_4',  # Redundant labels
    'is_mqtt',
}

# Add new config for data splitting
DATA_SPLIT_CONFIG = {
    'stratify_by': 'file_id',  # Use file_id for stratification
    'target_column': 'label_1'  # Our main target variable
}

CROSS_VALIDATION_CONFIG = {
    "enabled": True,  # Add this line
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
    "validation_size": 0.2,

    # Visualization settings
    "visualization": {
        "dpi": 300,
        "figsize": (12, 8),
        "save_format": "png",
        "style": "seaborn",
        "palette": "deep"
    },

    # Additional metrics to compute
    "metrics": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "confusion_matrix",
        "classification_report"
    ],

    # ROC curve settings
    "roc_curve": {
        "compute": True,
        "plot": True
    },

    # Learning curve settings
    "learning_curve": {
        "compute": True,
        "plot": True,
        "train_sizes": np.linspace(0.1, 1.0, 10)
    }
}

"""
EXCLUDE_COLUMNS = {'timestamp', 'file_id', 'packet_id', 'label',
                   'byte_rate_100', 'byte_rate_500',
                   'packet_rate_100', 'packet_rate_500',
                   'packet_arrival_rate_mean_100', 'packet_arrival_rate_mean_500',
                   'packet_arrival_rate_max_100', 'packet_arrival_rate_max_500',
                   'average_packet_size_100', 'average_packet_size_500',
                   'total_bytes_100', 'total_bytes_500',
                   'burst_count_100', 'burst_count_500',
                   'coefficient_of_variation_100', 'coefficient_of_variation_500',
                   'iat_coefficient_of_variation_100', 'iat_coefficient_of_variation_500',
                   'payload_coefficient_of_variation_100', 'payload_coefficient_of_variation_500',
                   'source_ip_address', 'destination_ip_address',
                   'source_mac_address', 'destination_mac_address',
                   'source_port', 'destination_port',
                   'exponential_moving_average_100', 'exponential_moving_average_500',
                   'moving_linear_regression_slope_100', 'moving_linear_regression_slope_500',
                   'mean_inter_arrival_time_100', 'mean_inter_arrival_time_500',
                   'flow_duration_100', 'flow_duration_500',
                   'time_gaps_between_events_100', 'time_gaps_between_events_500',
                   'inter_arrival_time_range_100', 'inter_arrival_time_range_500',
                   }

"""
