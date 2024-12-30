import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.exceptions import NotFittedError
import joblib

import config.sgd_config as cfg

# Set up logging
logger = logging.getLogger(__name__)


class ClassifierError(Exception):
    """Custom exception for classifier errors."""
    pass


def create_classifier(model_type: str = None) -> Union[SVC, SGDClassifier]:
    """
    Create a classifier based on configuration.

    Args:
        model_type: Override model type from config ('sgd' or 'sgd')

    Returns:
        Initialized classifier

    Raises:
        ClassifierError: If model type is invalid
    """
    if model_type is None:
        model_type = cfg.MODEL_CONFIG["model_type"]

    try:
        if model_type == "sgd":
            return SVC(
                **cfg.MODEL_CONFIG["sgd"],
                random_state=cfg.DATA_CONFIG["random_state"]
            )
        elif model_type == "sgd":
            return SGDClassifier(
                **cfg.MODEL_CONFIG["sgd"],
                random_state=cfg.DATA_CONFIG["random_state"]
            )
        else:
            raise ClassifierError(f"Invalid model type: {model_type}")

    except Exception as e:
        raise ClassifierError(f"Error creating classifier: {str(e)}")


def optimize_hyperparameters(X: np.ndarray,
                             y: np.ndarray,
                             model_type: str = None) -> Tuple[Dict[str, Any], float]:
    """
    Optimize model hyperparameters using RandomizedSearchCV.

    Args:
        X: Training features
        y: Training labels
        model_type: Override model type from config

    Returns:
        Tuple of (best_parameters, best_score)
    """
    if model_type is None:
        model_type = cfg.MODEL_CONFIG["model_type"]

    try:
        # Create base model
        base_model = create_classifier(model_type)

        # Get parameter distributions
        param_dist = cfg.OPTIMIZATION_CONFIG["param_distributions"][model_type]

        # Create parameter grid
        param_grid = {}
        for param, config in param_dist.items():
            if isinstance(config, dict) and "log" in config:
                # Log-uniform distribution
                param_grid[param] = np.logspace(
                    np.log10(config["min"]),
                    np.log10(config["max"]),
                    num=20
                )
            elif isinstance(config, list):
                # Categorical parameters
                param_grid[param] = config
            else:
                param_grid[param] = config

        # Initialize search
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=cfg.OPTIMIZATION_CONFIG["n_trials"],
            cv=cfg.OPTIMIZATION_CONFIG["cv_folds"],
            random_state=cfg.DATA_CONFIG["random_state"],
            n_jobs=cfg.DATA_CONFIG["n_jobs"]
        )

        # Perform search
        logger.info("Starting hyperparameter optimization...")
        search.fit(X, y)

        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_:.4f}")

        # Convert numpy types to native Python types
        best_params = {}
        for key, value in search.best_params_.items():
            if isinstance(value, np.generic):
                best_params[key] = value.item()
            else:
                best_params[key] = value

        return best_params, float(search.best_score_)

    except Exception as e:
        raise ClassifierError(f"Hyperparameter optimization failed: {str(e)}")


class AnomalyClassifier:
    """
    Unified classifier class for anomaly detection.
    Handles both SVM and SGD with consistent interface.
    """

    def __init__(self, model_type: str = None):
        """
        Initialize classifier.

        Args:
            model_type: Override model type from config
        """
        self.model_type = model_type or cfg.MODEL_CONFIG["model_type"]
        self.model: Optional[Union[SVC, SGDClassifier]] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        self.metadata: Dict[str, Any] = {}

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            feature_names: Optional[List[str]] = None,
            optimize: bool = True) -> None:
        """
        Fit the classifier to training data.

        Args:
            X: Training features
            y: Training labels
            feature_names: Optional list of feature names
            optimize: Whether to perform hyperparameter optimization
        """
        try:
            # Store feature names
            self.feature_names = feature_names

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Store scaling parameters in metadata
            self.metadata['scaling'] = {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }

            # Optimize if requested
            if optimize:
                best_params, best_score = optimize_hyperparameters(X_scaled, y, self.model_type)
                self.model = create_classifier(self.model_type)
                self.model.set_params(**best_params)
                self.metadata['optimization'] = {
                    'best_score': best_score,
                    'best_params': best_params
                }
            else:
                self.model = create_classifier(self.model_type)

            # Fit model
            logger.info("Training classifier...")
            self.model.fit(X_scaled, y)
            self.is_fitted = True

            # Update metadata
            self.metadata.update({
                'training_shape': X.shape,
                'feature_names': feature_names,
                'model_type': self.model_type,
                'timestamp': datetime.now().isoformat(),
                'model_params': self.model.get_params()
            })

        except Exception as e:
            self.is_fitted = False
            raise ClassifierError(f"Model training failed: {str(e)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features to predict

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise NotFittedError("Classifier must be fitted before making predictions")

        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            raise ClassifierError(f"Prediction failed: {str(e)}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features to predict

        Returns:
            Prediction probabilities

        Raises:
            NotFittedError: If classifier is not fitted
            ClassifierError: If probability prediction fails
        """
        if not self.is_fitted:
            raise NotFittedError("Classifier must be fitted before making predictions")

        try:
            X_scaled = self.scaler.transform(X)

            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X_scaled)
            elif hasattr(self.model, 'decision_function'):
                # Convert decision function to probabilities using sigmoid
                decision_scores = self.model.decision_function(X_scaled)
                if decision_scores.ndim == 1:
                    decision_scores = decision_scores.reshape(-1, 1)

                def sigmoid(x):
                    return 1 / (1 + np.exp(-x))

                # Handle both binary and multiclass cases
                if decision_scores.shape[1] == 1:
                    probs = sigmoid(decision_scores)
                    return np.hstack([1 - probs, probs])
                else:
                    probs = sigmoid(decision_scores)
                    probs = probs / probs.sum(axis=1, keepdims=True)
                    return probs
            else:
                raise AttributeError("Model does not support probability prediction")

        except Exception as e:
            raise ClassifierError(f"Probability prediction failed: {str(e)}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if available.

        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not available
        """
        if not self.is_fitted:
            return None

        try:
            # Get feature importance based on model type
            if hasattr(self.model, 'coef_'):
                importance_scores = np.abs(self.model.coef_[0])
            elif hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_
            else:
                logger.warning("Model does not provide feature importance scores")
                return None

            # Map scores to feature names
            feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance_scores))]

            return {
                name: float(score)
                for name, score in zip(feature_names, importance_scores)
            }

        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {str(e)}")
            return None

    def save(self, save_dir: Path) -> None:
        """
        Save model, scaler, and metadata.

        Args:
            save_dir: Directory to save model files
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Save model and scaler
            model_path = save_dir / f"model_{timestamp}.joblib"
            scaler_path = save_dir / f"scaler_{timestamp}.joblib"

            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)

            # Update and save metadata
            self.metadata.update({
                'saved_timestamp': timestamp,
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'is_fitted': self.is_fitted,
            })

            # Get feature importance if available
            importance_dict = self.get_feature_importance()
            if importance_dict:
                self.metadata['feature_importance'] = importance_dict

            # Save metadata
            with open(save_dir / f"metadata_{timestamp}.json", "w") as f:
                json.dump(self.metadata, f, indent=4)

            logger.info(f"Model saved to {save_dir}")

        except Exception as e:
            raise ClassifierError(f"Error saving model: {str(e)}")

    @classmethod
    def load(cls, model_dir: Path, timestamp: str) -> 'AnomalyClassifier':
        """
        Load saved model and scaler.

        Args:
            model_dir: Directory containing saved model files
            timestamp: Timestamp of the saved model

        Returns:
            Loaded classifier
        """
        try:
            # Load metadata
            with open(model_dir / f"metadata_{timestamp}.json", "r") as f:
                metadata = json.load(f)

            # Create instance
            instance = cls(model_type=metadata["model_type"])
            instance.metadata = metadata

            # Load model and scaler
            instance.model = joblib.load(metadata["model_path"])
            instance.scaler = joblib.load(metadata["scaler_path"])
            instance.is_fitted = metadata["is_fitted"]
            instance.feature_names = metadata.get("feature_names")

            return instance

        except Exception as e:
            raise ClassifierError(f"Error loading model: {str(e)}")
