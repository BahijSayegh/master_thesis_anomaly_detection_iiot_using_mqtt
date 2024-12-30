import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TypedDict, Literal, NotRequired

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score,
                             RocCurveDisplay, PrecisionRecallDisplay, roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.model_selection import StratifiedGroupKFold

import config.sgd_config as cfg

# Set up logging
logger = logging.getLogger(__name__)

MetricName = Literal[
    'accuracy', 'precision', 'recall', 'f1_score',
    'roc_auc', 'confusion_matrix', 'roc_data', 'pr_data'
]


class MetricStats(TypedDict):
    mean: float
    std: float
    min: float
    max: float
    ci_lower: float
    ci_upper: float


class BootstrapMetric(TypedDict):
    value: float
    ci_lower: float
    ci_upper: float


class ROCData(TypedDict):
    fpr: List[float]
    tpr: List[float]
    auc: float


class PRData(TypedDict):
    precision: List[float]
    recall: List[float]
    auc: float


class FeatureImportanceStats(TypedDict):
    importance: float
    std: float
    ci_lower: float
    ci_upper: float


class FoldScore(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    roc_data: ROCData
    pr_data: PRData
    bootstrap_metrics: Dict[str, BootstrapMetric]
    feature_importance: NotRequired[Dict[str, FeatureImportanceStats]]


class CVResults(TypedDict):
    fold_scores: List[FoldScore]
    models: List[Any]
    average_scores: Dict[str, MetricStats]
    metric_scores: Dict[str, List[float]]
    feature_importance: NotRequired[Dict[str, FeatureImportanceStats]]


class EvaluationMetrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    bootstrap_metrics: NotRequired[Dict[str, BootstrapMetric]]
    feature_importance: NotRequired[Dict[str, float]]


class EvaluationError(Exception):
    """Custom exception for evaluation errors."""
    pass


def _convert_cv_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy types to native Python types in CV metrics."""
    converted = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            converted[key] = value.item()
        elif isinstance(value, dict):
            converted[key] = _convert_cv_metrics(value)
        elif isinstance(value, (list, tuple)):
            converted[key] = [_convert_cv_metrics(x) if isinstance(x, dict) else
                              x.item() if isinstance(x, np.generic) else x
                              for x in value]
        else:
            converted[key] = value
    return converted


def setup_plotting_style() -> None:
    """Configure matplotlib for publication-quality plots."""
    try:
        # Use a basic style that's available in all matplotlib versions
        plt.style.use('default')

        # Then customize with our specific parameters
        plt.rcParams.update({
            'figure.figsize': cfg.EVALUATION_CONFIG['fig_size'],
            'figure.dpi': cfg.EVALUATION_CONFIG['fig_dpi'],
            'savefig.dpi': cfg.EVALUATION_CONFIG['fig_dpi'],
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'legend.title_fontsize': 12,
            'axes.grid': True,
            'grid.alpha': 0.3
        })

        # If seaborn is available, use its styling
        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid")
        except ImportError:
            logging.warning("Seaborn not available, using default matplotlib style")

    except Exception as e:
        logging.warning(f"Failed to set plotting style: {str(e)}. Using default style.")


def calculate_confidence_interval(metric_values: np.ndarray,
                                  confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for performance metrics.

    Args:
        metric_values: Array of metric values
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    try:
        if len(metric_values) == 0:
            return 0.0, 0.0

        valid_values = metric_values[np.isfinite(metric_values)]
        if len(valid_values) == 0:
            return 0.0, 0.0

        if len(valid_values) == 1:
            return float(valid_values[0]), float(valid_values[0])

        mean = np.mean(valid_values)
        sem = stats.sem(valid_values)

        if sem == 0:
            return float(mean), float(mean)

        ci = stats.t.interval(confidence, len(valid_values) - 1, loc=mean, scale=sem)

        if not np.isfinite(ci[0]) or not np.isfinite(ci[1]):
            return float(mean), float(mean)

        return float(ci[0]), float(ci[1])

    except Exception as e:
        logger.warning(f"Error calculating confidence interval: {str(e)}")
        mean = np.mean(metric_values)
        std = np.std(metric_values)
        return float(mean - std), float(mean + std)


def get_feature_importance(model: Any,
                           feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Safely extract feature importance from model.

    Args:
        model: Trained model
        feature_names: Optional list of feature names

    Returns:
        Dictionary mapping features to importance scores
    """
    try:
        # Try different methods to get feature importance
        if hasattr(model, 'coef_') and model.coef_ is not None:
            importance = np.abs(model.coef_[0])
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            logger.warning("Model does not provide feature importance scores")
            return {}

        # Ensure we have feature names
        if feature_names is None or len(feature_names) != len(importance):
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        # Create importance dictionary with native Python types
        return {
            name: float(score)
            for name, score in zip(feature_names, importance)
        }

    except Exception as e:
        logger.warning(f"Failed to extract feature importance: {str(e)}")
        return {}


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          save_path: Path) -> None:
    """Generate and save confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=['Normal', 'Anomaly'],
        yticklabels=['Normal', 'Anomaly']
    )

    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   save_path: Path) -> None:
    """Generate and save ROC curve visualization."""
    plt.figure(figsize=(10, 8))
    RocCurveDisplay.from_predictions(
        y_true,
        y_prob[:, 1],
        name="ROC curve",
        color="darkorange",
        plot_chance_level=True
    )
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                save_path: Path) -> None:
    """Generate and save Precision-Recall curve visualization."""
    plt.figure(figsize=(10, 8))
    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_prob[:, 1],
        name="Precision-Recall curve",
        color="darkorange"
    )
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cv_metrics_distribution(cv_results: Dict[str, Any], save_path: Path) -> None:
    """Plot distribution of metrics across CV folds."""
    try:
        # Set style at function level for safety
        plt.style.use('default')

        # Convert numpy types to native Python types
        cv_results = _convert_cv_metrics(cv_results)

        # Extract metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        scores = {metric: [] for metric in metrics}

        for fold_result in cv_results['fold_scores']:
            for metric in metrics:
                if metric in fold_result['bootstrap_metrics']:
                    scores[metric].append(fold_result['bootstrap_metrics'][metric]['value'])

        # Create visualization directory
        viz_path = save_path / 'cross_validation'
        viz_path.mkdir(parents=True, exist_ok=True)

        # Plot distributions
        plt.figure(figsize=(12, 6))
        plt.boxplot(
            [scores[m] for m in metrics],
            labels=metrics,
            patch_artist=True,
            medianprops=dict(color="black"),
            boxprops=dict(facecolor="lightblue")
        )

        plt.title('Cross-Validation Metrics Distribution')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(viz_path / 'cv_metrics_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save numerical results
        results_summary = {
            'metrics': {
                metric: {
                    'mean': float(np.mean(scores[metric])),
                    'std': float(np.std(scores[metric])),
                    'min': float(np.min(scores[metric])),
                    'max': float(np.max(scores[metric]))
                } for metric in metrics if scores[metric]
            },
            'fold_count': len(cv_results['fold_scores'])
        }

        with open(viz_path / 'cv_results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=4)

    except Exception as e:
        logger.error(f"Failed to generate CV visualizations: {str(e)}")


def visualize_feature_importance(importance_dict: Dict[str, float],
                                 save_path: Path) -> None:
    """Create and save feature importance visualizations."""
    try:
        viz_path = save_path / 'feature_importance'
        viz_path.mkdir(parents=True, exist_ok=True)

        # Get top 20 features
        top_features = dict(sorted(
            importance_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:20])

        plt.figure(figsize=(12, 8))
        features = list(top_features.keys())
        scores = list(top_features.values())

        bars = plt.barh(features, scores)
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Importance Score')

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height() / 2,
                     f'{width:.4f}',
                     ha='left', va='center')

        plt.tight_layout()
        plt.savefig(viz_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save importance scores
        with open(viz_path / 'importance_scores.json', 'w') as f:
            json.dump(importance_dict, f, indent=4)

    except Exception as e:
        logger.warning(f"Failed to create feature importance visualizations: {str(e)}")


def evaluate_classifier(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_prob: Optional[np.ndarray] = None,
                        model: Optional[Any] = None,
                        feature_names: Optional[List[str]] = None,
                        save_path: Optional[Path] = None,
                        n_bootstrap: int = 1000,
                        confidence: float = 0.95) -> EvaluationMetrics:
    """
    Evaluate classifier performance with comprehensive metrics and visualizations.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        model: Trained classifier model (optional)
        feature_names: List of feature names (optional)
        save_path: Path to save visualizations (optional)
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level for intervals

    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        setup_plotting_style()

        metrics: EvaluationMetrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'roc_auc': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]]
        }

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )

        metrics.update({
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        })

        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        if y_prob is not None:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))

            if save_path is not None:
                save_path.mkdir(parents=True, exist_ok=True)
                plot_confusion_matrix(y_true, y_pred, save_path)
                plot_roc_curve(y_true, y_prob, save_path)
                plot_precision_recall_curve(y_true, y_prob, save_path)

        # Calculate bootstrap confidence intervals
        bootstrap_metrics: Dict[str, BootstrapMetric] = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            bootstrap_values = []
            base_value = metrics[metric_name]

            for _ in range(n_bootstrap):
                indices = np.random.randint(0, len(y_true), size=len(y_true))
                if metric_name == 'roc_auc' and y_prob is not None:
                    value = roc_auc_score(y_true[indices], y_prob[indices, 1])
                else:
                    match metric_name:
                        case 'accuracy':
                            value = accuracy_score(y_true[indices], y_pred[indices])
                        case 'precision':
                            p, _, _, _ = precision_recall_fscore_support(
                                y_true[indices], y_pred[indices], average='binary'
                            )
                            value = p
                        case 'recall':
                            _, r, _, _ = precision_recall_fscore_support(
                                y_true[indices], y_pred[indices], average='binary'
                            )
                            value = r
                        case 'f1_score':
                            _, _, f, _ = precision_recall_fscore_support(
                                y_true[indices], y_pred[indices], average='binary'
                            )
                            value = f
                        case _:
                            continue

                bootstrap_values.append(value)

            ci_lower, ci_upper = calculate_confidence_interval(
                np.array(bootstrap_values),
                confidence=confidence
            )

            bootstrap_metrics[metric_name] = {
                'value': float(base_value),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper)
            }

        metrics['bootstrap_metrics'] = bootstrap_metrics

        # Feature importance analysis
        if model is not None:
            try:
                importance_dict = get_feature_importance(model, feature_names)
                if importance_dict and save_path is not None:
                    metrics['feature_importance'] = importance_dict
                    visualize_feature_importance(importance_dict, save_path)
            except Exception as e:
                logger.warning(f"Feature importance analysis failed: {str(e)}")

        # Log evaluation results
        logger.info("\nEvaluation Results:")
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if metric_name in bootstrap_metrics:
                bootstrap_result = bootstrap_metrics[metric_name]
                logger.info(
                    f"{metric_name}: {bootstrap_result['value']:.4f} "
                    f"({bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f})"
                )

        if save_path is not None:
            with open(save_path / 'evaluation_results.json', 'w') as f:
                json.dump(_convert_cv_metrics(metrics), f, indent=4)

        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise EvaluationError(f"Evaluation failed: {str(e)}")


def plot_cv_learning_curves(cv_results: Dict[str, Any], save_path: Path) -> None:
    """Plot learning curves across CV folds."""
    try:
        viz_path = save_path / 'cross_validation'
        viz_path.mkdir(parents=True, exist_ok=True)

        # Plot ROC curves for all folds
        plt.figure(figsize=(10, 8))
        mean_tpr = np.zeros_like(np.linspace(0, 1, 100))
        mean_fpr = np.linspace(0, 1, 100)

        for fold_idx, fold_scores in enumerate(cv_results['fold_scores']):
            if 'roc_data' in fold_scores:
                fpr = fold_scores['roc_data']['fpr']
                tpr = fold_scores['roc_data']['tpr']
                auc = fold_scores['roc_data']['auc']

                plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold_idx + 1} (AUC={auc:.3f})')
                mean_tpr += np.interp(mean_fpr, fpr, tpr)

        mean_tpr /= len(cv_results['fold_scores'])
        mean_auc = np.mean([fold['roc_data']['auc'] for fold in cv_results['fold_scores']])
        std_auc = np.std([fold['roc_data']['auc'] for fold in cv_results['fold_scores']])

        plt.plot(mean_fpr, mean_tpr, 'b-', label=f'Mean ROC (AUC={mean_auc:.3f}±{std_auc:.3f})',
                 linewidth=2)
        plt.plot([0, 1], [0, 1], 'r--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Across CV Folds')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_path / 'cv_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot PR curves for all folds
        plt.figure(figsize=(10, 8))
        mean_precision = np.zeros_like(np.linspace(0, 1, 100))
        mean_recall = np.linspace(0, 1, 100)

        for fold_idx, fold_scores in enumerate(cv_results['fold_scores']):
            if 'pr_data' in fold_scores:
                precision = fold_scores['pr_data']['precision']
                recall = fold_scores['pr_data']['recall']
                ap = fold_scores['pr_data']['auc']

                plt.plot(recall, precision, alpha=0.3, label=f'Fold {fold_idx + 1} (AP={ap:.3f})')
                mean_precision += np.interp(mean_recall, recall[::-1], precision[::-1])

        mean_precision /= len(cv_results['fold_scores'])
        mean_ap = np.mean([fold['pr_data']['auc'] for fold in cv_results['fold_scores']])
        std_ap = np.std([fold['pr_data']['auc'] for fold in cv_results['fold_scores']])

        plt.plot(mean_recall, mean_precision, 'b-',
                 label=f'Mean PR (AP={mean_ap:.3f}±{std_ap:.3f})',
                 linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Across CV Folds')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_path / 'cv_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot performance metrics distribution
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        values = {metric: [] for metric in metrics}
        for fold in cv_results['fold_scores']:
            for metric in metrics:
                if metric in fold['bootstrap_metrics']:
                    values[metric].append(fold['bootstrap_metrics'][metric]['value'])

        plt.figure(figsize=(12, 6))
        box_data = [values[metric] for metric in metrics]
        plt.boxplot(box_data, labels=metrics)
        plt.title('Distribution of Metrics Across CV Folds')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_path / 'cv_metrics_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"Failed to plot CV learning curves: {str(e)}")


def perform_cross_validation(X: np.ndarray,
                             y: np.ndarray,
                             file_ids: Optional[np.ndarray] = None,
                             model_trainer: callable = None,
                             save_path: Optional[Path] = None,
                             n_bootstrap: int = 1000,
                             confidence: float = 0.95) -> CVResults:
    """
    Perform stratified k-fold cross-validation with group-based splitting.

    Args:
        X: Feature matrix
        y: Target labels
        file_ids: Array of file IDs for group-based splitting
        model_trainer: Function that trains and returns a model
        save_path: Path to save results
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level for intervals

    Returns:
        Dictionary containing cross-validation results
    """
    if X.shape[0] != len(y):
        raise ValueError("Number of samples in X and y must match")
    if file_ids is None:
        raise ValueError("file_ids must be provided for group-based cross-validation")
    if len(file_ids) != len(y):
        raise ValueError("Number of file_ids must match number of samples")
    if model_trainer is None:
        raise ValueError("model_trainer function must be provided")

    setup_plotting_style()

    results: CVResults = {
        'fold_scores': [],
        'models': [],
        'average_scores': {},
        'metric_scores': {}
    }

    logger.info(f"Starting {cfg.CROSS_VALIDATION_CONFIG['n_splits']}-fold cross validation")
    logger.info(f"Number of unique files: {len(np.unique(file_ids))}")
    logger.info(f"Label distribution: {np.bincount(y.astype(int))}")

    cv_dir = save_path / 'cv_results' if save_path else None
    if cv_dir:
        cv_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedGroupKFold(
        n_splits=cfg.CROSS_VALIDATION_CONFIG["n_splits"],
        shuffle=cfg.CROSS_VALIDATION_CONFIG["shuffle"],
        random_state=cfg.CROSS_VALIDATION_CONFIG["random_state"]
    )

    feature_importance_scores = []

    for fold, (train_val_idx, test_idx) in enumerate(cv.split(X, y, file_ids), 1):
        fold_dir = cv_dir / f'fold_{fold}' if cv_dir else None
        if fold_dir:
            fold_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nFold {fold}/{cv.n_splits}")

        val_cv = StratifiedGroupKFold(
            n_splits=int(1 / cfg.CROSS_VALIDATION_CONFIG["validation_size"]),
            shuffle=True,
            random_state=cfg.CROSS_VALIDATION_CONFIG["random_state"]
        )

        X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
        files_train_val = file_ids[train_val_idx]

        train_idx_inner, val_idx_inner = next(val_cv.split(
            X_train_val,
            y_train_val,
            files_train_val
        ))

        train_idx = train_val_idx[train_idx_inner]
        val_idx = train_val_idx[val_idx_inner]

        X_train, y_train = X[train_idx], y[train_idx]
        train_files = file_ids[train_idx]

        X_val, y_val = X[val_idx], y[val_idx]
        val_files = file_ids[val_idx]

        X_test, y_test = X[test_idx], y[test_idx]
        test_files = file_ids[test_idx]

        # Check for data leakage
        train_file_set = set(train_files)
        val_file_set = set(val_files)
        test_file_set = set(test_files)

        if (train_file_set & val_file_set or
                train_file_set & test_file_set or
                val_file_set & test_file_set):
            logger.error("File ID leakage detected between splits!")
            raise ValueError("Data leakage detected between splits")

        logger.info("\nSplit sizes:")
        logger.info(f"Train: {len(X_train)} samples, {len(np.unique(train_files))} files")
        logger.info(f"  Label distribution: {np.bincount(y_train.astype(int))}")
        logger.info(f"Validation: {len(X_val)} samples, {len(np.unique(val_files))} files")
        logger.info(f"  Label distribution: {np.bincount(y_val.astype(int))}")
        logger.info(f"Test: {len(X_test)} samples, {len(np.unique(test_files))} files")
        logger.info(f"  Label distribution: {np.bincount(y_test.astype(int))}")

        try:
            # Train model
            model = model_trainer(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            # Calculate base metrics
            fold_metrics = evaluate_classifier(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                model=model,
                save_path=fold_dir,
                n_bootstrap=n_bootstrap,
                confidence=confidence
            )

            # Add ROC curve data
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            fold_metrics['roc_data'] = {
                'fpr': [float(x) for x in fpr],
                'tpr': [float(x) for x in tpr],
                'auc': float(roc_auc_score(y_test, y_prob[:, 1]))
            }

            # Add PR curve data
            precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
            fold_metrics['pr_data'] = {
                'precision': [float(x) for x in precision],
                'recall': [float(x) for x in recall],
                'auc': float(average_precision_score(y_test, y_prob[:, 1]))
            }

            results['fold_scores'].append(fold_metrics)
            results['models'].append(model)

            # Collect feature importance if available
            if 'feature_importance' in fold_metrics:
                feature_importance_scores.append(fold_metrics['feature_importance'])

            # Log fold results
            logger.info(f"\nFold {fold} results:")
            for metric_name, bootstrap_data in fold_metrics['bootstrap_metrics'].items():
                logger.info(
                    f"{metric_name}: {bootstrap_data['value']:.4f} "
                    f"({bootstrap_data['ci_lower']:.4f}, {bootstrap_data['ci_upper']:.4f})"
                )

        except Exception as e:
            logger.error(f"Error in fold {fold}: {str(e)}")
            continue

    # Calculate aggregate statistics if we have results
    if results['fold_scores']:
        metric_averages = {}
        metric_values = {}

        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            values = [
                fold['bootstrap_metrics'][metric_name]['value']
                for fold in results['fold_scores']
            ]

            if values:
                ci_lower, ci_upper = calculate_confidence_interval(
                    np.array(values),
                    confidence=confidence
                )

                metric_averages[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'ci_lower': float(ci_lower),
                    'ci_upper': float(ci_upper),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
                metric_values[metric_name] = values

        results['average_scores'] = metric_averages
        results['metric_scores'] = metric_values

        # Aggregate feature importance across folds if available
        if feature_importance_scores:
            aggregated_importance = {}
            for feature_name in feature_importance_scores[0].keys():
                importance_values = [
                    scores[feature_name]
                    for scores in feature_importance_scores
                ]
                ci_lower, ci_upper = calculate_confidence_interval(
                    np.array(importance_values),
                    confidence=confidence
                )
                aggregated_importance[feature_name] = {
                    'importance': float(np.mean(importance_values)),
                    'std': float(np.std(importance_values)),
                    'ci_lower': float(ci_lower),
                    'ci_upper': float(ci_upper)
                }

            results['feature_importance'] = aggregated_importance

        # Generate visualizations if save path provided
        if save_path is not None:
            try:
                plot_cv_metrics_distribution(results, save_path)
                plot_cv_learning_curves(results, save_path)

                # Save detailed results
                with open(save_path / 'cv_results.json', 'w') as f:
                    json.dump({
                        'average_scores': results['average_scores'],
                        'feature_importance': results.get('feature_importance', {}),
                        'configuration': {
                            'n_splits': cv.n_splits,
                            'n_bootstrap': n_bootstrap,
                            'confidence_level': confidence
                        }
                    }, f, indent=4)
            except Exception as e:
                logger.warning(f"Error generating CV visualizations: {str(e)}")

        # Log final results
        logger.info("\nCross validation results:")
        for metric_name, metric_stats in metric_averages.items():
            logger.info(
                f"{metric_name}: {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f} "
                f"({metric_stats['ci_lower']:.4f}, {metric_stats['ci_upper']:.4f})"
            )

    return results
