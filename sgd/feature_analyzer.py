import json
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union, Any, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import stats
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
CorrelationMethod = Union[
    Literal["pearson", "kendall", "spearman"],
    Callable[[np.ndarray, np.ndarray], float]
]


class FeatureAnalysisError(Exception):
    """Custom exception for feature analysis errors."""
    pass


def validate_feature_names(feature_names: List[str],
                           data_shape: int) -> List[str]:
    """
    Validate and adjust feature names to match data dimensions.

    Args:
        feature_names: List of feature names
        data_shape: Number of features in data

    Returns:
        Validated list of feature names
    """
    if len(feature_names) != data_shape:
        logger.warning(
            f"Feature name count mismatch. Names: {len(feature_names)}, "
            f"Data shape: {data_shape}"
        )
        if len(feature_names) > data_shape:
            logger.info(f"Truncating feature names to match data shape")
            return feature_names[:data_shape]
        else:
            logger.info(f"Adding generic names for missing features")
            additional = [f"feature_{i}" for i in range(len(feature_names), data_shape)]
            return feature_names + additional
    return feature_names


def calculate_correlation_chunk(args: Tuple[np.ndarray, List[str], slice, CorrelationMethod]
                                ) -> pd.DataFrame:
    """
    Calculate correlation for a chunk of features.

    Args:
        args: Tuple containing (data_chunk, feature_names, chunk_slice, method)

    Returns:
        Correlation matrix for the chunk
    """
    data, features, chunk_slice, method = args
    chunk_df = pd.DataFrame(data[:, chunk_slice], columns=features[chunk_slice])
    return chunk_df.corr(method=method)


def calculate_parallel_correlations(X: np.ndarray,
                                    feature_names: List[str],
                                    method: CorrelationMethod = "spearman",
                                    n_jobs: Optional[int] = None,
                                    chunk_size: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate correlation matrix using parallel processing.

    Args:
        X: Feature matrix
        feature_names: List of feature names
        method: Correlation method ('pearson', 'kendall', 'spearman', or callable)
        n_jobs: Number of processes to use
        chunk_size: Size of feature chunks for parallel processing

    Returns:
        Complete correlation matrix
    """
    n_features = X.shape[1]

    # Determine optimal chunk size and number of jobs
    if n_jobs is None or n_jobs < 1:
        n_jobs = max(1, cpu_count() - 1)  # Ensure at least 1 process

    if chunk_size is None:
        chunk_size = max(1, n_features // (n_jobs * 2))

    # Create chunks
    chunks = [(i, min(i + chunk_size, n_features))
              for i in range(0, n_features, chunk_size)]

    # Prepare arguments for parallel processing
    chunk_args = [
        (X, feature_names, slice(start, end), method)
        for start, end in chunks
    ]

    # Calculate correlations in parallel
    logger.info(f"Starting parallel correlation calculation with {n_jobs} processes")
    try:
        with Pool(n_jobs) as pool:
            chunk_results = list(tqdm(
                pool.imap(calculate_correlation_chunk, chunk_args),
                total=len(chunks),
                desc="Calculating correlations"
            ))

        # Combine results
        return pd.concat([pd.concat(chunk_results, axis=1)], axis=0)

    except Exception as e:
        logger.error(f"Parallel correlation calculation failed: {str(e)}")
        raise FeatureAnalysisError(f"Correlation calculation failed: {str(e)}")


def calculate_feature_correlations(X: np.ndarray,
                                   feature_names: List[str],
                                   method: CorrelationMethod = "spearman",
                                   sample_size: Optional[int] = None,
                                   n_jobs: Optional[int] = None,
                                   chunk_size: Optional[int] = None
                                   ) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, float]]]]:
    """
    Calculate comprehensive correlation matrix between features.

    Args:
        X: Feature matrix
        feature_names: List of feature names
        method: Correlation method
        sample_size: Number of samples to use (optional)
        n_jobs: Number of processes to use (optional)
        chunk_size: Size of feature chunks (optional)

    Returns:
        Tuple of (correlation_matrix, feature_correlations)
        where feature_correlations maps features to their correlated features
    """
    try:
        # Validate feature names
        validated_feature_names = validate_feature_names(feature_names, X.shape[1])
        logger.info(f"Using {len(validated_feature_names)} validated feature names")

        # Sample data if needed
        if sample_size and X.shape[0] > sample_size:
            logger.info(f"Sampling {sample_size} rows from {X.shape[0]} total rows")
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # Calculate correlation matrix in parallel
        correlation_matrix = calculate_parallel_correlations(
            X_sample,
            validated_feature_names,
            method=method,
            n_jobs=n_jobs,
            chunk_size=chunk_size
        )

        # Create dictionary of correlated features
        logger.info("Processing feature correlations...")
        feature_correlations: Dict[str, List[Tuple[str, float]]] = {}

        for feature in tqdm(validated_feature_names, desc="Processing correlations"):
            correlations = [
                (str(other_feat), float(correlation_matrix.at[feature, other_feat]))
                for other_feat in validated_feature_names
                if other_feat != feature and abs(correlation_matrix.at[feature, other_feat]) > 0.5]

            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            feature_correlations[feature] = correlations

        # Save visualization
        plot_correlation_heatmap(correlation_matrix, len(validated_feature_names))

        return correlation_matrix, feature_correlations

    except Exception as e:
        logger.error(f"Correlation calculation failed: {str(e)}")
        raise FeatureAnalysisError(f"Correlation calculation failed: {str(e)}")


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame,
                             n_features: int,
                             save_dir: Optional[Path] = None) -> None:
    """
    Create and save correlation heatmap visualization.

    Args:
        correlation_matrix: Feature correlation matrix
        n_features: Number of features
        save_dir: Optional directory to save visualization
    """
    try:
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(correlation_matrix))

        sns.heatmap(correlation_matrix,
                    mask=mask,
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={'shrink': 0.5})

        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')

        plt.close()

        # Save correlation statistics
        if save_dir:
            correl_stats = {
                'n_features': n_features,
                'high_correlations': int(np.sum(np.abs(correlation_matrix) > 0.8)),
                'moderate_correlations': int(np.sum((np.abs(correlation_matrix) > 0.5) &
                                                    (np.abs(correlation_matrix) <= 0.8))),
                'low_correlations': int(np.sum(np.abs(correlation_matrix) <= 0.5)),
                'timestamp': datetime.now().isoformat()
            }

            with open(save_dir / 'correlation_stats.json', 'w') as f:
                json.dump(correl_stats, f, indent=4)

    except Exception as e:
        logger.warning(f"Failed to create correlation heatmap: {str(e)}")


def _calculate_feature_leakage(args: Tuple[np.ndarray, np.ndarray, str]) -> Dict[str, float]:
    """
    Calculate leakage metrics for a single feature.

    Args:
        args: Tuple of (feature_values, target_values, feature_name)

    Returns:
        Dictionary containing leakage metrics for the feature
    """
    feature_values, target_values, feature_name = args

    try:
        # Calculate point biserial correlation
        correlation, p_value = stats.pointbiserialr(feature_values, target_values)

        # Calculate mutual information
        mi_score = mutual_info_classif(
            feature_values.reshape(-1, 1),
            target_values,
            random_state=42
        )[0]

        return {
            'feature': feature_name,
            'correlation': float(correlation),
            'mutual_info': float(mi_score),
            'p_value': float(p_value)
        }

    except Exception as e:
        logger.warning(f"Error calculating leakage for feature {feature_name}: {str(e)}")
        return {
            'feature': feature_name,
            'correlation': np.nan,
            'mutual_info': np.nan,
            'p_value': np.nan
        }


def detect_data_leakage(X: np.ndarray,
                        y: np.ndarray,
                        feature_names: List[str],
                        threshold: float = 0.9,
                        n_jobs: Optional[int] = None,
                        save_dir: Optional[Path] = None) -> List[Dict[str, float]]:
    """
    Detect potential data leakage by identifying features highly correlated with target.

    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        threshold: Correlation threshold for potential leakage
        n_jobs: Number of parallel jobs
        save_dir: Optional directory to save analysis results

    Returns:
        List of dictionaries containing leakage information
    """
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    # Prepare arguments for parallel processing
    process_args = [
        (X[:, i], y, feature_name)
        for i, feature_name in enumerate(feature_names)
    ]

    try:
        # Calculate leakage metrics in parallel
        logger.info(f"Starting parallel leakage detection with {n_jobs} processes")
        with Pool(n_jobs) as pool:
            results = list(tqdm(
                pool.imap(_calculate_feature_leakage, process_args),
                total=len(feature_names),
                desc="Detecting data leakage"
            ))

        # Filter results based on threshold
        leakage_info = [
            result for result in results
            if (np.abs(result['correlation']) > threshold or
                result['mutual_info'] > threshold) and
               np.isfinite(result['correlation']) and
               np.isfinite(result['mutual_info'])
        ]

        # Sort by absolute correlation
        leakage_info.sort(key=lambda x: abs(x['correlation']), reverse=True)

        # Log summary statistics
        total_features = len(feature_names)
        leakage_features = len(leakage_info)

        logger.info(f"Data leakage analysis complete:")
        logger.info(f"  Total features analyzed: {total_features}")
        logger.info(f"  Features with potential leakage: {leakage_features}")
        logger.info(f"  Leakage percentage: {(leakage_features / total_features) * 100:.2f}%")

        if leakage_features > 0:
            logger.warning("Top features with potential leakage:")
            for info in leakage_info[:5]:
                logger.warning(
                    f"  {info['feature']}: correlation={info['correlation']:.4f}, "
                    f"mutual_info={info['mutual_info']:.4f}"
                )

        # Save analysis results if directory provided
        if save_dir and leakage_info:
            save_dir.mkdir(parents=True, exist_ok=True)

            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'threshold': threshold,
                'total_features': total_features,
                'leakage_features': leakage_features,
                'leakage_percentage': (leakage_features / total_features) * 100,
                'detected_leakage': leakage_info
            }

            with open(save_dir / 'leakage_analysis.json', 'w') as f:
                json.dump(analysis_results, f, indent=4)

            # Create visualization
            plot_leakage_analysis(leakage_info, save_dir)

        return leakage_info

    except Exception as e:
        logger.error(f"Parallel leakage detection failed: {str(e)}")
        raise


def _calculate_stability(args: Tuple[np.ndarray, int]) -> float:
    """
    Calculate stability score for a single feature.

    Args:
        args: Tuple containing (feature_values, window_size)

    Returns:
        Stability score as float, or np.nan if calculation fails
    """
    values, win_size = args

    try:
        if len(values) < win_size:
            logger.warning(f"Insufficient data for window size {win_size}")
            return np.nan

        # Create windows
        windows = [
            values[i:i + win_size]
            for i in range(0, len(values) - win_size, win_size)
        ]

        if not windows:
            logger.warning("No windows created from data")
            return np.nan

        # Calculate window statistics
        try:
            with np.errstate(all='raise'):
                window_means = np.array([np.mean(w) for w in windows])
                window_stds = np.array([np.std(w) for w in windows])
        except FloatingPointError as e:
            logger.warning(f"Numerical error in window calculations: {str(e)}")
            return np.nan

        # Calculate stability measures
        try:
            mean_stability = 0.0
            std_stability = 0.0

            mean_denominator = np.mean(window_means)
            std_denominator = np.mean(window_stds)

            if not np.isclose(mean_denominator, 0):
                mean_stability = np.std(window_means) / mean_denominator
            if not np.isclose(std_denominator, 0):
                std_stability = np.std(window_stds) / std_denominator

            stability_score = max(mean_stability, std_stability)

            if np.isfinite(stability_score):
                return float(stability_score)
            else:
                logger.warning("Non-finite stability score calculated")
                return np.nan

        except (FloatingPointError, ZeroDivisionError) as e:
            logger.warning(f"Error in stability score calculation: {str(e)}")
            return np.nan

    except Exception as e:
        logger.error(f"Unexpected error in stability calculation: {str(e)}")
        return np.nan


def parallel_feature_stability(X: np.ndarray,
                               feature_names: List[str],
                               window_size: int = 1000,
                               n_jobs: Optional[int] = None,
                               save_dir: Optional[Path] = None) -> Dict[str, float]:
    """
    Analyze feature stability over time using parallel processing.

    Args:
        X: Feature matrix
        feature_names: List of feature names
        window_size: Size of rolling window
        n_jobs: Number of processes to use
        save_dir: Optional directory to save analysis results

    Returns:
        Dictionary mapping features to their stability scores
    """
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    stability_args = [(X[:, i], window_size) for i in range(X.shape[1])]

    try:
        with Pool(n_jobs) as pool:
            stability_scores = list(tqdm(
                pool.imap(_calculate_stability, stability_args),
                total=len(stability_args),
                desc="Analyzing feature stability"
            ))

        stability_dict = dict(zip(feature_names, stability_scores))

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

            # Calculate statistics
            valid_scores = [score for score in stability_scores if np.isfinite(score)]
            val_score_stats = {
                'timestamp': datetime.now().isoformat(),
                'window_size': window_size,
                'features_analyzed': len(feature_names),
                'features_with_valid_scores': len(valid_scores),
                'mean_stability': float(np.mean(valid_scores)),
                'std_stability': float(np.std(valid_scores)),
                'min_stability': float(np.min(valid_scores)),
                'max_stability': float(np.max(valid_scores)),
                'stability_scores': {
                    name: float(score)
                    for name, score in stability_dict.items()
                    if np.isfinite(score)
                }
            }

            with open(save_dir / 'stability_analysis.json', 'w') as f:
                json.dump(val_score_stats, f, indent=4)

            # Create visualization
            plot_stability_distribution(stability_dict, save_dir)

        return stability_dict

    except Exception as e:
        logger.error(f"Failed to complete stability analysis: {str(e)}")
        raise


def plot_stability_distribution(stability_scores: Dict[str, float],
                                save_dir: Path) -> None:
    """
    Create and save feature stability distribution visualization.

    Args:
        stability_scores: Dictionary of feature stability scores
        save_dir: Directory to save visualization
    """
    try:
        plt.figure(figsize=(12, 6))

        # Convert scores to list and handle infinite values
        scores = list(stability_scores.values())
        finite_scores = [score for score in scores
                         if not np.isinf(score) and not np.isnan(score)]

        if not finite_scores:
            logger.warning("No finite stability scores found")
            return

        # Create histogram
        plt.hist(finite_scores, bins=50, edgecolor='black')
        plt.title('Feature Stability Score Distribution')
        plt.xlabel('Stability Score')
        plt.ylabel('Count')

        # Add statistics box
        stats_text = (
            f'Total Features: {len(scores)}\n'
            f'Mean Stability: {np.mean(finite_scores):.4f}\n'
            f'Std Stability: {np.std(finite_scores):.4f}\n'
            f'Infinite/NaN: {len(scores) - len(finite_scores)}'
        )

        plt.text(0.95, 0.95, stats_text,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_dir / 'stability_distribution.png', dpi=300)
        plt.close()

    except Exception as e:
        logger.warning(f"Failed to plot stability distribution: {str(e)}")


def plot_leakage_analysis(leakage_info: List[Dict[str, float]],
                          save_dir: Path) -> None:
    """
    Create and save data leakage analysis visualization.

    Args:
        leakage_info: List of dictionaries containing leakage information
        save_dir: Directory to save visualization
    """
    try:
        if not leakage_info:
            return

        plt.figure(figsize=(12, 6))

        features = [info['feature'] for info in leakage_info]
        correlations = [abs(info['correlation']) for info in leakage_info]
        mutual_info = [info['mutual_info'] for info in leakage_info]

        x = np.arange(len(features))
        width = 0.35

        plt.bar(x - width / 2, correlations, width, label='Correlation')
        plt.bar(x + width / 2, mutual_info, width, label='Mutual Information')

        plt.xlabel('Features')
        plt.ylabel('Score')
        plt.title('Potential Data Leakage Analysis')
        plt.xticks(x, features, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(save_dir / 'leakage_analysis.png', dpi=300)
        plt.close()

    except Exception as e:
        logger.warning(f"Failed to plot leakage analysis: {str(e)}")


def _process_feature_chunk(args: Tuple[List[str], pd.DataFrame, float]) -> List[Tuple[str, str, float]]:
    """
    Process a chunk of features to find redundant pairs.

    Args:
        args: Tuple containing (features, corr_matrix, threshold)

    Returns:
        List of redundant feature pairs with correlation values
    """
    feature_list, corr_matrix, threshold = args
    redundant = []
    for i, feature_a in enumerate(feature_list):
        for feature_b in feature_list[i + 1:]:
            correlation = abs(corr_matrix.at[feature_a, feature_b])
            if correlation > threshold:
                redundant.append((feature_a, feature_b, float(correlation)))
    return redundant


def _calculate_importance_score(feature: str,
                                stability_scores: Dict[str, float],
                                leakage_info: List[Dict[str, Any]],
                                feature_correlations: Dict[str, List[Tuple[str, float]]]) -> float:
    """
    Calculate overall importance score for a feature.

    Args:
        feature: Feature name
        stability_scores: Dictionary of stability scores
        leakage_info: List of dictionaries containing leakage information
        feature_correlations: Dictionary of feature correlations

    Returns:
        Combined importance score
    """
    # Convert infinity to large number for stability
    stability = 1.0 / (1.0 + stability_scores.get(feature, float('inf')))

    # Get leakage score
    leakage = next((info['correlation'] for info in leakage_info
                    if info['feature'] == feature), 0.0)

    # Get maximum correlation with other features
    correlation = max((abs(correlation_value)
                       for _, correlation_value in feature_correlations.get(feature, [])),
                      default=0.0)

    # Combine scores (lower is better)
    return stability + abs(leakage) + correlation


def select_features_parallel(X: np.ndarray,
                             y: np.ndarray,
                             feature_names: List[str],
                             correlation_threshold: float = 0.95,
                             leakage_threshold: float = 0.9,
                             n_jobs: Optional[int] = None,
                             save_dir: Optional[Path] = None) -> Tuple[List[str], Dict[str, Any]]:
    """
    Select optimal features using parallel processing.

    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        correlation_threshold: Threshold for feature correlation
        leakage_threshold: Threshold for data leakage detection
        n_jobs: Number of processes to use
        save_dir: Optional directory to save results

    Returns:
        Tuple of (selected_features, selection_metadata)
    """
    try:
        if n_jobs is None or n_jobs < 1:
            n_jobs = max(1, cpu_count() - 1)

        logger.info(f"Starting parallel feature selection with {n_jobs} processes")

        # Step 1: Calculate correlations
        logger.info("Calculating feature correlations...")
        correlation_matrix, feature_correlations = calculate_feature_correlations(
            X, feature_names, n_jobs=n_jobs
        )

        # Step 2: Identify potential data leakage
        logger.info("Detecting potential data leakage...")
        leakage_info = detect_data_leakage(
            X, y, feature_names,
            threshold=leakage_threshold,
            n_jobs=n_jobs,
            save_dir=save_dir / 'leakage' if save_dir else None
        )

        # Step 3: Analyze feature stability
        logger.info("Analyzing feature stability...")
        stability_scores = parallel_feature_stability(
            X, feature_names,
            n_jobs=n_jobs,
            save_dir=save_dir / 'stability' if save_dir else None
        )

        # Step 4: Process redundant features
        chunk_size = max(1, len(feature_names) // (n_jobs * 2))
        feature_chunks = [
            feature_names[i:i + chunk_size]
            for i in range(0, len(feature_names), chunk_size)
        ]

        chunk_args = [
            (chunk, correlation_matrix, correlation_threshold)
            for chunk in feature_chunks
        ]

        with Pool(n_jobs) as pool:
            redundant_pairs = [
                pair for chunk in tqdm(
                    pool.imap(_process_feature_chunk, chunk_args),
                    total=len(chunk_args),
                    desc="Processing feature chunks"
                )
                for pair in chunk
            ]

        # Step 5: Feature Selection Logic
        features_to_remove: Set[str] = set()

        # Remove features with data leakage
        leakage_features = {
            info['feature'] for info in leakage_info
        }
        features_to_remove.update(leakage_features)

        # Process redundant pairs
        processed_pairs: Set[Tuple[str, str]] = set()
        for feat1, feat2, _ in redundant_pairs:
            if (feat1, feat2) not in processed_pairs and (feat2, feat1) not in processed_pairs:
                processed_pairs.add((feat1, feat2))

                # Skip if either feature is already marked for removal
                if feat1 in features_to_remove or feat2 in features_to_remove:
                    continue

                # Keep the more stable feature
                if stability_scores.get(feat1, float('inf')) > stability_scores.get(feat2, float('inf')):
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)

        # Create final feature list
        selected_features = [f for f in feature_names if f not in features_to_remove]

        # Calculate importance scores
        importance_scores = {
            feature: _calculate_importance_score(
                feature, stability_scores, leakage_info, feature_correlations
            )
            for feature in selected_features
        }

        # Create selection metadata
        selection_metadata = {
            'n_original_features': len(feature_names),
            'n_selected_features': len(selected_features),
            'removed_features': list(features_to_remove),
            'removal_reasons': {
                'leakage': list(leakage_features),
                'redundancy': [(f1, f2) for f1, f2 in processed_pairs],
                'low_stability': [f for f in features_to_remove
                                  if f not in leakage_features and
                                  not any(f in pair for pair in processed_pairs)]
            },
            'feature_importance': dict(sorted(
                importance_scores.items(),
                key=lambda x: x[1]
            )),
            'stability_scores': stability_scores,
            'correlation_threshold': correlation_threshold,
            'leakage_threshold': leakage_threshold,
            'selection_timestamp': datetime.now().isoformat()
        }

        # Save selection results
        if save_dir:
            save_selection_results(
                selected_features,
                selection_metadata,
                save_dir
            )

        logger.info(f"\nFeature selection complete:")
        logger.info(f"  Original features: {len(feature_names)}")
        logger.info(f"  Selected features: {len(selected_features)}")
        logger.info(f"  Removed features: {len(features_to_remove)}")
        logger.info(f"    - Leakage: {len(leakage_features)}")
        logger.info(f"    - Redundancy: {len(processed_pairs)}")

        return selected_features, selection_metadata

    except Exception as e:
        logger.error(f"Feature selection failed: {str(e)}")
        raise FeatureAnalysisError(f"Feature selection failed: {str(e)}")


def save_selection_results(selected_features: List[str],
                           metadata: Dict[str, Any],
                           save_dir: Path) -> None:
    """
    Save feature selection results and visualizations.

    Args:
        selected_features: List of selected feature names
        metadata: Selection metadata dictionary
        save_dir: Directory to save results
    """
    try:
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save main results
        with open(save_dir / 'feature_selection_results.json', 'w') as f:
            json.dump({
                'selected_features': selected_features,
                'metadata': metadata
            }, f, indent=4)

        # Create visualizations
        plot_feature_importance(metadata['feature_importance'], save_dir)
        plot_feature_removal_reasons(metadata['removal_reasons'], save_dir)

    except Exception as e:
        logger.error(f"Failed to save selection results: {str(e)}")


def plot_feature_importance(importance_dict: Dict[str, float],
                            save_dir: Path) -> None:
    """
    Create and save feature importance visualization.

    Args:
        importance_dict: Dictionary of feature importance scores
        save_dir: Directory to save visualization
    """
    try:
        plt.figure(figsize=(15, 10))

        # Sort features by importance
        sorted_features = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        # Plot top 30 features
        features = list(sorted_features.keys())[:30]
        scores = [sorted_features[f] for f in features]

        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title('Top 30 Most Important Features')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(save_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"Failed to plot feature importance: {str(e)}")


def plot_feature_removal_reasons(removal_reasons: Dict[str, List],
                                 save_dir: Path) -> None:
    """
    Create and save visualization of feature removal reasons.

    Args:
        removal_reasons: Dictionary of removal reasons and affected features
        save_dir: Directory to save visualization
    """
    try:
        plt.figure(figsize=(10, 6))

        reasons = list(removal_reasons.keys())
        counts = [len(removal_reasons[r]) for r in reasons]

        plt.bar(reasons, counts)
        plt.ylabel('Number of Features')
        plt.title('Features Removed by Reason')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(save_dir / 'removal_reasons.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save detailed removal information
        with open(save_dir / 'removal_details.json', 'w') as f:
            json.dump(removal_reasons, f, indent=4)

    except Exception as e:
        logger.error(f"Failed to plot removal reasons: {str(e)}")


def analyze_features(X: np.ndarray,
                     y: np.ndarray,
                     feature_names: List[str],
                     n_jobs: Optional[int] = None,
                     save_dir: Optional[Path] = None) -> Tuple[List[str], Dict[str, Any]]:
    """
    Main feature analysis pipeline.

    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        n_jobs: Number of parallel jobs
        save_dir: Optional directory to save results

    Returns:
        Tuple of (selected_features, analysis_results)
    """
    try:
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Select optimal features
        selected_features, selection_metadata = select_features_parallel(
            X, y, feature_names,
            n_jobs=n_jobs,
            save_dir=save_dir
        )

        # Calculate final metrics
        importance_dict = selection_metadata['feature_importance']
        top_features = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])

        logger.info("\nTop 5 features by importance:")
        for feature, score in top_features.items():
            logger.info(f"  {feature}: {score:.4f}")

        return selected_features, selection_metadata

    except Exception as e:
        logger.error(f"Feature analysis failed: {str(e)}")
        raise FeatureAnalysisError(f"Feature analysis failed: {str(e)}")
