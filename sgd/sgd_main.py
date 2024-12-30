import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np

import config.sgd_config as cfg
from feature_analyzer import (
    analyze_features,
    FeatureAnalysisError
)
from sgd_classifier import (
    AnomalyClassifier,
    ClassifierError
)
from sgd_data_loader import (
    prepare_dataset,
    DataLoadingError
)
from sgd_evaluator import (
    evaluate_classifier,
    perform_cross_validation,
    EvaluationError
)


class PipelineError(Exception):
    """Custom exception for pipeline execution errors."""
    pass


def setup_logging(output_dir: Path) -> None:
    """
    Configure logging for the pipeline.

    Args:
        output_dir: Directory for log files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"pipeline_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def convert_config_for_json(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert configuration dictionary to JSON-serializable format.

    Args:
        config: Configuration dictionary

    Returns:
        JSON-serializable configuration dictionary
    """
    def _convert_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.generic):
            return value.item()
        elif isinstance(value, dict):
            return {k: _convert_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [_convert_value(v) for v in value]
        return value

    return {k: _convert_value(v) for k, v in config.items()}

def initialize_pipeline() -> Tuple[Path, datetime]:
    """
    Initialize the pipeline execution environment.

    Returns:
        Tuple of (run_directory_path, timestamp)
    """
    try:
        timestamp = datetime.now()
        run_dir = cfg.OUTPUT_DIR / timestamp.strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        setup_logging(run_dir)

        # Convert and save configuration
        config_data = {
            'timestamp': timestamp.isoformat(),
            'data_config': convert_config_for_json(cfg.DATA_CONFIG),
            'model_config': convert_config_for_json(cfg.MODEL_CONFIG),
            'feature_config': convert_config_for_json(cfg.FEATURE_CONFIG),
            'evaluation_config': convert_config_for_json(cfg.EVALUATION_CONFIG),
            'cross_validation_config': convert_config_for_json(cfg.CROSS_VALIDATION_CONFIG)
        }

        with open(run_dir / 'run_config.json', 'w') as f:
            json.dump(config_data, f, indent=4)

        logging.info(f"Pipeline initialized in directory: {run_dir}")
        return run_dir, timestamp

    except Exception as e:
        logging.error(f"Pipeline initialization failed: {str(e)}")
        raise PipelineError(f"Pipeline initialization failed: {str(e)}")


def save_results(results: Dict[str, Any], run_dir: Path) -> None:
    """
    Save pipeline results and generate summary.

    Args:
        results: Pipeline results dictionary
        run_dir: Run directory path
    """
    try:
        # Convert numpy types to native Python types
        def convert_to_native_types(obj: Any) -> Any:
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_native_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native_types(x) for x in obj]
            return obj

        # Create serializable results dictionary
        serializable_results = {
            'evaluation_metrics': convert_to_native_types(results['evaluation_metrics']),
            'cross_validation_results': {
                'average_scores': convert_to_native_types(results['cross_validation_results']['average_scores']),
                'metric_scores': convert_to_native_types(results['cross_validation_results']['metric_scores']),
                'feature_importance': convert_to_native_types(
                    results['cross_validation_results'].get('feature_importance', {})
                )
            },
            'model_parameters': {
                str(k): str(v) for k, v in results['model_parameters'].items()
            },
            'timestamp': results['timestamp'],
            'feature_analysis': convert_to_native_types(results['feature_analysis'])
        }

        # Save detailed results
        with open(run_dir / 'pipeline_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=4)

        # Generate and save summary
        summary = {
            'timestamp': results['timestamp'],
            'metrics': convert_to_native_types(results['evaluation_metrics']),
            'model_parameters': {
                str(k): str(v) for k, v in results['model_parameters'].items()
            },
            'feature_analysis': {
                'n_original_features': len(results.get('feature_names', [])),
                'n_selected_features': results['feature_analysis']['n_selected_features'],
                'removed_features': results['feature_analysis']['removed_features']
            }
        }

        with open(run_dir / 'results_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)

        # Log summary metrics
        logging.info("Pipeline Results Summary:")
        for metric, value in summary['metrics'].items():
            if isinstance(value, (int, float)):
                logging.info(f"{metric}: {value:.4f}")
            else:
                logging.info(f"{metric}: {value}")

    except Exception as e:
        logging.error(f"Failed to save results: {str(e)}")


def execute_pipeline() -> None:
    """
    Execute the complete anomaly detection pipeline.

    Raises:
        PipelineError: If any pipeline stage fails
    """
    try:
        # Initialize pipeline
        run_dir, timestamp = initialize_pipeline()
        logging.info("Starting pipeline execution")

        # Create directory structure
        analysis_dir = run_dir / 'analysis'
        model_dir = run_dir / 'model'
        evaluation_dir = run_dir / 'evaluation'
        cv_dir = run_dir / 'cross_validation'

        for directory in [analysis_dir, model_dir, evaluation_dir, cv_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Prepare datasets
        logging.info("Preparing datasets")
        try:
            train_data, val_data, test_data, feature_names = prepare_dataset(
                test_size=cfg.DATA_CONFIG['test_size'],
                validation_size=cfg.DATA_CONFIG['validation_size'],
                random_state=cfg.DATA_CONFIG['random_state']
            )
            X_train, y_train, train_file_ids = train_data
            X_val, y_val, val_file_ids = val_data
            X_test, y_test, test_file_ids = test_data

        except DataLoadingError as e:
            raise PipelineError(f"Dataset preparation failed: {str(e)}")

        logging.info(f"Initial dataset shapes - Train: {X_train.shape}, "
                     f"Val: {X_val.shape}, Test: {X_test.shape}")

        # Perform feature analysis
        logging.info("Performing feature analysis")
        try:
            selected_features, analysis_results = analyze_features(
                X=X_train,
                y=y_train,
                feature_names=feature_names,
                n_jobs=cfg.DATA_CONFIG['n_jobs'],
                save_dir=analysis_dir
            )

            # Apply feature selection
            feature_mask = [feature in selected_features for feature in feature_names]
            X_train_selected = X_train[:, feature_mask]
            X_val_selected = X_val[:, feature_mask]
            X_test_selected = X_test[:, feature_mask]

            logging.info(f"Selected {len(selected_features)} features out of {len(feature_names)}")
            logging.info(f"Selected feature shapes - Train: {X_train_selected.shape}, "
                         f"Val: {X_val_selected.shape}, Test: {X_test_selected.shape}")

        except FeatureAnalysisError as e:
            raise PipelineError(f"Feature analysis failed: {str(e)}")

        # Perform cross-validation if enabled
        cv_results = None
        if cfg.CROSS_VALIDATION_CONFIG["enabled"]:
            logging.info("Performing cross-validation")
            try:
                def train_model_cv(X_cv: np.ndarray, y_cv: np.ndarray) -> AnomalyClassifier:
                    clf = AnomalyClassifier()
                    clf.fit(X_cv, y_cv, feature_names=selected_features, optimize=True)
                    return clf

                cv_results = perform_cross_validation(
                    X=X_train_selected,
                    y=y_train,
                    file_ids=train_file_ids,
                    model_trainer=train_model_cv,
                    save_path=cv_dir
                )

                logging.info("\nCross-validation Results:")
                for metric, stats in cv_results['average_scores'].items():
                    logging.info(f"{metric}:")
                    logging.info(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
                    logging.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

            except Exception as e:
                raise PipelineError(f"Cross-validation failed: {str(e)}")

        # Create and train final classifier
        logging.info("Training final classifier")
        try:
            classifier = AnomalyClassifier()
            classifier.fit(
                X_train_selected,
                y_train,
                feature_names=selected_features,
                optimize=True
            )
        except ClassifierError as e:
            raise PipelineError(f"Model training failed: {str(e)}")

        # Generate predictions
        y_pred = classifier.predict(X_test_selected)
        y_prob = classifier.predict_proba(X_test_selected)

        # Evaluate model
        logging.info("Evaluating model")
        try:
            evaluation_results = evaluate_classifier(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                model=classifier.model,
                feature_names=selected_features,
                save_path=evaluation_dir
            )
        except EvaluationError as e:
            raise PipelineError(f"Model evaluation failed: {str(e)}")

        # Save model and results
        classifier.save(model_dir)

        # Combine all results
        results = {
            'evaluation_metrics': evaluation_results,
            'cross_validation_results': cv_results,
            'model_parameters': classifier.model.get_params(),
            'timestamp': timestamp.isoformat(),
            'data_config': cfg.DATA_CONFIG,
            'model_config': cfg.MODEL_CONFIG,
            'feature_names': selected_features,
            'feature_analysis': {
                'n_original_features': len(feature_names),
                'n_selected_features': len(selected_features),
                'removed_features': list(set(feature_names) - set(selected_features)),
                'analysis_results': analysis_results
            }
        }

        # Save results
        save_results(results, run_dir)
        logging.info("Pipeline execution completed successfully")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise PipelineError(f"Pipeline execution failed: {str(e)}")
    finally:
        if cfg.DATA_CONFIG.get("clear_cache", True):
            gc.collect()


def main() -> None:
    """
    Main entry point for the anomaly detection pipeline.
    """
    try:
        execute_pipeline()
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
