"""Generate the ten screenshot-selected checkpoint submissions and training MAEs."""

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.dashboard_config import validate_dashboard_config
from src.dashboard_model import DashboardClaimModel
from src.data_loading import DEFAULT_TEST_PATH, DEFAULT_TRAINING_PATH, load_test_data, load_training_data
from src.feature_schema import ALL_FEATURE_COLUMNS, MAX_RANDOM_SEED, RANDOM_SEED, ROW_INDEX_COLUMN, TARGET_COLUMN

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = ROOT / 'configs/checkpoint2'
DEFAULT_OUTPUT_DIRECTORY = ROOT / 'data/checkpoint2'


def load_selection():
    selection = json.loads((CONFIG_DIRECTORY / 'selection.json').read_text(encoding='utf-8'))
    names = [entry['name'] for entry in selection['submissions']]
    if names != [f'2_2_{index}' for index in range(1, 11)]:
        raise ValueError('Expected the ten selected submissions in screenshot order.')
    return selection


def load_configuration(name, seed):
    config = json.loads((CONFIG_DIRECTORY / f'{name}.json').read_text(encoding='utf-8'))
    config['random_seed'] = seed
    if config['classifier']['family'] != 'checkpoint_blend':
        validate_dashboard_config(config)
        return config
    classifier = config['classifier']
    weights = np.asarray(classifier['weights'], dtype=float)
    if weights.shape != (2,) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Blend weights must be two nonnegative numbers summing to one.')
    if config['severity_method'] != 'blended_weighted_profile' or config['amount_cap'] is not None:
        raise ValueError('Selected blends use uncapped probability-weighted primary profile amounts.')
    for component in ['primary', 'secondary']:
        validate_dashboard_config(dict(config, classifier=classifier[component], severity_method='profile_median'))
    return config


def fitted_models(config, features, amounts, cache):
    classifier = config['classifier']
    settings_list = ([classifier['primary'], classifier['secondary']]
                     if classifier['family'] == 'checkpoint_blend' else [classifier])
    models = []
    for settings in settings_list:
        key = json.dumps(dict(settings={key: value for key, value in settings.items() if key != 'name'},
                              alpha=config['ridge_alpha'], seed=config['random_seed']), sort_keys=True)
        if key not in cache:
            cache[key] = DashboardClaimModel(settings, ridge_alpha=config['ridge_alpha'],
                                             random_seed=config['random_seed']).fit(features, amounts)
        models.append(cache[key])
    return models


def profile_amount(primary_model, features):
    ridge_amount = np.maximum(primary_model.regressor_.predict(features), 0.0)
    return primary_model.history_.predict_amount(features, ridge_amount)


def predict_configuration(config, models, features):
    if config['classifier']['family'] == 'checkpoint_blend':
        first, second = models
        weights = config['classifier']['weights']
        probability = (weights[0] * first.predict_claim_probability(features)
                       + weights[1] * second.predict_claim_probability(features))
        amount = profile_amount(first, features)
        return np.where(probability >= config['threshold'],
                        probability * amount * config['severity_scale'], 0.)
    model = models[0]
    # Fits can be reused, but the final threshold and amount policy belong to each submission.
    model.set_params(claim_probability_threshold=config['threshold'], severity_method=config['severity_method'],
                     severity_scale=config['severity_scale'], amount_cap=config['amount_cap'])
    return model.predict(features)


def software_versions():
    versions = {'python': platform.python_version()}
    for package in ['numpy', 'pandas', 'scikit-learn', 'lightgbm']:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def write_submission(entry, config, models, training, test, output_directory, data_hash, metrics_directory=None):
    actual = training[TARGET_COLUMN].to_numpy()
    training_prediction = predict_configuration(config, models, training[ALL_FEATURE_COLUMNS])
    prediction = predict_configuration(config, models, test[ALL_FEATURE_COLUMNS])
    for values in [training_prediction, prediction]:
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'{entry["name"]}: predictions must be finite and nonnegative.')
    metrics = dict(name=entry['name'], previous_name=entry['previous_name'], training_seed=config['random_seed'],
                   training_rows=len(training), training_claims=int((actual > 0).sum()),
                   training_mae=float(np.abs(actual - training_prediction).mean()),
                   training_zero_mae=float(np.abs(actual).mean()), test_rows=len(test),
                   nonzero_predictions=int((prediction > 0).sum()), training_data_sha256=data_hash,
                   configuration_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                   software_versions=software_versions(),
                   interpretation='In-sample MAE after fitting all rows; profile history includes those rows.')
    pd.DataFrame({ROW_INDEX_COLUMN: test[ROW_INDEX_COLUMN], TARGET_COLUMN: prediction}).to_csv(
        output_directory / f'{entry["name"]}.csv', index=False)
    metadata_directory = metrics_directory if metrics_directory is not None else output_directory
    (metadata_directory / f'{entry["name"]}.metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    print(f'{entry["name"]}: training MAE={metrics["training_mae"]:.4f}; '
          f'nonzero predictions={metrics["nonzero_predictions"]}/{len(test)}', flush=True)


def write_upload_comments(output_directory, entries, metrics_directory=None, comments_directory=None):
    metadata_directory = metrics_directory if metrics_directory is not None else output_directory
    destination = comments_directory if comments_directory is not None else output_directory
    lines = ['Checkpoint 2 / Group 2 - copy the matching training MAE into each CSV upload comment.', '']
    for entry in entries:
        path = metadata_directory / f'{entry["name"]}.metrics.json'
        if not path.exists() or not (output_directory / f'{entry["name"]}.csv').exists():
            continue
        metrics = json.loads(path.read_text(encoding='utf-8'))
        if metrics['previous_name'] != entry['previous_name']:
            continue
        lines.append(f'{entry["name"]}.csv: Training MAE (all {metrics["training_rows"]:,} training rows, '
                     f'seed {metrics["training_seed"]}) = {metrics["training_mae"]:.6f}.')
    (destination / 'checkpoint2_upload_comments.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    selection = load_selection()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', choices=[entry['name'] for entry in selection['submissions']])
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIRECTORY,
                        help='CSV destination; a custom folder also receives its training metrics and upload comments.')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    if not 0 <= args.seed <= MAX_RANDOM_SEED:
        parser.error('Seed is outside the supported range.')
    training_path = ROOT / 'data' / DEFAULT_TRAINING_PATH.name
    test_path = ROOT / 'data' / DEFAULT_TEST_PATH.name
    data_hash = hashlib.sha256(training_path.read_bytes()).hexdigest()
    if data_hash != selection['training_data_sha256']:
        raise ValueError('Training data differs from the frozen selection; review the configuration evidence first.')
    training, test = load_training_data(training_path), load_test_data(test_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the repository's organised layout; custom destinations remain self-contained.
    default_destination = args.output_dir.resolve() == DEFAULT_OUTPUT_DIRECTORY.resolve()
    metrics_directory = ROOT / 'data/checkpoint2 metrics' if default_destination else args.output_dir
    comments_directory = ROOT / 'data' if default_destination else args.output_dir
    metrics_directory.mkdir(parents=True, exist_ok=True)
    cache = {}
    entries = [entry for entry in selection['submissions'] if not args.submission or entry['name'] == args.submission]
    for entry in entries:
        config = load_configuration(entry['name'], args.seed)
        models = fitted_models(config, training[ALL_FEATURE_COLUMNS], training[TARGET_COLUMN].to_numpy(), cache)
        write_submission(entry, config, models, training, test, args.output_dir, data_hash, metrics_directory)
    write_upload_comments(args.output_dir, selection['submissions'], metrics_directory, comments_directory)
    print(f'Wrote {len(entries)} selected submissions and training metrics to {args.output_dir}.', flush=True)


if __name__ == '__main__':
    main()
