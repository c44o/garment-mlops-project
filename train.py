"""Train garment productivity regressors, track experiments, export best pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'garments_worker_productivity.csv'
MODEL_PATH = ROOT / 'models' / 'model.joblib'
METRICS_PATH = ROOT / 'artifacts' / 'metrics.json'
NUMERIC = [
    'targeted_productivity', 'smv', 'wip', 'over_time', 'incentive',
    'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers',
]
CATEGORICAL = ['quarter', 'department', 'day', 'team']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'actual_productivity'


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result['department'] = (
        result['department'].astype(str).str.strip().str.lower()
        .replace({'sweing': 'sewing'})  # Typo in the original Kaggle CSV.
    )
    return result


def create_pipeline(n_estimators: int, max_depth: int | None) -> Pipeline:
    preparation = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), NUMERIC),
        ('cat', Pipeline([
            ('fill', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore')),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ('preparation', preparation),
        ('regressor', RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, n_jobs=-1,
        )),
    ])


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    prediction = model.predict(x)
    return {
        'mae': float(mean_absolute_error(y, prediction)),
        'rmse': float(mean_squared_error(y, prediction) ** 0.5),
        'r2': float(r2_score(y, prediction)),
    }


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f'Download the CSV from Kaggle and put it at {DATA_PATH}'
        )
    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    df = normalize(pd.read_csv(DATA_PATH))
    missing = set(FEATURES + [TARGET, 'date']) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df['parsed_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='raise')
    df = df.sort_values('parsed_date').reset_index(drop=True)

    # Split by unique dates (not individual rows): train / validation / test.
    dates = sorted(df['parsed_date'].unique())
    if len(dates) < 10:
        raise ValueError('At least 10 unique dates are required for a time split')
    cut_train = dates[int(len(dates) * 0.60)]
    cut_val = dates[int(len(dates) * 0.80)]
    train = df[df['parsed_date'] < cut_train]
    val = df[(df['parsed_date'] >= cut_train) & (df['parsed_date'] < cut_val)]
    test = df[df['parsed_date'] >= cut_val]

    ROOT.joinpath('models').mkdir(exist_ok=True)
    ROOT.joinpath('artifacts').mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').resolve().as_posix()}")
    # SQLite backend keeps all experiments locally; large model artifacts go into mlruns.
    mlflow.set_experiment('garment-productivity')
    candidates = [
        {'n_estimators': 80, 'max_depth': 5},
        {'n_estimators': 120, 'max_depth': 10},
    ]
    best = None
    for params in candidates:
        with mlflow.start_run(run_name=f"rf_{params['n_estimators']}_depth_{params['max_depth']}"):
            pipeline = create_pipeline(**params)
            pipeline.fit(train[FEATURES], train[TARGET])
            scores = evaluate(pipeline, val[FEATURES], val[TARGET])
            mlflow.log_params(params)
            mlflow.log_param('dataset_sha256', data_hash)
            mlflow.log_param('split', 'chronological_dates_60_20_20')
            mlflow.log_metrics({f'val_{k}': v for k, v in scores.items()})
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )
            print('Candidate:', params, 'Validation:', scores)
            if best is None or scores['mae'] < best['validation']['mae']:
                best = {'params': params, 'validation': scores}

    assert best is not None
    # Refit on train+validation; evaluate once on untouched final test set.
    train_val = df[df['parsed_date'] < cut_val]
    final_model = create_pipeline(**best['params'])
    final_model.fit(train_val[FEATURES], train_val[TARGET])
    test_scores = evaluate(final_model, test[FEATURES], test[TARGET])
    joblib.dump(final_model, MODEL_PATH)
    summary = {
        'source': 'Kaggle: ish adss productivity prediction of garment employees (ishadss)',
        'dataset_sha256': data_hash,
        'features': FEATURES,
        'target': TARGET,
        'train_rows': len(train),
        'validation_rows': len(val),
        'test_rows': len(test),
        'selected_hyperparameters': best['params'],
        'validation': best['validation'],
        'test': test_scores,
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with mlflow.start_run(run_name='final_selected_model'):
        mlflow.log_params(best['params'])
        mlflow.log_param('dataset_sha256', data_hash)
        mlflow.log_metrics({f'test_{k}': v for k, v in test_scores.items()})
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.sklearn.log_model(
            final_model,
            name="model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

    print('Final test metrics:', test_scores)
    print('Saved:', MODEL_PATH, METRICS_PATH)


if __name__ == '__main__':
    main()
