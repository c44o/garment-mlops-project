# Garment Productivity — Kaggle MLOps Project (questions 46–49)

A small CPU-only ML pipeline: Kaggle dataset -> RandomForest regression -> MLflow experiment tracking -> saved sklearn pipeline -> FastAPI -> Docker.

## Dataset

- Kaggle: https://www.kaggle.com/datasets/ishadss/productivity-prediction-of-garment-employees
- Original UCI Dataset #597, CC BY 4.0: https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees
- Download and save `data/garments_worker_productivity.csv` (NOT included in this starter archive).

## Setup (Windows 11, PowerShell)

Install Python 3.12. In this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Question 46: train and save model

```powershell
python train.py
```

Results: `models/model.joblib`, `artifacts/metrics.json` and `mlflow.db` (local tracking database).
Model selection uses chronological train/validation/test splits, comparing two RandomForest hyperparameter settings by validation MAE, with test scores measured only on the refitted selected pipeline.

## Question 48 + 49: MLflow experiments and UI

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 . Take a screenshot of runs and validation metrics. All MLflow logging is inside `train.py`.

## Question 47: container

Start Docker Desktop, check `docker info`, then:

```powershell
docker build -t garment-mlops:1.0 .
docker run --rm -p 8000:8000 garment-mlops:1.0
```

Open http://localhost:8000/docs . Use the **POST /predict** form, sample input in `example-request.json`.
In another PowerShell window:

```powershell
Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/predict' -ContentType 'application/json' -InFile './example-request.json'
```

### What to submit

- Q46: `models/model.joblib`, `artifacts/metrics.json`, `train.py` (zip them; include data provenance).
- Q47: GitHub repo link with `Dockerfile`, `app.py`, requirements and README; if possible a Docker build/run screenshot.
- Q48: `train.py` or repo link to its logging code.
- Q49: `python train.py` plus `mlflow ui --backend-store-uri sqlite:///mlflow.db ...` and MLflow UI screenshot.

If sharing the trained model publicly, remember joblib/pickle files must only be loaded from trusted sources. Avoid publishing tokens, secrets or the local MLflow SQLite database.

## Reproducibility

Save the resolved local environment after installation as `python -m pip freeze > requirements-lock.txt`; for strict byte-identical reproducibility use the same resolved dependency versions on training and inference. The lightweight Dockerfile uses a separate inference dependency set; after verifying the training setup, pin inference dependencies to the exact resolved versions of pandas, scikit-learn and joblib used for training.

## License / attribution

Dataset is shared under CC BY 4.0 by its original authors on UCI. Credit: Abdullah Al Imran, Md Shamsur Rahim, Tanvir Ahmed, *Productivity Prediction of Garment Employees*, UCI Dataset 597. This project is an educational example and should not be used to make employment decisions.
