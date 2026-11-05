"""Serve the evaluated pipeline using its recorded classification threshold."""
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / 'artifacts_19'


class LoanApplication(BaseModel):
    """Numeric inputs at loan issuance; explicit nulls use training medians."""

    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

    loan_amnt: float = Field(gt=0, description='Loan amount in US dollars.')
    term: Literal[36, 60] = Field(description='Loan term in months: 36 or 60.')
    int_rate: float = Field(gt=0, le=100, description='Annual rate in percentage points: 13.99 means 13.99%.')
    installment: float = Field(gt=0, description='Monthly payment in US dollars.')
    grade: int = Field(ge=0, le=6, description='LendingClub grade: A=0, B=1, ..., G=6.')
    emp_length: int | None = Field(ge=0, le=10, description='Employment years: <1=0, 10+=10. Use null if unknown.')
    annual_inc: float | None = Field(ge=0, description='Annual income in US dollars; null if unknown.')
    pymnt_plan: Literal[0] = Field(default=0, description='Inactive compatibility field. Always 0; not used as a predictor.')
    dti: float | None = Field(ge=0, description='Debt-to-income ratio in percentage points; null if unknown.')
    delinq_2yrs: int | None = Field(ge=0, description='Number of delinquencies in the preceding two years.')
    fico_range_low: float | None = Field(ge=300, le=850, description='Lower FICO bound at issuance.')
    fico_range_high: float | None = Field(ge=300, le=850, description='Upper FICO bound at issuance.')
    inq_last_6mths: int | None = Field(ge=0, description='Credit inquiries in the preceding six months.')
    open_acc: int | None = Field(ge=0, description='Number of open credit accounts.')
    pub_rec: int | None = Field(ge=0, description='Number of derogatory public records.')
    revol_bal: float | None = Field(ge=0, description='Revolving credit balance in US dollars.')
    revol_util: float | None = Field(ge=0, le=100, description='Revolving utilization in percentage points, 0-100.')
    total_acc: int | None = Field(ge=0, description='Total number of credit accounts.')
    credit_history_years: int | None = Field(ge=0, description='Issue calendar year minus earliest-credit-line calendar year.')

    @model_validator(mode='after')
    def validate_fico_bounds(self):
        if (self.fico_range_low is not None and self.fico_range_high is not None
                and self.fico_range_low > self.fico_range_high):
            raise ValueError('fico_range_low must not exceed fico_range_high')
        return self


class Prediction(BaseModel):
    prediction: Literal['Default', 'Fully Paid']
    default_probability: float = Field(ge=0, le=1, description='Model score; calibration as a real-world default probability is not established.')
    risk_level: Literal['High', 'Low']
    threshold: float = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    features = json.loads((ARTIFACT_DIR / 'feature_columns_19.json').read_text(encoding='utf-8'))
    metadata = json.loads((ARTIFACT_DIR / 'metrics_19.json').read_text(encoding='utf-8'))
    pipeline = joblib.load(ARTIFACT_DIR / 'loan_default_19_pipeline.joblib')
    threshold = float(metadata['test_results']['lightgbm']['threshold'])
    if not (math.isfinite(threshold) and 0 <= threshold <= 1):
        raise RuntimeError('Invalid threshold in metrics_19.json')
    if len(features) != 19 or len(set(features)) != 19:
        raise RuntimeError('Expected exactly 19 unique input features')
    if set(features) != set(LoanApplication.model_fields):
        raise RuntimeError('API schema and artifact feature names do not match')
    if list(pipeline.feature_names_in_) != features or metadata['features'] != features:
        raise RuntimeError('Model, feature order, and metrics do not match')
    if list(pipeline.classes_) != [0, 1]:
        raise RuntimeError('Expected classes [0, 1], with charge-off encoded as 1')
    app.state.pipeline = pipeline
    app.state.features = features
    app.state.threshold = threshold
    yield
    app.state.pipeline = None


app = FastAPI(
    title='Loan Default Prediction API',
    version='2.0.0',
    description=(
        'Educational model of charge-off versus full repayment among resolved, accepted loans. '
        'Uses the evaluated 19-field pipeline and validation-selected threshold. '
        'Scores are not established as calibrated probabilities or suitable for lending decisions.'
    ),
    lifespan=lifespan,
)


@app.get('/')
def root():
    return {'message': 'Loan Default Prediction API is running!', 'docs': '/docs', 'version': app.version}


@app.get('/health')
def health(request: Request):
    return {
        'status': 'ok',
        'model_loaded': request.app.state.pipeline is not None,
        'expected_features': len(request.app.state.features),
        'model_version': 'evaluated-19-input-v1',
        'threshold': request.app.state.threshold,
    }


@app.post('/predict', response_model=Prediction)
def predict(loan: LoanApplication, request: Request):
    values = pd.DataFrame([loan.model_dump()], columns=request.app.state.features, dtype=float)
    values = values.fillna(np.nan)
    score = float(request.app.state.pipeline.predict_proba(values)[0, 1])
    is_default = score >= request.app.state.threshold
    return Prediction(
        prediction='Default' if is_default else 'Fully Paid',
        default_probability=score,
        risk_level='High' if is_default else 'Low',
        threshold=request.app.state.threshold,
    )
