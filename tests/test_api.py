"""Integration checks against the saved, evaluated model and reference example."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.main import app

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / 'artifacts_19/example_request.json').read_text())
EXPECTED = json.loads((ROOT / 'artifacts_19/example_response.json').read_text())


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as instance:
        yield instance


def test_health_and_documentation(client):
    health = client.get('/health')
    assert health.status_code == 200
    assert health.json()['model_loaded'] is True
    assert health.json()['expected_features'] == 19
    assert health.json()['threshold'] == EXPECTED['threshold']
    assert client.get('/').status_code == 200
    assert client.get('/docs').status_code == 200
    schema = client.get('/openapi.json').json()
    assert len(schema['components']['schemas']['LoanApplication']['properties']) == 19


def test_prediction_matches_notebook_reference(client):
    response = client.post('/predict', json=EXAMPLE)
    assert response.status_code == 200
    result = response.json()
    assert result['default_probability'] == pytest.approx(EXPECTED['default_probability'], abs=1e-12)
    assert result['prediction'] == EXPECTED['prediction']
    assert result['threshold'] == EXPECTED['threshold']
    assert result['risk_level'] == 'Low'


def test_uses_evaluated_threshold_instead_of_point_five(client):
    request = {**EXAMPLE, 'grade': 4, 'int_rate': 20.0, 'annual_inc': 40000, 'dti': 25.0}
    response = client.post('/predict', json=request)
    assert response.status_code == 200
    result = response.json()
    assert result['threshold'] <= result['default_probability'] < 0.5
    assert result['prediction'] == 'Default'
    assert result['risk_level'] == 'High'


def test_unknown_employment_uses_training_imputation(client):
    unknown = client.post('/predict', json={**EXAMPLE, 'emp_length': None})
    median = client.post('/predict', json={**EXAMPLE, 'emp_length': 6})
    assert unknown.status_code == median.status_code == 200
    assert unknown.json() == median.json()


def test_payment_plan_defaults_to_disabled(client):
    request = dict(EXAMPLE)
    request.pop('pymnt_plan')
    assert client.post('/predict', json=request).json() == client.post('/predict', json=EXAMPLE).json()


@pytest.mark.parametrize('changes', [
    {'term': 48}, {'grade': 7}, {'emp_length': 11}, {'pymnt_plan': 1},
    {'int_rate': -1}, {'revol_util': 101}, {'fico_range_low': 800, 'fico_range_high': 700},
    {'unexpected_field': 1}, {'open_acc': 1.5},
])
def test_invalid_inputs_return_validation_errors(client, changes):
    assert client.post('/predict', json={**EXAMPLE, **changes}).status_code == 422


def test_missing_required_input_is_rejected(client):
    request = dict(EXAMPLE)
    request.pop('loan_amnt')
    assert client.post('/predict', json=request).status_code == 422
