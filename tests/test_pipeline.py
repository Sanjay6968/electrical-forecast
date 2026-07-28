import pytest
import pandas as pd
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.app import app
from src.preprocess import clean_data
from src.feature_engineering import feature_engineering
client = TestClient(app)

def test_api_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'

def test_model_version():
    response = client.get('/model/version')
    assert response.status_code == 200
    assert 'version' in response.json()

def test_metrics_endpoint():
    response = client.get('/metrics')
    assert response.status_code == 200
    assert 'total_predictions' in response.json()

def test_clean_data_logic():
    df = pd.DataFrame({'Date': ['16/12/2006', '16/12/2006'], 'Time': ['17:24:00', '18:24:00'], 'Global_active_power': [4.216, 5.36]})
    cleaned = clean_data(df)
    assert not cleaned.empty
    assert 'hourly_energy_demand' in cleaned.columns
    assert len(cleaned) == 2

def test_feature_engineering_logic():
    dates = pd.date_range(start='2006-12-16', periods=200, freq='1h')
    df = pd.DataFrame({'hourly_energy_demand': [x for x in range(200)]}, index=dates)
    features = feature_engineering(df)
    assert not features.empty
    assert 'lag_24' in features.columns
    assert 'target_h24' in features.columns