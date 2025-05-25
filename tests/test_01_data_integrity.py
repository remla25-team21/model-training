import os
import pytest
import joblib
import pandas as pd

RAW_DATA_PATH = "data/raw/a1_RestaurantReviews_HistoricDump.tsv"

@pytest.fixture(scope="module")
def raw_data():
    assert os.path.exists(RAW_DATA_PATH), f"Data file not found at {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH, sep='\t')
    df.columns = df.columns.str.strip()
    return df

def test_column_schema(raw_data):
    """Check that expected columns exist"""
    expected = {'Review', 'Liked'}
    actual = set(raw_data.columns)
    missing = expected - actual
    assert not missing, f"Missing expected columns: {missing}"

def test_no_missing_values(raw_data):
    """Ensure no nulls in important columns"""
    for col in ['Review', 'Liked']:
        assert raw_data[col].isnull().sum() == 0, f"Missing values found in {col}"

def test_liked_label_values(raw_data):
    """Ensure 'Liked' is binary (0 or 1)"""
    assert raw_data['Liked'].isin([0, 1]).all(), "'Liked' column contains non-binary values"

def test_review_length(raw_data):
    """Check that Review has sufficient length"""
    assert raw_data['Review'].str.len().gt(10).all(), "Some reviews are too short"

def test_exact_duplicate_rows(raw_data):
    """Check for fully duplicated rows with same Review and Liked"""
    duplicates = raw_data.duplicated().sum()
    assert duplicates <= 10, f"Unusual number of exact duplicate rows: {duplicates}"
