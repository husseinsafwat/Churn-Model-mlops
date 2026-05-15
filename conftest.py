import pytest
from unittest.mock import patch, MagicMock, mock_open
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(autouse=True)
def mock_pickle():
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = [[0.1] * 10]
    mock_preprocessor.get_feature_names_out.return_value = [f"f{i}" for i in range(10)]

    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.8, 0.2]]

    with patch("builtins.open", mock_open()), \
         patch("pickle.load", side_effect=[mock_preprocessor, mock_model]):
        yield mock_preprocessor, mock_model


@pytest.fixture
def client(mock_pickle):
    mock_preprocessor, mock_model = mock_pickle
    with TestClient(app) as c:
        c.app.state.preprocessor = mock_preprocessor
        c.app.state.model = mock_model
        yield c