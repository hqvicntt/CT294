import os
import pickle
import pytest
from app import app
from train_core import load_and_prepare_data, models


def test_home_route():
    test_client = app.test_client()
    response = test_client.get('/')
    assert response.status_code == 200
    assert "Ứng dụng Dự đoán Nấm" in response.data.decode('utf-8')


def test_training_produces_model():
    if os.path.exists("best_model.pkl"):
        os.remove("best_model.pkl")

    df, encoders, replace_map = load_and_prepare_data()
    X = df.drop('class', axis=1)
    y = df['class']
    model = list(models.values())[0]
    model.fit(X, y)

    with open("best_model.pkl", "wb") as f:
        pickle.dump((model, encoders, replace_map), f)

    assert os.path.exists("best_model.pkl")


def test_model_can_predict():
    assert os.path.exists("best_model.pkl")

    with open("best_model.pkl", "rb") as f:
        model, label_encoders, replace_map = pickle.load(f)

    df, _, _ = load_and_prepare_data()
    X = df.drop('class', axis=1)

    prediction = model.predict(X[:1])
    assert prediction.shape[0] == 1


@pytest.mark.parametrize("col", ["cap-shape", "odor", "gill-size"])
def test_label_encoder_columns_exist(col):
    _, label_encoders, _ = load_and_prepare_data()
    assert col in label_encoders
