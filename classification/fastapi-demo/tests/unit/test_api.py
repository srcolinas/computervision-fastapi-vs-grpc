import cv2
import pytest
import numpy as np
from fastapi.testclient import TestClient

from fastapi_demo.main import app


@pytest.mark.parametrize("format", ["jpg", "png"])
def test_predict(format: str, client: TestClient):
    arr = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    bytes = cv2.imencode(f".{format}", arr)[1].tobytes()
    response = client.post(
        "/predict",
        files=[
            (
                "files",
                ("sample1", bytes, f"image/{format}"),
            ),
            (
                "files",
                ("sample2", bytes, f"image/{format}"),
            ),
        ],
    )
    assert response.status_code == 200
    assert len(response.json()) == 2


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client
