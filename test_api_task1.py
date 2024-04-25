import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, UploadFile, File
from task1 import app

@pytest.fixture
def client():
    return TestClient(app)

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {}


def test_predict_endpoint(client):
    image_path = "/Users/nikhilanand/FastAPI_BDL_Assignment/resized5.jpg"

    with open(image_path,'rb') as file:
        file_content=file.read()
    files = {"file": ("file.jpg", file_content)}

    resp = client.post("/predict",files=files)
    assert resp.status_code == 200
    assert resp.json() == {"digit": "5"}