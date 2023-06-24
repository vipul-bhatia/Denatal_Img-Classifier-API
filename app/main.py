from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict
from app.model.model import __version__ as model_version

app = FastAPI()

class Image(BaseModel):
    image: str
class Prediction(BaseModel):
    prediction: str

@app.get("/")
def home () :
    return {"health_check": "OK", "model_version": "model_version"}
@app.post("/predict", response_model=Prediction)
def predict(image: Image):
    prediction = predict(image.image)
    return {"prediction": prediction}