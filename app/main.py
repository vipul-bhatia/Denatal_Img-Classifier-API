from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.model.model import predict, model
from app.model.model import __version__ as model_version
from app.util import create_data_batches, get_pred_label  
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to be more restrictive in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Image(BaseModel):
    image: str
class Prediction(BaseModel):
    prediction: str
    probability: str

@app.get("/")
def home () :
    return {"health_check": "OK", "model_version": "model_version", 'priyam':'Good'}
@app.post("/predict", response_model=Prediction)
async def predict(image: UploadFile = File(...)):
    image_content = await image.read()

    # Use your custom create_data_batches function
    processed_image = create_data_batches([image_content], test_data=True)

    try:
        # Get prediction probabilities from model
        prediction = model.predict(processed_image)
    except Exception as e:
        return {"prediction": "", "probability": "error"}

    prediction_probability = np.max(prediction) * 100

    # Use get_pred_label function to convert prediction to a label
    prediction_label = get_pred_label(prediction)

    # Check if prediction probability is less than 80%
    if prediction_probability < 80:
        return {
            "prediction": "Chances of no disease:)",
            "probability": "we recommend you to visit us for a detailed checkup."
        }

    prediction_probability_str = "{:.2f}%".format(prediction_probability)
    return {"prediction": prediction_label, "probability": prediction_probability_str}
