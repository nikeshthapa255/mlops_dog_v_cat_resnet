from fastapi import APIRouter, HTTPException, UploadFile, File
from PIL import Image
import numpy as np
import json

from dog_vs_cat import __version__ as model_version
from dog_vs_cat.predict import make_prediction
from app import __version__, schemas
from app.config import settings
from app.utils import convert_image_to_dataframe

api_router = APIRouter()

@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(file: UploadFile = File(...)):
    """
    Dog vs Cat prediction with the AI model.
    """

    try:
        # Await the conversion function since it's asynchronous
        img_array = await convert_image_to_dataframe(file)
        
        # Make prediction
        results = make_prediction(input_data=img_array)

        if results["errors"] is not None:
            raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {str(e)}")
