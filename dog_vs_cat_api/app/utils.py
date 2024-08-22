import pandas as pd
from PIL import Image
import numpy as np
from fastapi import UploadFile, File



async def convert_image_to_dataframe(file: UploadFile) -> np.ndarray:
    # Convert the uploaded file to an image
    img = Image.open(file.file)
    img = img.resize((150, 150))
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
