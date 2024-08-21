import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np
from fastapi import UploadFile, File

async def convert_image_to_dataframe(file: UploadFile) -> pd.DataFrame:
    # Open the uploaded image file
    img = Image.open(file.file)
    
    # Resize and convert the image to the format expected by the model
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    return img_array
