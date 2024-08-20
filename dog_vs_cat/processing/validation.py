from typing import Union, Tuple, Optional
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from pydantic import ValidationError

from dog_vs_cat.config.core import config

# Function to load and preprocess image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to validate image input
def validate_image_input(img_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Validate if the image path leads to a valid image file."""
    try:
        img = load_and_preprocess_image(img_path)
        return img, None
    except Exception as e:
        return None, str(e)

def validate_dataframe_input(df: pd.DataFrame) -> Optional[str]:
    """Validate if the DataFrame contains image data in the right shape."""
    expected_shape_single = (150, 150, 3)
    expected_shape_multiple = (-1, 150, 150, 3)  # -1 represents any number of images

    if df.shape == expected_shape_single or len(df.shape) == 4 and df.shape[1:] == expected_shape_single:
        return None  # No error, the shape is valid
    else:
        return f"Invalid DataFrame shape. Expected {expected_shape_single} for single image or {expected_shape_multiple} for multiple images, but got {df.shape}."


# Updated validate_inputs function
def validate_inputs(input_data: Union[pd.DataFrame, str]) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[str]]:
    """Validate input data for either DataFrame or image path."""
    
    # If input is a DataFrame
    if isinstance(input_data, pd.DataFrame):
        error = validate_dataframe_input(input_data)
        if error:
            return None, error
        return input_data, None  # Return the DataFrame as is if it passes validation
    
    
    # If input is a string, assume it's an image path
    elif isinstance(input_data, str):
        img, error = validate_image_input(input_data)
        if error:
            return None, error
        return img, None
    
    # If input is neither, return an error
    else:
        return None, "Invalid input type. Expected a DataFrame or image path string."

