from typing import Union, Tuple, Optional
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from pathlib import Path
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


# Updated validate_inputs function
def validate_inputs(input_data: Union[np.ndarray, str]) -> Tuple[Union[np.ndarray, np.ndarray], Optional[str]]:
    """Validate input data for either DataFrame or image path."""

    # If input is a numpy array
    if isinstance(input_data, np.ndarray):
        if input_data.shape == (1, 150, 150, 3):
            return input_data, None  # The shape is valid
        else:
            return None, f"Invalid input array shape. Expected (1, 150, 150, 3), but got {input_data.shape}"
    
    
    # If input is a string, assume it's an image path
    elif isinstance(input_data, (str, Path)):
        img, error = validate_image_input(input_data)
        if error:
            return None, error
        return img, None
    
    # If input is neither, return an error
    return None, "Invalid input type. Expected a numpy array or image path string."

