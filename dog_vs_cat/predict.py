import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image

from dog_vs_cat import __version__ as _version
from dog_vs_cat.config.core import config
from dog_vs_cat.processing.data_manager import load_pipeline
from dog_vs_cat.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.keras"
loaded_model = load_pipeline(file_name=pipeline_file_name)


def make_prediction(input_data: Union[np.ndarray, str]) -> dict:
    """Make a prediction using a saved model with image or numpy array input."""

    validated_data, error = validate_inputs(input_data=input_data)

    results = {"predictions": None, "version": _version, "errors": error}
    
    if error:
        print(f"Error in input validation: {error}")
        return results
    
    
    predictions = loaded_model.predict(validated_data)
    predicted_class = 'Dog' if np.argmax(predictions) == 1 else 'Cat'
    
    results["predictions"] = predicted_class
    return results

if __name__ == "__main__":
    # Check if an image path is passed as an argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        prediction = make_prediction(input_data=image_path)
        print(f"Prediction for image at {image_path}: {prediction}")
    else:
        pass
        # # Default data input for testing when no image path is provided
        # data_in = 
        # prediction = make_prediction(input_data=pd.DataFrame(data_in))
        # print(f"Prediction for data input: {prediction}")
        