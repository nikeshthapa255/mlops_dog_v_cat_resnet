import warnings
warnings.filterwarnings("ignore", message=".*input_shape.*")

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pytest
from dog_vs_cat.predict import make_prediction

def test_make_prediction_with_dog_image(sample_dog_image):
    # Given
    expected_class = 'Dog'  # Expected prediction for the dog image

    # When
    result = make_prediction(input_data=sample_dog_image)

    # Then
    predictions = result.get("predictions")
    assert predictions == expected_class
    assert result.get("errors") is None

def test_make_prediction_with_cat_image(sample_cat_image):
    # Given
    expected_class = 'Cat'  # Expected prediction for the cat image

    # When
    result = make_prediction(input_data=sample_cat_image)

    # Then
    predictions = result.get("predictions")
    assert predictions == expected_class
    assert result.get("errors") is None

def test_make_prediction_with_dog_image_path(sample_dog_image_path):
    # Given

    # When
    result = make_prediction(input_data=sample_dog_image_path)

    # Then
    predictions = result.get("predictions")
    assert predictions == 'Dog'  # Ensure it predicts Dog
    assert result.get("errors") is None

def test_make_prediction_with_cat_image_path(sample_cat_image_path):
    # Given

    # When
    result = make_prediction(input_data=sample_cat_image_path)

    # Then
    predictions = result.get("predictions")
    assert predictions == 'Cat'  # Ensure it predicts Cat
    assert result.get("errors") is None

def test_make_prediction_with_invalid_input(sample_invalid_input_data):
    # Given

    # When
    result = make_prediction(input_data=sample_invalid_input_data)

    # Then
    assert result.get("predictions") is None
    assert "Invalid input array shape" in result.get("errors")

def test_make_prediction_with_nonexistent_image():
    # Given
    nonexistent_image_path = "data/nonexistent_image.webp"

    # When
    result = make_prediction(input_data=nonexistent_image_path)

    # Then
    assert result.get("predictions") is None
    assert "No such file or directory" in result.get("errors")
