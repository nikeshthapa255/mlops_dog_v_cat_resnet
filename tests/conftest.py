import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

DOG_PATH = Path(parent, "data/dog_image.webp")
CAT_PATH = Path(parent, "data/cat_image.webp")

@pytest.fixture
def sample_dog_image():
    # Load the dog image for testing
    img_path = DOG_PATH
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    print('TESTING', img_array.shape)
    return img_array

@pytest.fixture
def sample_cat_image():
    # Load the cat image for testing
    img_path = CAT_PATH
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

@pytest.fixture
def sample_dog_image_path():
    return DOG_PATH

@pytest.fixture
def sample_cat_image_path():
    return CAT_PATH

@pytest.fixture
def sample_invalid_input_data():
    # Create a sample invalid input (wrong shape)
    img_array = np.random.rand(150, 150, 3)
    return img_array
