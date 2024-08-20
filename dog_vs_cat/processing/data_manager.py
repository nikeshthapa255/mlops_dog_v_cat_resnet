import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import zipfile


from dog_vs_cat import __version__ as _version
from dog_vs_cat.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


DataGenerators = t.Tuple[DirectoryIterator, DirectoryIterator]

def extract_zip(path_to_zip: str, extract_to: str):
    # Extract the zip file
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Extracted files to {extract_to}")

    return Path(extract_to)

##  Pre-Pipeline Preparation
def download_and_prepare_data() -> tuple[Path, Path]:
    # Download and extract dataset if not already present
    _URL = config.app_config.DATASET_PATH
    ZIP_FILE = 'cats_and_dogs.zip'
    FOLDER_NAME = 'cats_and_dogs_filtered'
    ZIP_FILE_PATH = Path(f"{DATASET_DIR}/{ZIP_FILE}") 

    path_to_zip = tf.keras.utils.get_file( ZIP_FILE_PATH, origin=_URL, extract=True)
    FOLDER_PATH =  Path(f"{ Path(path_to_zip).parent}/{FOLDER_NAME}")
    
    extract_zip(ZIP_FILE_PATH, DATASET_DIR)

    train_dir = FOLDER_PATH / 'train'
    validation_dir = FOLDER_PATH / 'validation'
    return train_dir, validation_dir

def create_data_generators(
    train_dir: Path, 
    validation_dir: Path
) -> DataGenerators:
    train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=config.base_config.batch_size,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=config.base_config.batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator
  



# def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
#     dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
#     return dataframe

def load_dataset() -> DataGenerators:
    ## TODO: load it from DVC or local data
    train_dir, validation_dir = download_and_prepare_data()
    return create_data_generators(train_dir, validation_dir)


def save_pipeline(*, model: t.Any) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.keras"
    save_path = TRAINED_MODEL_DIR / save_file_name
    model.save(save_path)

    # remove_old_pipelines(files_to_keep=[save_file_name])
    # joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    loaded_model = tf.keras.models.load_model(file_path)

    return loaded_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
