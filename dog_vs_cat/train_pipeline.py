import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
import tensorflow as tf
from dog_vs_cat.config.core import config
from dog_vs_cat.processing import load_dataset, save_pipeline
from dog_vs_cat.model import ClassificationModel

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    train_generator, validation_generator = load_dataset()

    # load model
    model = ClassificationModel(input_shape=(150, 150, 3), num_classes=2)
    
     # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        epochs=config.base_config.epochs,
                        verbose=2)

    # Save the model

    # persist trained model
    save_pipeline(model = model)
    # printing the score
    
if __name__ == "__main__":
    run_training()