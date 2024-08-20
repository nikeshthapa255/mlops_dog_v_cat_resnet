import tensorflow as tf
from tensorflow import keras

class ClassificationModel(keras.Model):
    """A model for classifying images as either dog or cat using MobileNetV2 as the base."""

    def __init__(self, input_shape=(150, 150, 3), num_classes=2, **kwargs):
        """Creates an instance of the image classification model."""
        super(ClassificationModel, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Load the base MobileNetV2 model pre-trained on ImageNet
        base_model = keras.applications.MobileNetV2(
            weights="imagenet",
            input_shape=input_shape,
            include_top=False,
        )
        
        # Freeze the base model layers
        base_model.trainable = False

        self.base_model = base_model
        self.avgpool = keras.layers.GlobalAveragePooling2D()
        self.dense_1 = keras.layers.Dense(512, activation='relu')
        self.dropout_1 = keras.layers.Dropout(0.5)
        self.dense_2 = keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        """
        Forward pass in the neural network.

        Args:
            inputs: Input tensor.
            training: Boolean indicating if the call is during training.

        Returns:
            Output tensor with predictions.
        """
        x = self.base_model(inputs, training=training)
        x = self.avgpool(x)
        x = self.dense_1(x)
        x = self.dropout_1(x, training=training)
        return self.dense_2(x)

    def get_config(self):
        """Return the configuration of the model."""
        config = super(ClassificationModel, self).get_config()
        config.update({
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create an instance of the model from the configuration dictionary."""
        return cls(**config)
