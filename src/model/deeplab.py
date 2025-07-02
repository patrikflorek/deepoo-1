import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from ..config import IMG_SIZE, NUM_CLASSES

def DeepLabV3Plus(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze backbone for feature extraction
    layer_names = [
        'block_1_expand_relu',   # low-level features
        'out_relu',              # high-level features
    ]
    low_level_feat = base_model.get_layer(layer_names[0]).output
    x = base_model.get_layer(layer_names[1]).output

    # ASPP (simplified)
    b0 = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.Activation('relu')(b0)

    def resize_to(tensor, target):
        target_shape = tf.shape(target)[1:3]
        return tf.image.resize(tensor, target_shape, method='bilinear')

    x = layers.Lambda(lambda tensors: resize_to(tensors[0], tensors[1]))([b0, low_level_feat])
    low_level_feat = layers.Conv2D(48, 1, padding='same', use_bias=False)(low_level_feat)
    low_level_feat = layers.BatchNormalization()(low_level_feat)
    low_level_feat = layers.Activation('relu')(low_level_feat)
    x = layers.Concatenate()([x, low_level_feat])

    # Decoder
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Lambda(lambda t: tf.image.resize(t, IMG_SIZE, method='bilinear'))(x)

    outputs = layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model
