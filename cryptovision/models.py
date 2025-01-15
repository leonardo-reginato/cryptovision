import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Layer, layers, backend                 # type: ignore
from tensorflow.keras import applications as keras_apps             # type: ignore
from tensorflow.keras.saving import register_keras_serializable     # type: ignore 

SEED = 42

# Python random seed
random.seed(SEED)

# NumPy random seed
np.random.seed(SEED)

# TensorFlow random seed
tf.random.set_seed(SEED)

# Set environment variable for deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Enable mixed precision for Apple Silicon
tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def basic_multioutput(
    pretrain, 
    preprocess, 
    input_shape=(224, 224, 3), 
    outputs_size=[10, 20, 30], 
    dropout_rate=0.3, 
    name=None, 
    augmentation=None,
    pretrain_trainable = False,
):
    
    pretrain.trainable = pretrain_trainable
    
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Feature Extracion from Pretrained Model
    features = layers.GlobalMaxPooling2D(name='GlobMaxPool2D')(x)
    
    # Shared Layer
    shared_layer = layers.Dense(features.shape[-1], name='shared_layer')(features)
    shared_layer = layers.BatchNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    shared_layer = layers.Dropout(dropout_rate)(shared_layer)
    
    # Family Output
    family_output = layers.Dense(outputs_size[0], activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    genus_output = layers.Dense(outputs_size[1], activation='softmax', name='genus')(shared_layer)
    
    # Species Output
    species_output = layers.Dense(outputs_size[2], activation='softmax', name='species')(shared_layer)
    
    model = tf.keras.Model(
        inputs, 
        [family_output, genus_output, species_output], 
        name=name if name else 'MultiOutputBasic'
    )
    
    return model

def phorcys_v09 (
    labels, augmentation, input_shape=(224, 224, 3), shared_neurons=512, name=None,
    genus_neurons=256, species_neurons=128, dropout_rate=0.3
):
    
    pretrain = keras_apps.ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    pretrain.trainable = False

    inputs = layers.Input(shape=input_shape, name='input')
    x = augmentation(inputs)
    x = keras_apps.resnet_v2.preprocess_input(x)
    x = pretrain(x, training=False)
    features = layers.GlobalAveragePooling2D()(x)

    shared_layer = layers.Dense(shared_neurons,name='shared_layer')(features)
    shared_layer = layers.BatchNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    shared_layer = layers.Dropout(dropout_rate)(shared_layer)

    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)

    # Genus Output
    genus_features = layers.Concatenate()([shared_layer, family_output])
    genus_hidden = layers.Dense(genus_neurons, activation='relu')(genus_features)
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_hidden)

    # Species Output
    species_features = layers.Concatenate()([shared_layer, family_output, genus_output])
    species_hidden = layers.Dense(species_neurons, activation='relu')(species_features)
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(species_hidden)

    model = tf.keras.Model(
        inputs, 
        [family_output, genus_output, species_output],
        name = "PhorcysV9" if name is None else name
    )
    
    return model

def hidden_based(pretrain, preprocess, outputs_size=[10, 20, 30], name=None, augmentation=None, input_shape=(224, 224, 3)):
    
    def dense_block(input_layer, units, dropout, name, norm=True):
        x = layers.Dense(units, name=name)(input_layer)
        if norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout)(x)
        return x
    
    
    inputs = layers.Input(shape=input_shape, name='input')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    features = layers.GlobalAveragePooling2D()(x)
    
    # Family
    family_hidden = dense_block(features, 2048, 0.3, 'family_hidden')
    family_output = layers.Dense(outputs_size[0], activation='softmax', name='family')(family_hidden)
    
    # Genus
    genus_hidden = dense_block(family_hidden, 1024, 0.2, 'genus_hidden', norm=False)
    genus_output = layers.Dense(outputs_size[1], activation='softmax', name='genus')(genus_hidden)
    
    # Species Output
    species_hidden = dense_block(genus_hidden, 1024, 0.1, 'species_hidden', norm=False)
    species_output = layers.Dense(outputs_size[2], activation='softmax', name='species')(species_hidden)
    
    model = tf.keras.Model(
        inputs, 
        [family_output, genus_output, species_output], 
        name=name if name else 'HiddenBasedModel'
    )
    
    return model    
    
def dummy():
    pass

