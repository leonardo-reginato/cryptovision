import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models                 #type: ignore
from tensorflow.keras import applications as keras_apps     #type: ignore

SEED = 42

# =============================================================================
# Set seeds and deterministic behavior
# =============================================================================
SEED = 42
random.seed(SEED)                           # Python SEED
np.random.seed(SEED)                        # NumPy SEED
tf.random.set_seed(SEED)                    # TensorFlow SEED
os.environ['TF_DETERMINISTIC_OPS'] = '1'    

# Enable mixed precision (useful on e.g. Apple Silicon)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Allow memory growth for GPU devices
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def dense_block(input_layer, name:str, units:int, dropout:float, activation:str='relu', norm:bool=True):
    
    x = layers.Dense(units, name=name)(input_layer)
    if norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout)(x)
    return x

def pretrain_model(backbone, preprocess, input_shape=(224, 224, 3), name=None, augmentation=None):
    
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = backbone(x, training=False)
    
    return tf.keras.Model(inputs, x, name=name)

def basic_multioutput(pretrain, preprocess, output_units: tuple[int, int, int], input_shape=(224, 224, 3), dropout=0.3, name=None, augmentation=None, trainable:bool = False, shared_units:int=None):
    
    pretrain.trainable = trainable
    premodel = pretrain_model(pretrain, preprocess, input_shape=input_shape, name='pretrain', augmentation=augmentation)
    
    # Feature Extracion from Pretrained Model
    features = layers.GlobalMaxPooling2D(name='GlobMaxPool2D')(premodel.output)
    
    # Shared Layer
    shared_layer = dense_block(features, 'shared_layer', shared_units or features.shape[-1], dropout, activation='relu', norm=True)
    
    # Outputs (Family, Genus, Species)
    family_output = layers.Dense(output_units[0], activation='softmax', name='family')(shared_layer)
    genus_output = layers.Dense(output_units[1], activation='softmax', name='genus')(shared_layer)
    species_output = layers.Dense(output_units[2], activation='softmax', name='species')(shared_layer)
    
    model = tf.keras.Model(
        premodel.input,
        [family_output, genus_output, species_output], 
        name=name or 'MultiOutputBasic'
    )
    
    return model

def phorcys_v09 (output_units: tuple[int, int, int], augmentation, input_shape=(224, 224, 3), shared_neurons=512, name=None, genus_neurons=256, species_neurons=128, dropout_rate=0.3):
    
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
    family_output = layers.Dense(output_units[0], activation='softmax', name='family')(shared_layer)

    # Genus Output
    genus_features = layers.Concatenate()([shared_layer, family_output])
    genus_hidden = layers.Dense(genus_neurons, activation='relu')(genus_features)
    genus_output = layers.Dense(output_units[1], activation='softmax', name='genus')(genus_hidden)

    # Species Output
    species_features = layers.Concatenate()([shared_layer, family_output, genus_output])
    species_hidden = layers.Dense(species_neurons, activation='relu')(species_features)
    species_output = layers.Dense(output_units[2], activation='softmax', name='species')(species_hidden)

    model = tf.keras.Model(
        inputs, 
        [family_output, genus_output, species_output],
        name = name or "PhorcysV9"
    )
    
    return model

def hidden_based(pretrain, preprocess, output_units: tuple[int, int, int], hidden_size: tuple[int] = (2048, 1024, 1024), name=None, augmentation=None, input_shape=(224, 224, 3)):
    
    pretrain.trainable = False
    
    inputs = layers.Input(shape=input_shape, name='input')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    features = layers.GlobalAveragePooling2D()(x)
    
    # Family
    family_hidden = dense_block(features, hidden_size[0], 0.3, 'family_hidden')
    family_output = layers.Dense(output_units[0], activation='softmax', name='family')(family_hidden)
    
    # Genus
    genus_hidden = dense_block(family_hidden, hidden_size[1], 0.2, 'genus_hidden', norm=False)
    genus_output = layers.Dense(output_units[1], activation='softmax', name='genus')(genus_hidden)
    
    # Species Output
    species_hidden = dense_block(genus_hidden, hidden_size[2], 0.1, 'species_hidden', norm=False)
    species_output = layers.Dense(output_units[2], activation='softmax', name='species')(species_hidden)
    
    model = tf.keras.Model(
        inputs, 
        [family_output, genus_output, species_output], 
        name=name if name else 'HiddenBasedModel'
    )
    
    return model    

def conv_based(pretrain, preprocess, outputs_size=[10, 20, 30], name=None, augmentation=None, input_shape=(224, 224, 3)):
    
    pretrain.trainable = False
    
    inputs = layers.Input(shape=input_shape, name='input')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Feature Extracion from Pretrained Model
    #features = layers.GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    # Conv2D Block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    
    # Family Output
    family_hidden = layers.Dense(2048, activation='relu')(x)
    family_output = layers.Dense(outputs_size[0], activation='softmax', name='family')(family_hidden)
    
    # Genus Output
    genus_hidden = layers.Dense(1024, activation='relu')(family_hidden)
    genus_output = layers.Dense(outputs_size[1], activation='softmax', name='genus')(genus_hidden)
    
    # Species Output
    species_hidden = layers.Dense(1024, activation='relu')(genus_hidden)
    species_output = layers.Dense(outputs_size[2], activation='softmax', name='species')(species_hidden)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model
    
def dummy():
    pass



if __name__ == "__main__":
    
    backbone = keras_apps.ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    backbone.trainable = False
    model = basic_multioutput(backbone, keras_apps.resnet_v2.preprocess_input, (10, 20, 30), shared_units=512)
    
    model.summary()