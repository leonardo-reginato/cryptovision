import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models as keras_models     # type: ignore
from tensorflow.keras import applications as keras_apps         # type: ignore

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class CryptoVisionModels:
    @staticmethod
    def backbone_and_preprocess(pretrain: str, image_shape=(224, 224, 3)):
        model_map = {
            "rn50v2": (keras_apps.ResNet50V2, keras_apps.resnet_v2.preprocess_input),
            "rn152v2": (keras_apps.ResNet152V2, keras_apps.resnet_v2.preprocess_input),
            "efv2b0": (keras_apps.EfficientNetV2B0, keras_apps.efficientnet_v2.preprocess_input),
            "efv2b1": (keras_apps.EfficientNetV2B1, keras_apps.efficientnet_v2.preprocess_input),
            "vgg16": (keras_apps.VGG16, keras_apps.vgg16.preprocess_input),
        }
        model = model_map[pretrain][0](include_top=False, weights='imagenet', input_shape=image_shape)
        preprocess = model_map[pretrain][1]
        return model, preprocess

    @staticmethod
    def build_feature_extractor(backbone, preprocess, input_shape, name=None, augmentation=None):
        inputs = layers.Input(shape=input_shape, name='input_layer')
        x = augmentation(inputs) if augmentation else inputs
        x = preprocess(x)
        x = backbone(x, training=False)
        return keras_models.Model(inputs, x, name=name)

    @staticmethod
    def dense_block(input_layer, name, units, dropout=0.3, activation='relu', norm=True, **kwargs):
        """
        Create a flexible dense block.

        Parameters:
        - input_layer: input tensor
        - name: name of the dense layer
        - units: number of units in dense layer
        - dropout: dropout rate (0.0 disables dropout)
        - activation: activation function
        - norm: apply BatchNormalization if True
        - **kwargs: passed to tf.keras.layers.Dense (e.g., kernel_regularizer, initializer, etc.)

        Returns:
        - output tensor
        """
        x = layers.Dense(units, name=name, **kwargs)(input_layer)
        if norm:
            x = layers.BatchNormalization(name=f"{name}_BatchNorm")(x)
        x = layers.Activation(activation, name=f"{name}_Activ_{activation}")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"{name}_DropOut_{dropout}")(x)
        return x

    @staticmethod
    def basic(imagenet_name:str, output_neurons:tuple, input_shape:tuple[int]=(224, 224, 224), shared_dropout:float=0.3, feat_dropout:float=0.3, pooling_type:str='max', concatenate:bool=False, name:str=None, augmentation=None, trainable:bool=False, shared_layer_neurons=None):
        
        # Load pretrained backbone and preprocessing function
        imagenet_model, preprocess = CryptoVisionModels.backbone_and_preprocess(imagenet_name, input_shape)
        imagenet_model.trainable = trainable
        
        # Create backbone feature extractor model
        feature_extractor = CryptoVisionModels.build_feature_extractor(
            imagenet_model, preprocess, input_shape, name='pretrain', augmentation=augmentation
        )
        
        # Global pooling from feature extractor output
        if pooling_type == 'max':
            features = layers.GlobalMaxPool2D(name='GlobMaxPool2D')(feature_extractor.output)
        elif pooling_type == 'avg':
            features = layers.GlobalAveragePooling2D(name='GlobAvgPool2D')(feature_extractor.output)
        else:
            raise ValueError(f"Invalid pooling type: {pooling_type}. Choose 'max' or 'avg'.")
        
        # Features dropout
        if feat_dropout and feat_dropout > 0:
            features = layers.BatchNormalization(name='features_BatchNorm')(features)
            features = layers.Dropout(feat_dropout, name=f'features_DropOut_{feat_dropout}')(features)
        
        # Shared layers
        shared_layer = CryptoVisionModels.dense_block(
            features, 'shared_layer', shared_layer_neurons or features.shape[-1], shared_dropout
        )
        
        # Family
        family_output = layers.Dense(output_neurons[0], activation='softmax', name='family')(shared_layer)
        
        #  Genus
        genus_input = layers.Concatenate(name='genus_input')([shared_layer, family_output]) if concatenate else genus_input = shared_layer
        genus_output = layers.Dense(output_neurons[1], activation='softmax', name='genus')(genus_input)
        
        # Species
        species_input = layers.Concatenate(name='species_input')([shared_layer, genus_output]) if concatenate else species_input = shared_layer
        species_output = layers.Dense(output_neurons[2], activation='softmax', name='species')(species_input)
        
        return keras_models.Model(
            feature_extractor.input, 
            [family_output, genus_output, species_output], 
            name=name or 'CVisionBasic'
        )

