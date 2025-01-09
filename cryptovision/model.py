from tqdm import tqdm
import tensorflow as tf
from loguru import logger
from colorama import Fore, Style
from tensorflow.keras.callbacks import Callback                                                                 # type: ignore
from tensorflow.keras.applications import ResNet50V2, EfficientNetV2B0                                          # type: ignore
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess                       # type: ignore
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess           # type: ignore
from tensorflow.keras.layers import (                                                                           # type: ignore    
    Input, RandomFlip, RandomRotation, RandomZoom, 
    RandomTranslation, RandomContrast, RandomBrightness,
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation,
    Concatenate, Dropout, LeakyReLU, Lambda, Layer, Add, Multiply,
    Reshape, Conv2D, Flatten
) 


class TQDMProgressBar(Callback):
    
    def __init__(self):
        super(TQDMProgressBar, self).__init__()
        self.epoch_bar = None  # Ensure clean initialization
    
    def on_train_begin(self, logs=None):
        # Close any lingering progress bars from previous runs
        if self.epoch_bar:
            self.epoch_bar.close()
        self.epoch_bar = None
    
    def on_epoch_begin(self, epoch, logs=None):
        # Close any existing progress bar to avoid overlaps
        if self.epoch_bar:
            self.epoch_bar.close()
            
        # Create a new progress bar for each epoch
        self.epoch_bar = tqdm(total=self.params['steps'], 
                              desc=f"Epoch {epoch+1}/{self.params['epochs']}", 
                              unit="batch", 
                              dynamic_ncols=True)
    
    def on_batch_end(self, batch, logs=None):
        if self.epoch_bar:
            self.epoch_bar.update(1)
        
        # Reduce updates to avoid clutter
        if batch % 1 == 0 or batch == self.params['steps'] - 1:
            self.epoch_bar.set_postfix({
                'loss': f"{logs.get('loss', 0):.4f}",
                'family_acc': f"{logs.get('family_accuracy', 0):.4f}",
                'genus_acc': f"{logs.get('genus_accuracy', 0):.4f}",
                'species_acc': f"{logs.get('species_accuracy', 0):.4f}",
            })
    
    def colorize_accuracy(self, value):
        if value < 0.75:
            return Fore.RED + f"{value:.4f}" + Style.RESET_ALL
        elif 0.75 <= value < 0.85:
            return Fore.YELLOW + f"{value:.4f}" + Style.RESET_ALL
        elif 0.85 <= value < 0.92:
            return Fore.GREEN + f"{value:.4f}" + Style.RESET_ALL
        else:
            return Fore.MAGENTA + f"{value:.4f}" + Style.RESET_ALL
    
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_bar:
            self.epoch_bar.close()
            self.epoch_bar = None  # Reset for the next epoch
        
        val_family_acc = logs.get('val_family_accuracy', 0)
        val_genus_acc = logs.get('val_genus_accuracy', 0)
        val_species_acc = logs.get('val_species_accuracy', 0)

        val_family_acc_colored = self.colorize_accuracy(val_family_acc)
        val_genus_acc_colored = self.colorize_accuracy(val_genus_acc)
        val_species_acc_colored = self.colorize_accuracy(val_species_acc)

        summary_message = (
            f"Epoch {epoch+1} completed - Loss: {logs.get('loss', 0):.4f}, "
            f"Val Family Accuracy: {val_family_acc_colored}, "
            f"Val Genus Accuracy: {val_genus_acc_colored}, "
            f"Val Species Accuracy: {val_species_acc_colored}"
        )
        logger.info(summary_message)
    
    def on_train_end(self, logs=None):
        if self.epoch_bar:
            self.epoch_bar.close()
            self.epoch_bar = None


class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.query_dense = Dense(input_shape[-1])
        self.key_dense = Dense(input_shape[-1])
        self.value_dense = Dense(input_shape[-1])
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        q = self.query_dense(inputs)
        k = self.key_dense(inputs)
        v = self.value_dense(inputs)
        
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        return tf.matmul(attention_scores, v) + inputs


class ModulatedAttention(Layer):
    def __init__(self, **kwargs):
        super(ModulatedAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_dense = Dense(input_shape[-1])
        super(ModulatedAttention, self).build(input_shape)

    def call(self, inputs, modulating_signal):
        attention_weights = tf.nn.softmax(self.attention_dense(modulating_signal), axis=-1)
        return Multiply()([inputs, attention_weights]) + inputs


def simple (name, labels, pretrain, preprocess, input_shape, augmentation=None):
    
    inputs = Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = Lambda(preprocess, name='preprocess')(x)
    x = pretrain(x, training=False)
    
    # Feature Extracion from Pretrained Model
    features = GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    # Family Output
    family_output = Dense(len(labels['family']), activation='softmax', name='family')(features)
    
    # Genus Output
    genus_output = Dense(len(labels['genus']), activation='softmax', name='genus')(features)
    
    # Species Output
    species_output = Dense(len(labels['species']), activation='softmax', name='species')(features)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def simple_shared (name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    
    inputs = Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = Lambda(preprocess, name='preprocess')(x)
    x = pretrain(x, training=False)
    
    # Feature Extracion from Pretrained Model
    features = GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    shared_layer = Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)
    
    # Family Output
    family_output = Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    genus_output = Dense(len(labels['genus']), activation='softmax', name='genus')(shared_layer)
    
    # Species Output
    species_output = Dense(len(labels['species']), activation='softmax', name='species')(shared_layer)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def simple_shared_concat (name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    
    inputs = Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = Lambda(preprocess, name='preprocess')(x)
    x = pretrain(x, training=False)
    
    # Feature Extracion from Pretrained Model
    features = GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    shared_layer = Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)
    
    # Family Output
    family_output = Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    genus_features = Concatenate()([shared_layer, family_output])
    genus_output = Dense(len(labels['genus']), activation='softmax', name='genus')(genus_features)
    
    # Species Output
    species_features = Concatenate()([shared_layer, family_output, genus_output])
    species_output = Dense(len(labels['species']), activation='softmax', name='species')(species_features)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def phorcys_v09 (labels, input_shape):
    
    pretrain = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    pretrain.trainable = False

    augmentation = tf.keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.2),
            RandomZoom(0.2),
            RandomTranslation(0.1, 0.1),
            RandomContrast(0.2),
            RandomBrightness(0.2),
        ],
        name='data_augmentation'
    )

    inputs = Input(shape=input_shape, name='input')
    x = augmentation(inputs)
    x = resnet_preprocess(x)
    x = pretrain(x, training=False)
    features = GlobalAveragePooling2D()(x)

    shared_layer = Dense(512,name='shared_layer')(features)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)
    shared_layer = Dropout(0.3)(shared_layer)

    # Family Output
    family_output = Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)

    # Genus Output
    genus_features = Concatenate()([shared_layer, family_output])
    genus_hidden = Dense(256, activation='relu')(genus_features)
    genus_output = Dense(len(labels['genus']), activation='softmax', name='genus')(genus_hidden)

    # Species Output
    species_features = Concatenate()([shared_layer, family_output, genus_output])
    species_hidden = Dense(256, activation='relu')(species_features)
    species_output = Dense(len(labels['species']), activation='softmax', name='species')(species_hidden)

    model = tf.keras.Model(
        inputs, 
        [family_output, genus_output, species_output],
        name = "PhorcysV9"
    )
    
    return model

def model_attention (name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    
    inputs = Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = Lambda(preprocess, name='preprocess')(x)
    x = pretrain(x, training=False)
    
    # Attention
    x = SelfAttention()(x)
    
    # Feature Extracion from Pretrained Model
    features = GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    shared_layer = Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)
    
    # Family Output
    family_output = Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    genus_features = Concatenate()([shared_layer, family_output])
    genus_output = Dense(len(labels['genus']), activation='softmax', name='genus')(genus_features)
    
    # Species Output
    species_features = Concatenate()([shared_layer, family_output, genus_output])
    species_output = Dense(len(labels['species']), activation='softmax', name='species')(species_features)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def model_attention_simple (name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    
    inputs = Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = Lambda(preprocess, name='preprocess')(x)
    x = pretrain(x, training=False)
    
    # Attention
    x = SelfAttention()(x)
    
    # Feature Extracion from Pretrained Model
    features = GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    shared_layer = Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)
    
    # Family Output
    family_output = Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    genus_output = Dense(len(labels['genus']), activation='softmax', name='genus')(shared_layer)
    
    # Species Output
    species_output = Dense(len(labels['species']), activation='softmax', name='species')(shared_layer)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def model_residual(name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    
    inputs = Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = Lambda(preprocess, name='preprocess')(x)
    x = pretrain(x, training=False)
    
    # Feature Extraction from Pretrained Model
    features = GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    # Shared Layer (Base Features)
    shared_layer = Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)
    
    # Family Output
    family_output = Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output (with Residual Skip Connection)
    genus_projection = Dense(shared_neurons)(family_output)  # Project family output to same size as shared_layer
    genus_hidden = Add()([shared_layer, genus_projection])  # Skip connection
    genus_hidden = Activation('relu')(genus_hidden)
    genus_output = Dense(len(labels['genus']), activation='softmax', name='genus')(genus_hidden)
    
    # Species Output (with Residual Skip Connection from Family and Genus)
    genus_projection_species = Dense(shared_neurons)(genus_output)
    family_projection_species = Dense(shared_neurons)(family_output)
    
    species_hidden = Add()([shared_layer, genus_projection_species, family_projection_species])  # Skip connections
    species_hidden = Activation('relu')(species_hidden)
    species_output = Dense(len(labels['species']), activation='softmax', name='species')(species_hidden)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def simple_model_with_conv(
    name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, 
    conv_filters=2048, kernel_size=(3, 3), augmentation=None
):
    # Input Layer
    inputs = Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = Lambda(preprocess, name='preprocess')(x)
    
    # Pre-trained Backbone (Feature Extractor)
    x = pretrain(x, training=False)
    
    # Pooling to Reduce Dimensionality
    x = GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    # Shared Fully Connected Layer
    shared_layer = Dense(shared_neurons, name='shared_layer')(x)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)
    
    # Family Output
    family_output = Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output - Conv2D Layer
    genus_features = Reshape((1, 1, shared_neurons))(shared_layer)
    genus_features = Conv2D(conv_filters, kernel_size, activation='relu', padding='same')(genus_features)
    genus_features = Flatten()(genus_features)
    genus_output = Dense(len(labels['genus']), activation='softmax', name='genus')(genus_features)
    
    # Species Output - Conv2D Layer
    species_features = Reshape((1, 1, shared_neurons))(genus_features)
    species_features = Conv2D(conv_filters, kernel_size, activation='relu', padding='same')(species_features)
    species_features = Flatten()(species_features)
    species_output = Dense(len(labels['species']), activation='softmax', name='species')(species_features)
    
    # Compile Model
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def simple_model_with_larger_kernels(
    name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, 
    conv_filters=2048, kernel_size=(5, 5), augmentation=None
):
    # Input Layer
    inputs = Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = Lambda(preprocess, name='preprocess')(x)
    
    # Pre-trained Backbone (Feature Extractor)
    x = pretrain(x, training=False)
    
    # Pooling to Reduce Dimensionality
    x = GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    # Shared Fully Connected Layer
    shared_layer = Dense(shared_neurons, name='shared_layer')(x)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)
    
    # Family Output
    family_output = Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output - Conv2D with Larger Kernel + GAP
    genus_features = Reshape((1, 1, shared_neurons))(shared_layer)
    genus_features = Conv2D(conv_filters, kernel_size, activation='relu', padding='same')(genus_features)
    genus_features = BatchNormalization()(genus_features)
    genus_features = GlobalAveragePooling2D()(genus_features)
    genus_output = Dense(len(labels['genus']), activation='softmax', name='genus')(genus_features)
    
    # Species Output - Conv2D with Larger Kernel + GAP
    species_features = Reshape((1, 1, shared_neurons))(genus_features)
    species_features = Conv2D(conv_filters, kernel_size, activation='relu', padding='same')(species_features)
    species_features = BatchNormalization()(species_features)
    species_features = GlobalAveragePooling2D()(species_features)
    species_output = Dense(len(labels['species']), activation='softmax', name='species')(species_features)
    
    # Compile Model
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def model_conv_new (name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    
    inputs = Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = Lambda(preprocess, name='preprocess')(x)
    x = pretrain(x, training=False)
    
    # Shared Convolutional Refinement
    x = Conv2D(2048, (5,5), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Feature Extracion from Pretrained Model
    features = GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    shared_layer = Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)
    
    # Family Output
    family_output = Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    genus_features = Concatenate()([shared_layer, family_output])
    genus_output = Dense(len(labels['genus']), activation='softmax', name='genus')(genus_features)
    
    # Species Output
    species_features = Concatenate()([shared_layer, family_output, genus_output])
    species_output = Dense(len(labels['species']), activation='softmax', name='species')(species_features)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

