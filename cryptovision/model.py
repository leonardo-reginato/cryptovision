from tqdm import tqdm
import tensorflow as tf
from loguru import logger
from colorama import Fore, Style   
from tensorflow.keras import Layer                                                                              # type: ignore
from tensorflow.keras.callbacks import Callback                                                                 # type: ignore
from tensorflow.keras.applications import ResNet50V2, EfficientNetV2B0, EfficientNetV2B3, ResNet101V2, EfficientNetB3        # type: ignore
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess                       # type: ignore
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess           # type: ignore
from tensorflow.keras import layers                                                                             # type: ignore
from tensorflow.keras import backend as K                                                                       # type: ignore
from tensorflow.keras.saving import register_keras_serializable                                                 # type: ignore


# Enable mixed precision for Apple Silicon
tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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


@register_keras_serializable()
class SoftAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SoftAttention, self).__init__(**kwargs)
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense_relu = layers.Dense(units=None, activation='relu')  # `units` will be set in `build`
        self.dense_sigmoid = layers.Dense(units=None, activation='sigmoid')  # `units` will be set in `build`
        self.multiply = layers.Multiply()
        self.reshape_layer = None  # Initialize to None

    def build(self, input_shape):
        # Define layer parameters that depend on the input shape
        channel = input_shape[-1]
        self.dense_relu.units = channel
        self.dense_sigmoid.units = channel
        self.reshape_layer = layers.Reshape(target_shape=(1, 1, channel))  # Define Reshape here
        super(SoftAttention, self).build(input_shape)

    def call(self, inputs):
        attention = self.global_avg_pool(inputs)
        attention = self.dense_relu(attention)
        attention = self.dense_sigmoid(attention)
        attention = self.reshape_layer(attention)  # Use the defined Reshape layer
        return self.multiply([inputs, attention])

@register_keras_serializable()
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def call(self, inputs):
        channel = inputs.shape[-1]
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        max_pool = layers.GlobalMaxPooling2D()(inputs)

        avg_dense = layers.Dense(channel // self.reduction_ratio, activation='relu')(avg_pool)
        avg_dense = layers.Dense(channel, activation='sigmoid')(avg_dense)

        max_dense = layers.Dense(channel // self.reduction_ratio, activation='relu')(max_pool)
        max_dense = layers.Dense(channel, activation='sigmoid')(max_dense)

        attention = layers.Add()([avg_dense, max_dense])
        attention = layers.Reshape((1, 1, channel))(attention)
        return layers.Multiply()([inputs, attention])

@register_keras_serializable()
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        # Define static layers
        self.avg_pool = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))
        self.max_pool = layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True))
        self.concat = layers.Concatenate(axis=-1)
        self.conv = layers.Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid')
        self.multiply = layers.Multiply()

    def call(self, inputs):
        # Apply average and max pooling
        avg_pooled = self.avg_pool(inputs)
        max_pooled = self.max_pool(inputs)
        
        # Concatenate pooled features
        concat = self.concat([avg_pooled, max_pooled])
        
        # Apply convolution to generate attention map
        attention = self.conv(concat)
        
        # Multiply input by attention map
        return self.multiply([inputs, attention])

@register_keras_serializable()
class SelfAttention(tf.keras.layers.Layer):
    def call(self, inputs):
        shape = tf.shape(inputs)
        batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]

        flat = tf.reshape(inputs, (batch_size, height * width, channels))
        query = layers.Dense(channels // 8)(flat)
        key = layers.Dense(channels // 8)(flat)
        value = layers.Dense(channels)(flat)

        attention_weights = tf.matmul(query, key, transpose_b=True) / tf.sqrt(float(channels))
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)

        weighted_values = tf.matmul(attention_weights, value)
        weighted_values = tf.reshape(weighted_values, (batch_size, height, width, channels))
        return layers.Add()([inputs, weighted_values])

@register_keras_serializable()
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads=8, key_dim=64, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

        # Define static layers
        self.query_dense = layers.Dense(num_heads * key_dim)
        self.key_dense = layers.Dense(num_heads * key_dim)
        self.value_dense = layers.Dense(num_heads * key_dim)
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.combine_heads = layers.Dense(key_dim * num_heads)

    def _split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _combine_heads(self, x):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        batch_size = tf.shape(x)[0]
        return tf.reshape(x, (batch_size, -1, self.num_heads * self.key_dim))

    def call(self, inputs):
        # Ensure inputs are float32 for calculations
        inputs = tf.cast(inputs, dtype=tf.float32)

        # Generate query, key, and value tensors
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Split into multiple heads
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # Compute attention scores
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores /= tf.sqrt(float(self.key_dim))

        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)

        # Compute attention output
        attention_output = tf.matmul(attention_weights, value)
        attention_output = self._combine_heads(attention_output)

        # Combine heads
        return self.combine_heads(attention_output)

def basic_model (name, labels, pretrain, preprocess, input_shape, augmentation=None):
    
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Feature Extracion from Pretrained Model
    features = layers.GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(features)
    
    # Genus Output
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(features)
    
    # Species Output
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(features)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def shared_feature_model (name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Feature Extracion from Pretrained Model
    #features = layers.GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    #features = layers.GlobalMaxPooling2D()(x)
    
    # Feature Extracion from Pretrained Model
    features = layers.Flatten()(x)
    
    shared_layer = layers.Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = layers.BatchNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    shared_layer = layers.Dropout(0.5)(shared_layer)
    
    #shared_layer = layers.Dense(shared_neurons // 2, activation=None, name='shared_dense2')(shared_layer)
    #shared_layer = layers.BatchNormalization()(shared_layer)
    #shared_layer = layers.Activation('relu')(shared_layer)
    #shared_layer = layers.Dropout(0.3)(shared_layer)
    
    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(shared_layer)
    
    # Species Output
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(shared_layer)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def shrdhid_model (name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, hidden_neurons=[64, 128, 256], augmentation=None):
    
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Feature Extracion from Pretrained Model
    features = layers.GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    shared_layer = layers.Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = layers.BatchNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    
    # Family Output
    family_hidden = layers.Dense(hidden_neurons[0], name='family_hidden')(shared_layer)
    family_hidden = layers.BatchNormalization()(family_hidden)
    family_hidden = layers.Activation('relu')(family_hidden)
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(family_hidden)
    
    # Genus Output
    genus_hidden = layers.Dense(hidden_neurons[1], name='genus_hidden')(shared_layer)
    genus_hidden = layers.BatchNormalization()(genus_hidden)
    genus_hidden = layers.Activation('relu')(genus_hidden)
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_hidden)
    
    # Species Output
    species_hidden = layers.Dense(hidden_neurons[2], name='species_hidden')(shared_layer)
    species_hidden = layers.BatchNormalization()(species_hidden)
    species_hidden = layers.Activation('relu')(species_hidden)
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(species_hidden)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def shared_concat_model (name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Feature Extracion from Pretrained Model
    #features = layers.GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    features = layers.Flatten()(x)
    
    shared_layer = layers.Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = layers.BatchNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    shared_layer = layers.Dropout(0.3)(shared_layer)
    
    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    family_to_genus = layers.Dense(len(labels['genus']), activation='relu', name='family_to_genus_mapping')(family_output)
    genus_features = layers.Concatenate()([shared_layer, family_to_genus])
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_features)
    
    # Species Output
    genus_to_species = layers.Dense(len(labels['species']), activation='relu', name='genus_to_species_mapping')(genus_output)
    species_features = layers.Concatenate()([shared_layer, genus_to_species])
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(species_features)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def shared_concat_modelv2 (name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Feature Extracion from Pretrained Model
    features = layers.GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    #residual_features = layers.GlobalMaxPooling2D(name='GlobMaxPool2D')(x)
    #features = layers.Concatenate(name='concat_features')([features, residual_features])
    
    shared_layer = layers.Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = layers.LayerNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    shared_layer = layers.Dropout(0.3)(shared_layer)
    
    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    family_mask = layers.Dense(shared_neurons, activation='relu', name='family_to_genus_mapping')(family_output)
    genus_features = layers.Multiply()([shared_layer, family_mask])
    genus_features = layers.LayerNormalization()(genus_features)
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_features)
    
    # Species Output
    genus_mask = layers.Dense(shared_neurons, activation='relu', name='genus_to_species_mapping')(genus_output)
    species_features = layers.Multiply()([shared_layer, genus_mask])
    species_features = layers.LayerNormalization()(species_features)
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(species_features)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def shared_residual_model(name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Feature Extraction from Pretrained Model
    features = layers.GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    shared_layer = layers.Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = layers.BatchNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    shared_layer = layers.Dropout(0.3)(shared_layer)
    
    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    genus_residual = layers.Dense(shared_neurons, activation='relu', name='genus_residual')(family_output)
    genus_features = layers.Add()([shared_layer, genus_residual])
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_features)
    
    # Species Output
    species_residual = layers.Dense(shared_neurons, activation='relu', name='species_residual')(genus_output)
    species_features = layers.Add()([shared_layer, species_residual])
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(species_features)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    return model

def shared_split_model(name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Feature Extraction from Pretrained Model
    features = layers.GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    # Split shared features for hierarchical outputs
    shared_family = layers.Dense(shared_neurons // 2, activation='relu', name='shared_family')(features)
    shared_family = layers.BatchNormalization()(shared_family)
    shared_family = layers.Activation('relu')(shared_family)
    shared_family = layers.Dropout(0.3)(shared_family)
    
    shared_genus_species = layers.Dense(shared_neurons // 2, activation='relu', name='shared_genus_species')(features)
    shared_genus_species = layers.BatchNormalization()(shared_genus_species)
    shared_genus_species = layers.Activation('relu')(shared_genus_species)
    shared_genus_species = layers.Dropout(0.3)(shared_genus_species)
    
    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(shared_family)
    
    # Genus Output (uses family features indirectly via shared_genus_species)
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(shared_genus_species)
    
    # Species Output (uses genus features indirectly via shared_genus_species)
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(shared_genus_species)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    return model

def local_attention_model (name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Feature Extracion from Pretrained Model
    features = layers.GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    #features = layers.GlobalMaxPooling2D()(x)
    
    shared_layer = layers.Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = layers.BatchNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    shared_layer = layers.Dropout(0.3)(shared_layer)
    
    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    genus_attention = layers.Dense(shared_neurons, activation='relu', name='genus_attention')(family_output)
    genus_features = layers.Multiply()([shared_layer, genus_attention])
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_features)
    
    # Species Output
    species_attention = layers.Dense(shared_neurons, activation='relu', name='species_attention')(genus_output)
    species_features = layers.Multiply()([shared_layer, species_attention])
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(species_features)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def advanced_conv_model(name, labels, pretrain, preprocess, input_shape, shared_filters=256, augmentation=None):
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Average Pooling to retain spatial structure
    pooled_features = layers.AveragePooling2D(pool_size=(2, 2), name='AvgPool2D')(x)
    
    # Shared Convolutional Layers
    shared_conv = layers.Conv2D(shared_filters, kernel_size=(3, 3), padding='same', activation='relu')(pooled_features)
    shared_conv = layers.BatchNormalization()(shared_conv)
    shared_conv = layers.Dropout(0.3)(shared_conv)

    # Family Branch
    family_conv = layers.Conv2D(shared_filters // 2, kernel_size=(3, 3), padding='same', activation='relu')(shared_conv)
    family_conv = layers.BatchNormalization()(family_conv)
    family_flatten = layers.Flatten()(family_conv)
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(family_flatten)

    # Genus Branch
    genus_conv = layers.Conv2D(shared_filters // 2, kernel_size=(3, 3), padding='same', activation='relu')(shared_conv)
    genus_conv = layers.BatchNormalization()(genus_conv)
    genus_flatten = layers.Flatten()(genus_conv)
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_flatten)

    # Species Branch
    species_conv = layers.Conv2D(shared_filters // 2, kernel_size=(3, 3), padding='same', activation='relu')(shared_conv)
    species_conv = layers.BatchNormalization()(species_conv)
    species_flatten = layers.Flatten()(species_conv)
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(species_flatten)

    # Build the model
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    return model

def advanced_conv_model3(name, labels, pretrain, preprocess, input_shape, shared_filters=256, augmentation=None):
    inputs = tf.keras.layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Average Pooling to retain spatial structure
    pooled_features = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), name='AvgPool2D')(x)
    
    # Shared Convolutional Layers
    shared_conv = tf.keras.layers.Conv2D(shared_filters // 2, kernel_size=(3, 3), padding='same', activation='relu')(pooled_features)
    shared_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(shared_conv)  # Add padding='same' to handle small sizes
    shared_conv = tf.keras.layers.Conv2D(shared_filters // 2, kernel_size=(3, 3), padding='same', activation='relu')(shared_conv)
    
    # Add a check to ensure pooling only happens if the dimensions are large enough
    if shared_conv.shape[1] > 1 and shared_conv.shape[2] > 1:
        shared_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(shared_conv)

    # Family Branch
    family_flatten = tf.keras.layers.Flatten()(shared_conv)
    family_flatten = tf.keras.layers.Dropout(0.5)(family_flatten)
    family_output = tf.keras.layers.Dense(len(labels['family']), activation='softmax', name='family')(family_flatten)

    # Genus Branch
    genus_conv = tf.keras.layers.Conv2D(shared_filters, kernel_size=(3, 3), padding='same', activation='relu')(shared_conv)
    genus_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(genus_conv)
    genus_flatten = tf.keras.layers.Flatten()(genus_conv)
    genus_flatten = tf.keras.layers.Dropout(0.5)(genus_flatten)
    genus_output = tf.keras.layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_flatten)

    # Species Branch
    species_conv = tf.keras.layers.Conv2D(shared_filters // 2, kernel_size=(3, 3), padding='same', activation='relu')(shared_conv)
    if species_conv.shape[1] > 1 and species_conv.shape[2] > 1:
        species_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(species_conv)
    species_flatten = tf.keras.layers.Flatten()(species_conv)
    species_flatten = tf.keras.layers.Dropout(0.5)(species_flatten)
    species_output = tf.keras.layers.Dense(len(labels['species']), activation='softmax', name='species')(species_flatten)

    # Build the model
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    return model


def advanced_conv_model2(name, labels, pretrain, preprocess, input_shape, shared_filters=256, augmentation=None):
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Average Pooling to retain spatial structure
    pooled_features = layers.AveragePooling2D(pool_size=(2, 2), name='AvgPool2D')(x)
    
    # Shared Convolutional Layers
    shared_conv = layers.Conv2D(shared_filters, kernel_size=(3, 3), padding='same', activation='relu')(pooled_features)
    shared_conv = layers.BatchNormalization()(shared_conv)
    shared_conv = layers.Dropout(0.3)(shared_conv)

    # Family Branch
    family_conv = layers.Conv2D(shared_filters // 2, kernel_size=(3, 3), padding='same', activation='relu')(shared_conv)
    family_conv = layers.BatchNormalization()(family_conv)
    family_conv = layers.GlobalAveragePooling2D()(family_conv)  # Replace Flatten with GlobalAveragePooling2D
    family_conv = layers.Dropout(0.5)(family_conv)  # Add Dropout
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(family_conv)

    # Genus Branch
    genus_conv = layers.Conv2D(shared_filters // 2, kernel_size=(3, 3), padding='same', activation='relu')(shared_conv)
    genus_conv = layers.BatchNormalization()(genus_conv)
    genus_conv = layers.GlobalMaxPooling2D()(genus_conv)  # Replace Flatten with GlobalMaxPooling2D
    genus_conv = layers.Dropout(0.5)(genus_conv)  # Add Dropout
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_conv)

    # Species Branch
    species_conv = layers.Conv2D(shared_filters // 2, kernel_size=(3, 3), padding='same', activation='relu')(shared_conv)
    species_conv = layers.BatchNormalization()(species_conv)
    species_conv = layers.GlobalAveragePooling2D()(species_conv)  # Replace Flatten with GlobalAveragePooling2D
    species_conv = layers.Dropout(0.5)(species_conv)  # Add Dropout
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(species_conv)

    # Build the model
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    return model

def attention_model (name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Attention
    x = MultiHeadAttention()(x)
    
    # Feature Extracion from Pretrained Model
    features = layers.GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    shared_layer = layers.Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = layers.BatchNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    shared_layer = layers.Dropout(0.3)(shared_layer)
    
    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    genus_features = layers.Concatenate()([shared_layer, family_output])
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_features)
    
    # Species Output
    species_features = layers.Concatenate()([shared_layer, family_output, genus_output])
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(species_features)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model


def phorcys_v09 (labels, input_shape):
    
    pretrain = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    pretrain.trainable = False

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
        ],
        name='data_augmentation'
    )

    inputs = layers.Input(shape=input_shape, name='input')
    x = augmentation(inputs)
    x = resnet_preprocess(x)
    x = pretrain(x, training=False)
    features = layers.GlobalAveragePooling2D()(x)

    shared_layer = layers.Dense(512,name='shared_layer')(features)
    shared_layer = layers.BatchNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    shared_layer = layers.Dropout(0.3)(shared_layer)

    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)

    # Genus Output
    genus_features = layers.Concatenate()([shared_layer, family_output])
    genus_hidden = layers.Dense(256, activation='relu')(genus_features)
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_hidden)

    # Species Output
    species_features = layers.Concatenate()([shared_layer, family_output, genus_output])
    species_hidden = layers.Dense(128, activation='relu')(species_features)
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(species_hidden)

    model = tf.keras.Model(
        inputs, 
        [family_output, genus_output, species_output],
        name = "PhorcysV9"
    )
    
    return model

def phorcys_v10 (labels, input_shape):
    
    pretrain = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    pretrain.trainable = False

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
        ],
        name='data_augmentation'
    )

    inputs = layers.Input(shape=input_shape, name='input')
    x = augmentation(inputs)
    x = resnet_preprocess(x)
    x = pretrain(x, training=False)
    features = layers.GlobalAveragePooling2D()(x)

    shared_layer = layers.Dense(2048,name='shared_layer')(features)
    shared_layer = layers.BatchNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    shared_layer = layers.Dropout(0.3)(shared_layer)

    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)

    # Genus Output
    family_features = layers.Dense(len(labels['genus']), activation='relu')(family_output)
    shared_features_genus = layers.Dense(len(labels['genus']), activation='relu')(shared_layer)
    genus_features = layers.Multiply()([shared_features_genus, family_features])
    genus_features = layers.LayerNormalization()(genus_features)
    #genus_hidden = layers.Dense(512, activation='relu')(genus_features)
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_features)

    # Species Output
    genus_features = layers.Dense(len(labels['species']), activation='relu')(genus_output)
    shared_features_species = layers.Dense(len(labels['species']), activation='relu')(shared_layer)
    species_features = layers.Multiply()([shared_features_species, genus_features])
    species_features = layers.LayerNormalization()(species_features)
    #species_hidden = layers.Dense(512, activation='relu')(species_features)
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(species_features)

    model = tf.keras.Model(
        inputs, 
        [family_output, genus_output, species_output],
        name = "PhorcysV10"
    )
    
    return model

def soat_shrdconc_model (name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Attention
    x = SoftAttention()(x)
    
    # Feature Extracion from Pretrained Model
    features = layers.GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    shared_layer = layers.Dense(shared_neurons, name='shared_layer')(features)
    shared_layer = layers.BatchNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    
    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    genus_features = layers.Concatenate()([shared_layer, family_output])
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_features)
    
    # Species Output
    species_features = layers.Concatenate()([shared_layer, family_output, genus_output])
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(species_features)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

def model_attention_simple (name, labels, pretrain, preprocess, input_shape, shared_neurons=2048, augmentation=None):
    
    inputs = Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
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
    x = preprocess(x)
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
    x = preprocess(x)
    
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
    x = preprocess(x)
    
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
    x = preprocess(x)
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


if __name__ in "__main__":
    
    import pandas as pd
    from cryptovision.tools import (
        image_directory_to_pandas,
        split_image_dataframe,
        tf_dataset_from_pandas
    )
    
    # Dataset
    SETUP = {
        "class_samples_threshold": 49,
        "test_size": .15,
        "validation_size": .15,
        "batch_size": 128,
        
        # Training
        "image_shape": (128, 128, 3),
        "epochs": 10,
        "learning_rate": 1e-4,
        "loss_type": {
            "family": "categorical_focal_crossentropy",
            "genus": "categorical_focal_crossentropy",
            "species": "categorical_focal_crossentropy",
        },
        "metrics": {
            "family": ["accuracy", "AUC", "Precision", "Recall"],
            "genus": ["accuracy", "AUC", "Precision", "Recall"],
            "species": ["accuracy", "AUC", "Precision", "Recall"],
        },
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 5,
            "restore_best_weights": True
        },
        "reduce_lr": {
            "monitor": "val_loss",
            "lr_factor": 0.5,
            "lr_patience": 3,
            "lr_min": 1e-6
        },   
    }
    
    #df_lab = image_directory_to_pandas("/Users/leonardo/Library/CloudStorage/Box-Box/CryptoVision/Data/fish_functions/Species_v03")
    #df_web = image_directory_to_pandas("/Users/leonardo/Library/CloudStorage/Box-Box/CryptoVision/Data/web/Species_v01")
    #df_inatlist = image_directory_to_pandas("/Users/leonardo/Library/CloudStorage/Box-Box/CryptoVision/Data/inaturalist/Species_v02")

    df = image_directory_to_pandas("/Users/leonardo/Documents/Projects/cryptovision/data/processed/cv_images_dataset")
    
    #df = pd.concat([df_lab, df_web, df_inatlist], ignore_index=True, axis=0)

    # find in the species column the values with lass than 50 occurences
    counts = df['species'].value_counts()
    df = df[df['species'].isin(counts[counts > SETUP['class_samples_threshold']].index)]

    train_df, val_df, test_df = split_image_dataframe(df, test_size=0.15, val_size=0.15, stratify_by='folder_label')

    names = {
        'family': sorted(df['family'].unique()),
        'genus': sorted(df['genus'].unique()),
        'species': sorted(df['species'].unique()),
    }

    train_ds, _, _, _ = tf_dataset_from_pandas(train_df, batch_size=128, image_size=SETUP['image_shape'][:2])
    val_ds, _, _, _ = tf_dataset_from_pandas(val_df, batch_size=128, image_size=SETUP['image_shape'][:2])
    test_ds, _, _, _ = tf_dataset_from_pandas(test_df, batch_size=128, image_size=SETUP['image_shape'][:2])
    

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    valid_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    
    efnet = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=SETUP['image_shape'])
    efnet.trainable = False
    
    efnetb3 = EfficientNetV2B3(include_top=False, weights='imagenet', input_shape=SETUP['image_shape'])
    efnetb3.trainable = False
    
    resnet = ResNet50V2(include_top=False, weights='imagenet', input_shape=SETUP['image_shape'])
    resnet.trainable = False
    
    resnet101 = ResNet101V2(include_top=False, weights='imagenet', input_shape=SETUP['image_shape'])
    resnet101.trainable = False
    
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(height_factor=(0.05, 0.1), width_factor=(0.05, 0.1)),  # Wider zoom range
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomCrop(SETUP['image_shape'][0], SETUP['image_shape'][1]),
            layers.GaussianNoise(0.1),
        ],
        name='augmentation'
    )
    
    
    import wandb
    from loguru import logger
    from wandb.integration.keras import WandbMetricsLogger
    import datetime

    wandb.require("core")

    PROJ = "CryptoVision - Architecture Testing"
    ARCH = "phorcys_v10"
    PTRAIN = "rn50v2"
    VERSION = datetime.datetime.now().strftime("%y.%m%d.%H%M")
    NICKNAME = f"cvis_{PTRAIN}_{SETUP['image_shape'][0]}_{ARCH}_v{VERSION}"
    SETUP['augmentation'] = True
    SETUP['shared_neurons'] = 2048
    SETUP['dropout'] = 0.5
    #SETUP['shared_filters'] = 128
    #SETUP['hidden_neurons'] = [512, 1024, 2048]


    with wandb.init(project=PROJ, name=NICKNAME, config={**SETUP}) as run:
        
        #model = shared_concat_modelv2(
        #    name=NICKNAME,
        #    labels=names,
        #    pretrain=resnet,
        #    shared_neurons=SETUP['shared_neurons'],
        #    #shared_filters=SETUP['shared_filters'],
        #    preprocess= resnet_preprocess,
        #    augmentation=augmentation,
        #    input_shape=SETUP['image_shape'],
        #)
        
        model = phorcys_v09(
            labels=names,
            input_shape=SETUP['image_shape'],
        )
        
        logger.info(print(model.summary(show_trainable=True)))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=SETUP["learning_rate"]),
            loss=SETUP['loss_type'],
            metrics=SETUP['metrics'],
            #loss_weights=SETUP["loss_weights"],
        )
        
        wandb_logger = WandbMetricsLogger()
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=SETUP['early_stopping']["monitor"],
            patience=SETUP["early_stopping"]['patience'],
            restore_best_weights=SETUP['early_stopping']['restore_best_weights'],
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=SETUP['reduce_lr']["monitor"],
            factor=SETUP['reduce_lr']["lr_factor"],
            patience=SETUP['reduce_lr']["lr_patience"],
            min_lr=SETUP['reduce_lr']["lr_min"],
        )
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"models/{NICKNAME}.keras",
            monitor="val_loss", 
            save_best_only=True,  
            mode="min",  
            verbose=0  
        )
        
        history = model.fit(
            train_ds,
            epochs=SETUP["epochs"],
            validation_data=val_ds,
            callbacks=[wandb_logger, early_stopping, reduce_lr, checkpoint, TQDMProgressBar()],
            verbose=0,
        )
        
        logger.success(f"Model {NICKNAME} trained and logged to wandb.")
        