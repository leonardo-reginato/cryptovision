import tensorflow as tf
from tensorflow.keras.layers import ( # type: ignore
    Dense, GlobalAveragePooling2D, Dropout,
    BatchNormalization, Activation, Multiply,
    Add, Concatenate, Input, Reshape, Layer, Attention,
    MultiHeadAttention, Conv2D, MaxPooling2D, Flatten
)
from tensorflow.keras import backend as K                                                       # type: ignore
from tensorflow.keras.models import Model                                                       # type: ignore
from tensorflow.keras.applications import ResNet50V2                                            # type: ignore
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess       # type: ignore

# Data Augmentation
def augmentation_layer(
    flip="horizontal",
    rotation=0.2,
    zoom=0.2,
    translation=(0.1, 0.1),
    contrast=0.2,
    brightness=0.2
):
    if flip not in {"horizontal", "vertical", "horizontal_and_vertical"}:
        raise ValueError("flip must be 'horizontal', 'vertical', or 'horizontal_and_vertical'")

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(flip),
            tf.keras.layers.RandomRotation(rotation),
            tf.keras.layers.RandomZoom(zoom),
            tf.keras.layers.RandomTranslation(*translation),
            tf.keras.layers.RandomContrast(contrast),
            tf.keras.layers.RandomBrightness(brightness),
        ],
        name="data_augmentation_layer"
    )


# Define a custom SE Block for 4D inputs
def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)

    return Multiply()([input_tensor, se])


# Define a custom Self-Attention Layer
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

        return tf.matmul(attention_scores, v) + inputs  # Residual connection


# Transformer Layer Function
def transformer_layer(neurons, activation='relu', dropout_rate=0.2):
    def layer(x):
        x = Dense(neurons, activation=None)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(dropout_rate)(x)
        return x
    return layer


def independent_multi_output_model(
    input_shape=(384, 384, 3),
    n_families=10,
    n_genera=10,
    n_species=10,
    base_weights="imagenet",
    base_trainable=False,
    augmentation_layer=None,
    shared_layer_neurons=512,
    family_neurons=256,
    genus_neurons=256,
    species_neurons=256,
    dropout_rate=0.3
):
    """
    Creates a multi-output deep learning model where each output is independent,
    with no shared layers between the outputs.

    Args:
        input_shape (tuple): Shape of the input image.
        n_families (int): Number of family classes.
        n_genera (int): Number of genus classes.
        n_species (int): Number of species classes.
        base_weights (str): Weights for the base model (e.g., "imagenet").
        base_trainable (bool): Whether the base model is trainable.
        augmentation_layer (Layer): Data augmentation layer to apply before the model.
        family_neurons (int): Number of neurons in the dense layer for family prediction.
        genus_neurons (int): Number of neurons in the dense layer for genus prediction.
        species_neurons (int): Number of neurons in the dense layer for species prediction.
        dropout_rate (float): Dropout rate to apply in dense layers.

    Returns:
        Model: A Keras model with independent multi-output architecture.
    """
    # Base Model
    base_model = ResNet50V2(include_top=False, weights=base_weights, input_shape=input_shape)
    base_model.trainable = base_trainable

    # Input and data augmentation layers
    inputs = Input(shape=input_shape)
    x = augmentation_layer(inputs) if augmentation_layer else inputs
    x = resnet_preprocess(x)
    x = base_model(x, training=False)


    # Shared dense layer for better feature learning
    global_features = GlobalAveragePooling2D()(x)
    shared_layer = Dense(shared_layer_neurons, activation=None, name='shared_layer')(global_features)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)
    
    # Family branch
    family_branch = Dense(family_neurons, activation="relu")(global_features)
    family_branch = Dropout(dropout_rate)(family_branch)
    family_output = Dense(n_families, activation="softmax", name="family")(family_branch)

    # Genus branch
    genus_branch = Dense(genus_neurons, activation="relu")(global_features)
    genus_branch = Dropout(dropout_rate)(genus_branch)
    genus_output = Dense(n_genera, activation="softmax", name="genus")(genus_branch)

    # Species branch
    species_branch = Dense(species_neurons, activation="relu")(global_features)
    species_branch = Dropout(dropout_rate)(species_branch)
    species_output = Dense(n_species, activation="softmax", name="species")(species_branch)

    # Create the model
    model = Model(inputs=inputs, outputs=[family_output, genus_output, species_output])

    return model

# Proteon Model Function
def proteon(
    input_shape=(224, 224, 3), 
    n_families=10, 
    n_genera=10, 
    n_species=10, 
    base_weights="imagenet", 
    base_trainable=False, 
    se_ratio=16, 
    shared_layer_neurons=512, 
    shared_layer_dropout=0.3, 
    family_transform_neurons=512, 
    genus_transform_neurons=512, 
    species_transform_neurons=512, 
    attention_neurons=512, 
    augmentation_layer=None,  # Custom data augmentation layer
    transformer_activation="relu", 
    transformer_dropout=0.2
):
    
    # Base Model
    base_model = ResNet50V2(include_top=False, weights=base_weights, input_shape=input_shape)
    base_model.trainable = base_trainable

    # Input and data augmentation layers
    inputs = Input(shape=input_shape)
    x = augmentation_layer(inputs) if augmentation_layer else inputs  
    x = resnet_preprocess(x)  
    x = base_model(x, training=False)

    # Squeeze-and-Excite (SE) Block
    x = squeeze_excite_block(x, ratio=se_ratio)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(shared_layer_dropout)(x)

    # Shared Layer with Self-Attention
    shared_layer = Dense(shared_layer_neurons, activation=None)(x)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation("relu")(shared_layer)
    shared_layer = SelfAttention()(shared_layer)
    shared_layer = Dropout(shared_layer_dropout)(shared_layer)

    # Family Output
    family_transform = transformer_layer(
        neurons=family_transform_neurons, 
        activation=transformer_activation, 
        dropout_rate=transformer_dropout
    )(shared_layer)
    family_output = Dense(n_families, activation="softmax", name="family")(family_transform)

    # Family Features
    family_attention = Dense(shared_layer_neurons, activation="sigmoid")(family_transform)
    family_features = Multiply()([shared_layer, family_attention])
    family_features = Concatenate()([family_features, family_output])

    # Genera Output
    genus_transform = transformer_layer(
        neurons=genus_transform_neurons, 
        activation=transformer_activation, 
        dropout_rate=transformer_dropout
    )(family_features)
    genus_residual = Dense(attention_neurons, activation="relu")(family_output)
    genus_hidden = Add()([genus_transform, genus_residual])
    genus_output = Dense(n_genera, activation="softmax", name="genus")(genus_hidden)

    # Genus Features
    genus_attention = Dense(shared_layer_neurons, activation="sigmoid")(genus_hidden)
    genus_features = Multiply()([shared_layer, genus_attention])
    genus_features = Concatenate()([genus_features, family_output, genus_output])

    # Species Output
    species_transform = transformer_layer(
        neurons=species_transform_neurons, 
        activation=transformer_activation, 
        dropout_rate=transformer_dropout
    )(genus_features)
    species_residual = Dense(attention_neurons, activation="relu")(Concatenate()([family_output, genus_output]))
    species_hidden = Add()([species_transform, species_residual])
    species_output = Dense(n_species, activation="softmax", name="species")(species_hidden)

    return Model(inputs=inputs, outputs=[family_output, genus_output, species_output])


# Simple Model
def phorcys(
    n_families, 
    n_genera, 
    n_species,
    attention=False, 
    input_shape=(224,224,3), 
    base_weights="imagenet", 
    base_trainable=False, 
    augmentation_layer=None,
    shared_layer_neurons=512,
    shared_layer_dropout=0.3,
    genus_hidden_neurons=512,
    species_hidden_neurons=512,
    num_heads=4,
):
    
    # Base Model
    base_model = ResNet50V2(include_top=False, weights=base_weights, input_shape=input_shape)
    base_model.trainable = base_trainable

    # Input and data augmentation layers
    inputs = Input(shape=input_shape)
    x = augmentation_layer(inputs) if augmentation_layer else inputs  
    x = resnet_preprocess(x)  
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Shared dense layer for better feature learning
    shared_layer = tf.keras.layers.Dense(shared_layer_neurons, activation=None, name='shared_layer')(x)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('relu')(shared_layer)
    
    # Attention Layer
    if attention:
        shared_layer_reshaped = Reshape((1, shared_layer_neurons))(shared_layer)
        if attention == "mha":
            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=shared_layer_neurons)(shared_layer_reshaped, shared_layer_reshaped)
        else:
            attention_output = Attention()([shared_layer_reshaped, shared_layer_reshaped])
        attention_output = Reshape((shared_layer_neurons,))(attention_output)  # Reshape back to 2D
        shared_layer = Add()([shared_layer, attention_output])  # Residual connection
    
    shared_layer = Dropout(shared_layer_dropout)(shared_layer)

    # Define family output
    family_output = tf.keras.layers.Dense(n_families, activation='softmax', name='family')(shared_layer)

    # Concatenate the family output with the base model output
    family_features = tf.keras.layers.Concatenate()([shared_layer, family_output])

    # Define genus output, using family features as additional input
    genus_hidden = tf.keras.layers.Dense(genus_hidden_neurons, activation='relu')(family_features)
    genus_output = tf.keras.layers.Dense(n_genera, activation='softmax', name='genus')(genus_hidden)

    # Concatenate the family and genus outputs with the base model output
    genus_features = tf.keras.layers.Concatenate()([shared_layer, family_output, genus_output])

    # Define species output, using both family and genus features as additional input
    species_hidden = tf.keras.layers.Dense(species_hidden_neurons, activation='relu')(genus_features)
    species_output = tf.keras.layers.Dense(n_species, activation='softmax', name='species')(species_hidden)

    # Create the hierarchical model
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output])
    
    return model


# Simple Model
def phorcys_conv(
    n_families, 
    n_genera, 
    n_species,
    attention=False, 
    input_shape=(224,224,3), 
    base_weights="imagenet", 
    base_trainable=False, 
    augmentation_layer=None,
    shared_layer_neurons=512,
    shared_layer_dropout=0.3,
    genus_hidden_neurons=512,
    species_hidden_neurons=512,
    num_heads=4,
):
    
    # Base Model
    base_model = ResNet50V2(include_top=False, weights=base_weights, input_shape=input_shape)
    base_model.trainable = base_trainable

    # Input and data augmentation layers
    inputs = Input(shape=input_shape)
    x = augmentation_layer(inputs) if augmentation_layer else inputs  
    x = resnet_preprocess(x)  
    x = base_model(x, training=False)
    rn_features_extracted = x  # Retain spatial features from ResNet
    
    # Global pooling for shared feature extraction
    shared_features = GlobalAveragePooling2D()(x)
    shared_features = Dense(shared_layer_neurons, activation=None, name='shared_layer')(shared_features)
    shared_features = BatchNormalization()(shared_features)
    shared_features = Activation('relu')(shared_features)
    shared_features = Dropout(shared_layer_dropout)(shared_features)
    
    # Attention Layer
    if attention:
        shared_features_reshaped = Reshape((1, shared_layer_neurons))(shared_features)
        attention_output = Attention()([shared_features_reshaped, shared_features_reshaped])
        attention_output = Reshape((shared_layer_neurons,))(attention_output) 
        shared_features = Add()([shared_features, attention_output])
    

    # Define family output
    family_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(rn_features_extracted)
    family_pool = MaxPooling2D((2, 2))(family_conv)
    family_flatten = Flatten()(family_pool)
    family_features = Concatenate()([shared_features, family_flatten])
    family_output = Dense(n_families, activation='softmax', name='family')(family_features)

    # Genus output
    genus_conv = Conv2D(128, (3, 3), activation='relu', padding='same')(rn_features_extracted)
    genus_pool = MaxPooling2D((2, 2))(genus_conv)
    genus_flatten = Flatten()(genus_pool)
    genus_features = Concatenate()([shared_features, family_output, genus_flatten])
    genus_hidden = Dense(genus_hidden_neurons, activation='relu')(genus_features)
    genus_output = Dense(n_genera, activation='softmax', name='genus')(genus_hidden)

    # Species output
    species_conv = Conv2D(256, (3, 3), activation='relu', padding='same')(rn_features_extracted)
    species_pool = MaxPooling2D((2, 2))(species_conv)
    species_flatten = Flatten()(species_pool)
    species_features = Concatenate()([shared_features, family_output, genus_output, species_flatten])
    species_hidden = Dense(species_hidden_neurons, activation='relu')(species_features)
    species_output = Dense(n_species, activation='softmax', name='species')(species_hidden)

    # Create the hierarchical model
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output])
    
    return model

# Stable Convolutional Model
def stable_phorcys_conv(
    n_families, 
    n_genera, 
    n_species,
    input_shape=(384,384,3), 
    base_weights="imagenet", 
    base_trainable=False, 
    augmentation_layer=None,
    shared_layer_neurons=512,
    shared_layer_dropout=0.3,
    genus_hidden_neurons=512,
    species_hidden_neurons=512
):
    """
    A stable, simplified version of a hierarchical classification model.
    Concepts:
    - Pretrained backbone (ResNet50V2) for robustness.
    - GlobalAveragePooling2D to produce a stable shared feature vector.
    - Minimal, single convolutional layers before each output head.
    - BatchNormalization, Dropout, and a single Dense layer for shared features.
    - Parallel predictions without attention or complex hierarchical logic.
    
    This provides a solid baseline that you can refine if needed.
    """

    # Base model
    base_model = ResNet50V2(include_top=False, weights=base_weights, input_shape=input_shape)
    base_model.trainable = base_trainable

    # Input and optional augmentation
    inputs = Input(shape=input_shape)
    x = augmentation_layer(inputs) if augmentation_layer else inputs
    x = resnet_preprocess(x)
    x = base_model(x, training=False)

    # Shared features: stable and simple
    shared_features = GlobalAveragePooling2D()(x)
    shared_features = Dense(shared_layer_neurons, activation=None, name='shared_layer')(shared_features)
    shared_features = BatchNormalization()(shared_features)
    shared_features = Activation('relu')(shared_features)
    shared_features = Dropout(shared_layer_dropout)(shared_features)

    # Family branch
    family_conv = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    family_pool = MaxPooling2D((2, 2))(family_conv)
    family_flatten = Flatten()(family_pool)
    family_combined = Concatenate()([shared_features, family_flatten])
    family_output = Dense(n_families, activation='softmax', name='family')(family_combined)

    # Genus branch
    genus_conv = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    genus_pool = MaxPooling2D((2, 2))(genus_conv)
    genus_flatten = Flatten()(genus_pool)
    genus_combined = Concatenate()([shared_features, genus_flatten])
    genus_hidden = Dense(genus_hidden_neurons, activation='relu')(genus_combined)
    genus_output = Dense(n_genera, activation='softmax', name='genus')(genus_hidden)

    # Species branch
    species_conv = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    species_pool = MaxPooling2D((2, 2))(species_conv)
    species_flatten = Flatten()(species_pool)
    species_combined = Concatenate()([shared_features, species_flatten])
    species_hidden = Dense(species_hidden_neurons, activation='relu')(species_combined)
    species_output = Dense(n_species, activation='softmax', name='species')(species_hidden)

    # Create the hierarchical model with parallel outputs
    model = Model(inputs, [family_output, genus_output, species_output])
    return model


# Focal Loss function
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = -y_true * (alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred))
        return K.sum(loss, axis=1)
    return focal_loss_fixed


# Hierarchical consistency loss
def hierarchical_consistency_loss(y_true_family, y_pred_family, y_true_genus, y_pred_genus, y_true_species, y_pred_species):
    # Calculate family-level loss
    family_loss = focal_loss()(y_true_family, y_pred_family)
    
    # Genus-level loss with hierarchical penalty based on family misclassification
    genus_loss = focal_loss()(y_true_genus, y_pred_genus)
    genus_penalty = tf.reduce_mean(family_loss) * genus_loss  # Apply penalty if family is wrong
    
    # Species-level loss with penalty based on both family and genus misclassification
    species_loss = focal_loss()(y_true_species, y_pred_species)
    species_penalty = (tf.reduce_mean(family_loss) + tf.reduce_mean(genus_loss)) * species_loss
    
    # Total loss as a combination of focal loss for each level and penalties for hierarchical inconsistency
    return family_loss + genus_penalty + species_penalty


# Final custom loss function combining Focal Loss and Hierarchical Consistency Loss
def combined_hierarchical_loss(y_true_family, y_pred_family, y_true_genus, y_pred_genus, y_true_species, y_pred_species):
    family_loss = focal_loss()(y_true_family, y_pred_family)
    genus_loss = focal_loss()(y_true_genus, y_pred_genus)
    species_loss = focal_loss()(y_true_species, y_pred_species)
    
    # Hierarchical consistency loss
    hierarchy_loss = hierarchical_consistency_loss(y_true_family, y_pred_family, y_true_genus, y_pred_genus, y_true_species, y_pred_species)
    
    # Combined loss with hierarchy penalty
    return family_loss + genus_loss + species_loss + 0.1 * hierarchy_loss  # Adjust weighting as needed

