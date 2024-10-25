
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import wandb
from cryptovision.config import MODEL_PARAMS, PROCESSED_DATA_DIR
from tensorflow.keras.layers import Layer

# Vision Transformer specific layers
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

# Modify the MLP function to output the correct dimension
def mlp(x, hidden_units, projection_dim, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    # Ensure the final output matches the projection dimension
    x = layers.Dense(projection_dim)(x)
    return x

# Update the create_vit_classifier function to pass projection_dim to MLP
def create_vit_classifier(input_shape, patch_size, num_patches, projection_dim, transformer_layers, num_heads, mlp_units):
    inputs = layers.Input(shape=input_shape)

    # Data augmentation
    data_augmentation = tf.keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(224, 224),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )
    augmented = data_augmentation(inputs)

    # Create patches
    patches = Patches(patch_size)(augmented)

    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple transformer layers
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP with final output matching the projection dimension
        x3 = mlp(x3, hidden_units=mlp_units, projection_dim=projection_dim, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    # Classify outputs
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    # MLP head for classification
    features = mlp(representation, hidden_units=mlp_units, projection_dim=projection_dim, dropout_rate=0.5)
    outputs = layers.Dense(len(class_names), activation='softmax')(features)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    # Prepare dataset
    train_dir = PROCESSED_DATA_DIR / "train"
    valid_dir = PROCESSED_DATA_DIR / "valid"
    test_dir = PROCESSED_DATA_DIR / "test"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=MODEL_PARAMS["batch_size"],
        image_size=MODEL_PARAMS["image_shape"],
        seed=MODEL_PARAMS["random_state"],
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="categorical",
        batch_size=MODEL_PARAMS["batch_size"],
        image_size=MODEL_PARAMS["image_shape"],
        seed=MODEL_PARAMS["random_state"],
    )
    
    class_names = train_ds.class_names

    # Create Vision Transformer classifier
    vit_classifier = create_vit_classifier(
        input_shape=(224, 224, 3),
        patch_size=16,
        num_patches=196,
        projection_dim=64,
        transformer_layers=8,
        num_heads=4,
        mlp_units=[2048, 1024]
    )

    # Compile the model
    vit_classifier.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6)
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Train the model
    vit_classifier.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=MODEL_PARAMS["epochs"],
        callbacks=[reduce_lr, early_stop]
    )
