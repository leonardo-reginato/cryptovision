import pathlib
import numpy as np
import tensorflow as tf
import wandb
from loguru import logger
from wandb.integration.keras import WandbMetricsLogger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tensorflow.keras.applications import ( # type: ignore
    Xception, VGG16, ResNet50V2, ResNet152V2, InceptionV3,
    MobileNetV2, EfficientNetV2B0, InceptionResNetV2
)
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess                          # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess                                # type: ignore
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess                           # type: ignore
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess                     # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess                  # type: ignore
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess               # type: ignore
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_preprocess       # type: ignore

wandb.require("core")

tf.keras.mixed_precision.set_global_policy('mixed_float16')

settings = {
    'batch_size': 32,
    'epochs': 10,
    'fine_tune_epochs': 10,
    'img_size': (224, 224),
    'seed': 42,
    'base_learning_rate': 0.0001,
    'fine_tune_learning_rate': 0.00001,
    'dropout_rate': 0.2,
    'random_flip': 'horizontal',
    'random_rotation': 0.2,
    'random_zoom': 0.2,
    'random_translation': 0.1,
    'random_contrast': 0.2,
    'random_brightness': 0.2,
    'fine_tune_last_layers': 70,
    'training_metrics': ['accuracy', 'precision', 'recall',],
    'training_loss': 'categorical_crossentropy',
}


# Paths for datasets
data_dirs = {
    'train': pathlib.Path("/Users/leonardo/Documents/Projects/cryptovision/data/processed/train"),
    'val': pathlib.Path("/Users/leonardo/Documents/Projects/cryptovision/data/processed/valid"),
    'test': pathlib.Path("/Users/leonardo/Documents/Projects/cryptovision/data/processed/test")
}

# Data Loading and Preprocessing
def load_datasets(data_dirs, img_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dirs['train'],
        shuffle=True,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dirs['val'],
        shuffle=True,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dirs['test'],
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    class_names = train_ds.class_names
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names

# Function to evaluate and log metrics
def evaluation(model, model_name, test_ds):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.append(np.argmax(labels, axis=1))
        y_pred.append(np.argmax(preds, axis=1))
    
    # Concatenate arrays
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    logger.success(f'{model_name} - Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    
    return accuracy, f1, precision, recall

model_dict = {
    'VGG16': (VGG16, vgg16_preprocess),
    'ResNet50V2': (ResNet50V2, resnet_preprocess),
    #'ResNet152V2': (ResNet152V2, resnet_preprocess),
    #'InceptionV3': (InceptionV3, inception_preprocess),
    'MobileNetV2': (MobileNetV2, mobilenet_v2_preprocess),
    'EfficientNetV2B0': (EfficientNetV2B0, efficientnet_preprocess),
    #'InceptionResNetV2': (InceptionResNetV2, inception_resnet_preprocess)
}

train_ds, val_ds, test_ds, class_names = load_datasets(
    data_dirs, 
    settings['img_size'], 
    settings['batch_size']
)

# Train, Fine-tune, and Evaluate Models
models_and_histories = {}
for model_name, (model_class, preprocess_input) in model_dict.items():
    
    # Initialize WandB project with configuration
    with wandb.init(
        project='CryptoVision - PreTrain Model Selection', 
        name=model_name,
        config={
            **settings,
            'model_name': model_name
        }
    ) as run:
    
        config = wandb.config
        
        logger.info(f"Training model: {model_name}")
        
        # Data Augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(config.random_flip),
            tf.keras.layers.RandomRotation(config.random_rotation),
            tf.keras.layers.RandomZoom(config.random_zoom),
            tf.keras.layers.RandomTranslation(config.random_translation, config.random_translation),
            tf.keras.layers.RandomContrast(config.random_contrast),
            tf.keras.layers.RandomBrightness(config.random_brightness),
        ])
        
        # Base model
        base_model = model_class(
            input_shape=config.img_size + [3],
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        # Model building
        inputs = tf.keras.Input(shape=config.img_size + [3])
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(config.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)
        
        # Compile and train model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.base_learning_rate),
            loss=config.training_loss,
            metrics=config.training_metrics
        )
        
        wandb_logger = WandbMetricsLogger()
        
        checkpoint_path = f'/Users/leonardo/Documents/Projects/cryptovision/models/{model_name}_best_model.keras'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            callbacks=[wandb_logger, model_checkpoint_callback]
        )
        
        # Fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-config.fine_tune_last_layers]:
            layer.trainable = False
            
        logger.info(f"Unfreezing the last {config.fine_tune_last_layers} layers")

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=config.fine_tune_learning_rate),
            loss=config.training_loss,
            metrics=config.training_metrics
        )
        
        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs + config.fine_tune_epochs,
            initial_epoch=history.epoch[-1],
            callbacks=[wandb_logger]
        )
        
        acc, f1, precision, recall = evaluation(model, model_name, test_ds)
        
        # Log metrics to wandb
        wandb.log({
            'accuracy': acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
        })
        
        # Save model and history to wandb and dictionary
        wandb.log_artifact(
            f'{model_name}_model.keras', 
            name=model_name, 
            type='model',
            aliases=[f'ACC{int(acc*10_000)}', 'TLNR', 'FTUN', 'ONEOUT', 'SPECIES', f'{model_name}_base_model']
        )
        
        logger.success(f"Model {model_name} trained and logged to wandb.")

        wandb.finish()