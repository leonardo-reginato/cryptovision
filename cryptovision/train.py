import os
import wandb
import random
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from wandb.integration.keras import WandbMetricsLogger
from sklearn.model_selection import train_test_split
from cryptovision.tools import (
    TQDMProgressBar, image_dir_pandas, 
    split_dataframe, tensorflow_dataset
)

from tensorflow.keras import Layer, layers, backend                 # type: ignore
from tensorflow.keras import applications as keras_apps             # type: ignore
from tensorflow.keras.saving import register_keras_serializable     # type: ignore

from cryptovision import models

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
    
wandb.require("core")

class SaveModelAtEpochs(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, save_epochs):
        """
        Callback to save the model at specific epochs.
        
        Parameters:
        - save_dir: str, Directory to save the models.
        - save_epochs: list, List of epochs at which to save the model.
        """
        super(SaveModelAtEpochs, self).__init__()
        self.save_dir = save_dir
        self.save_epochs = set(save_epochs)  # Use a set for faster lookup
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """
        Save the model at specified epochs.
        """
        # Epochs in Keras are 0-indexed, so add 1
        current_epoch = epoch + 1
        if current_epoch in self.save_epochs:
            model_path = os.path.join(self.save_dir, f"model_epoch_{current_epoch}.keras")
            self.model.save(model_path)
            print(f"\nModel saved at: {model_path}")
    
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

def phorcys_v10 (labels, augmentation, input_shape=(224, 224, 3), size="m",  name=None, dropout_rate=0.3):
    """
    Phorcys V10 model architecture.

    Args:
    - labels: dictionary, Labels for each level of the hierarchy.
    - augmentation: function, Data augmentation function.
    - input_shape: tuple, Shape of the input data.
    - size: str, Size of the model. Must be one of 'xs', 's', 'm', 'l'.
    """
    if size == "xs":
        factor = .25
    elif size == "s":
        factor = .5
    elif size == "m":
        factor = 1
    elif size == "l":
        factor = 2    
    else:
        raise ValueError("Size must be one of 'xs', 's', 'm', 'l'")
    
    #pretrain = keras_apps.ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    pretrain = keras_apps.ResNet152V2(include_top=False, weights='imagenet', input_shape=input_shape)
    pretrain.trainable = False

    inputs = layers.Input(shape=input_shape, name='input')
    x = augmentation(inputs)
    x = keras_apps.resnet_v2.preprocess_input(x)
    x = pretrain(x, training=False)
    features = layers.GlobalAveragePooling2D()(x)
    
    layer_size = int(1024 * factor)

    shared_layer = layers.Dense(layer_size, name='shared_layer')(features)
    shared_layer = layers.BatchNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    shared_layer = layers.Dropout(dropout_rate)(shared_layer)

    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)

    # Genus Output
    family_mask = layers.Dense(layer_size, activation='relu', name='family_mask')(family_output)
    genus_features = layers.Multiply()([shared_layer, family_mask])
    genus_features = layers.LayerNormalization()(genus_features)
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(genus_features)

    # Species Output
    genus_mask = layers.Dense(layer_size, activation='relu', name='genus_mask')(genus_output)
    species_features = layers.Multiply()([shared_layer, genus_mask])
    species_features = layers.LayerNormalization()(species_features)
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(species_features)

    model = tf.keras.Model(
        inputs, 
        [family_output, genus_output, species_output],
        name = "PhorcysV10" if name is None else name
    )
    
    return model

def basic_model (name, labels, pretrain, preprocess, input_shape, augmentation=None):
    
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = augmentation(inputs) if augmentation else inputs
    x = preprocess(x)
    x = pretrain(x, training=False)
    
    # Feature Extracion from Pretrained Model
    features = layers.GlobalAveragePooling2D(name='GlobAvgPool2D')(x)
    
    # Shared Layer
    shared_layer = layers.Dense(features.shape[-1], name='shared_layer')(features)
    shared_layer = layers.BatchNormalization()(shared_layer)
    shared_layer = layers.Activation('relu')(shared_layer)
    shared_layer = layers.Dropout(0.3)(shared_layer)
    
    
    
    # Family Output
    family_output = layers.Dense(len(labels['family']), activation='softmax', name='family')(shared_layer)
    
    # Genus Output
    genus_output = layers.Dense(len(labels['genus']), activation='softmax', name='genus')(shared_layer)
    
    # Species Output
    species_output = layers.Dense(len(labels['species']), activation='softmax', name='species')(shared_layer)
    
    model = tf.keras.Model(inputs, [family_output, genus_output, species_output], name=name)
    
    return model

if __name__ == '__main__':
    
    SETUP = {
        "seed": SEED,
        "verbose": 0,
        "version": f"v{datetime.datetime.now().strftime('%y%m.%d.%H%M')}",
        "project": 'DataSet Comparison',
        "pretrain": "RN50v2",
        "finetune": False,
        
        "model": {
            "function": models.basic_multioutput,
            "args": {
                "dropout_rate": 0.3,
            },
            "save": False,
        },
        
        "image": {
            "size": (128, 128),
            "shape": (128, 128, 3),
        },
       
        "dataset": {
            "version": "v2.0.0",
            "test_size": .15,
            "validation_size": .15,
            "batch_size": 128,
            "class_samples_threshold": 90,
            "stratify_by": 'folder_label',
            "sources": ['fish_functions_v02','web', 'inaturalist_v03',],
        },
        
        "compile": {
            "lr": 1e-4,
            "loss": {
                "family": "categorical_focal_crossentropy",
                "genus": "categorical_focal_crossentropy",
                "species": "categorical_focal_crossentropy",
            },
            "metrics": {
                "family": ["accuracy", "AUC", "Precision", "Recall"],
                "genus": ["accuracy", "AUC", "Precision", "Recall"],
                "species": ["accuracy", "AUC", "Precision", "Recall"],
            },
            "loss_weights": {
                "family": 1.0,
                "genus": 1.5,
                "species": 2.0,
            }
        },
        
        "train": {
            "epochs": 10,  
        },
        
        "early_stop": {
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
        
        "checkpoint": {
            "monitor": "val_loss",
            "save_best_only": True,
            "mode": "min",
        }, 
    
        "fine_tune": {
            "epochs": 15,
            "layers": 25,
            "lr": 1e-5,
            "pretrain_layer": "resnet50v2",
            "patience": 7
        },
    }
    
    df_inat = image_directory_to_pandas(
        '/Volumes/T7_shield/CryptoVision/Data/Images/Sources/INaturaList/Species/v250116/images'
    )

    df_inat_clean = image_directory_to_pandas(
        '/Volumes/T7_shield/CryptoVision/Data/Images/Sources/INaturaList/Species/v250128/images'
    )

    species_list = df_inat_clean['species'].unique()

    df_inat = df_inat[df_inat['species'].isin(species_list)]
    
    df = df_inat_clean.copy()
    
    #counts = df['species'].value_counts()
    #df = df[df['species'].isin(counts[counts >= SETUP['dataset']['class_samples_threshold']].index)]

    train_df, val_df, test_df = split_image_dataframe(
        df, 
        test_size=SETUP['dataset']['test_size'], 
        val_size=SETUP['dataset']['validation_size'], 
        stratify_by=SETUP['dataset']['stratify_by'],
        random_state=SEED
    )

    names = {
        'family': sorted(df['family'].unique()),
        'genus': sorted(df['genus'].unique()),
        'species': sorted(df['species'].unique()),
    }

    train_ds, _, _, _ = tf_dataset_from_pandas(
        train_df, 
        batch_size=SETUP['dataset']['batch_size'], 
        image_size=SETUP['image']['size'],
    )
    
    val_ds, _, _, _ = tf_dataset_from_pandas(
        val_df, 
        batch_size=SETUP['dataset']['batch_size'], 
        image_size=SETUP['image']['size'],
    )
    
    test_ds, _, _, _ = tf_dataset_from_pandas(
        test_df, 
        batch_size=SETUP['dataset']['batch_size'], 
        image_size=SETUP['image']['size'],
    )
    
    train_ds = train_ds.cache().shuffle(buffer_size=1000, seed=SEED).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    logger.info(f"Dataset sizes:\n\tTrain: {train_df.shape[0]}\n\tVal: {val_df.shape[0]}\n\tTest: {test_df.shape[0]}")
    logger.info(f"Number of classes:\n\tFamily: {len(names['family'])}\n\tGenus: {len(names['genus'])}\n\tSpecies: {len(names['species'])}")
    
    pretrain_models = {
        'RN50v2': {
            'model': keras_apps.ResNet50V2(
                include_top=False, weights='imagenet', input_shape=SETUP['image']['shape']
            ),
            'preprocess': keras_apps.resnet_v2.preprocess_input
        },
        'RN101v2': {
            'model': keras_apps.ResNet101V2(
                include_top=False, weights='imagenet', input_shape=SETUP['image']['shape']
            ),
            'preprocess': keras_apps.resnet_v2.preprocess_input
        },
        'RN152v2': {
            'model': keras_apps.ResNet152V2(
                include_top=False, weights='imagenet', input_shape=SETUP['image']['shape']
            ),
            'preprocess': keras_apps.resnet_v2.preprocess_input
        },
        'EFv2b0': {
            'model': keras_apps.EfficientNetV2B0(
                include_top=False, weights='imagenet', input_shape=SETUP['image']['shape']
            ),
            'preprocess': keras_apps.efficientnet_v2.preprocess_input
        },
        'EFv2b1': {
            'model': keras_apps.EfficientNetV2B1(
                include_top=False, weights='imagenet', input_shape=SETUP['image']['shape']
            ),
            'preprocess': keras_apps.efficientnet_v2.preprocess_input
        },
        'EFv2b3': {
            'model': keras_apps.EfficientNetV2B2(
                include_top=False, weights='imagenet', input_shape=SETUP['image']['shape']
            ),
            'preprocess': keras_apps.efficientnet_v2.preprocess_input
        },
        'VGG16': {
            'model': keras_apps.VGG16(
                include_top=False, weights='imagenet', input_shape=SETUP['image']['shape']
            ),
            'preprocess': keras_apps.vgg16.preprocess_input
        },
        'VGG19': {
            'model': keras_apps.VGG19(
                include_top=False, weights='imagenet', input_shape=SETUP['image']['shape']
            ),
            'preprocess': keras_apps.vgg19.preprocess_input
        },
        'IncRNv2': {
            'model': keras_apps.InceptionResNetV2(
                include_top=False, weights='imagenet', input_shape=SETUP['image']['shape']
            ),
            'preprocess': keras_apps.inception_resnet_v2.preprocess_input
        },
        'EFv2S': {
            'model': keras_apps.EfficientNetV2S(
                include_top=False, weights='imagenet', input_shape=SETUP['image']['shape']
            ),
            'preprocess': keras_apps.efficientnet_v2.preprocess_input    
        },
    }

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.1, seed=SEED),
            layers.RandomZoom(height_factor=(0.05, 0.1), width_factor=(0.05, 0.1), seed=SEED),  # Wider zoom range
            layers.RandomContrast(0.2, seed=SEED),
            layers.RandomBrightness(0.2, seed=SEED),
            layers.RandomTranslation(0.1, 0.1, seed=SEED),
            layers.RandomCrop(SETUP['image']['size'][0], SETUP['image']['size'][1], seed=SEED),
            layers.GaussianNoise(0.1, seed=SEED),
        ],
        name='augmentation'
    )
    
    NICKNAME = f"{SETUP['pretrain']}_{SETUP['image']['size'][0]}_{SETUP['version']}"
    NICKNAME = "DataSet_WebPlus3"
    TAGS = [SETUP['pretrain']]
    
    with wandb.init(project=SETUP['project'], name=NICKNAME, config={**SETUP}, tags=TAGS) as run:
        
        logger.info(f"Dataset Size: Train {train_df.shape[0]} ({train_df.shape[0] / df.shape[0] * 100:.2f}%) - Val {val_df.shape[0]} ({val_df.shape[0] / df.shape[0] * 100:.2f}%) - Test {test_df.shape[0]} ({test_df.shape[0] / df.shape[0] * 100:.2f}%)")
        logger.info(f"Levels Amount: Family {len(names['family'])} - Genus {len(names['genus'])} - Species {len(names['species'])}")
        
        logger.info(f"Model: {SETUP['model']['function'].__name__}")
        
        model = SETUP['model']['function'](
            pretrain = pretrain_models[SETUP['pretrain']]['model'],
            preprocess = pretrain_models[SETUP['pretrain']]['preprocess'],
            augmentation = augmentation,
            #dropout_rate = SETUP['model']['args']['dropout_rate'],
            input_shape = SETUP['image']['shape'],
            outputs_size = [
                len(names['family']),
                len(names['genus']),
                len(names['species'])
            ]
        )
        
        logger.info(model.summary(show_trainable=True))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=SETUP['compile']['lr']),
            loss=SETUP['compile']['loss'],
            metrics=SETUP['compile']['metrics'],
        )
        
        os.makedirs(f"models/{NICKNAME}", exist_ok=True)
        
        if SETUP['model']['save']:
            model.save(f"models/{NICKNAME}/model_untrained.keras")
        
        wandb_logger = WandbMetricsLogger()
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=SETUP['early_stop']["monitor"],
            patience=SETUP["early_stop"]['patience'],
            restore_best_weights=SETUP['early_stop']['restore_best_weights'],
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=SETUP['reduce_lr']["monitor"],
            factor=SETUP['reduce_lr']["lr_factor"],
            patience=SETUP['reduce_lr']["lr_patience"],
            min_lr=SETUP['reduce_lr']["lr_min"],
        )
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"models/{NICKNAME}/model_trained.keras",
            monitor=SETUP['checkpoint']["monitor"], 
            save_best_only=SETUP['checkpoint']['save_best_only'] if SETUP['model']['save'] else False,  
            mode=SETUP['checkpoint']['mode'],  
            verbose=0  
        )
        
        history = model.fit(
            train_ds,
            epochs=SETUP['train']['epochs'],
            validation_data=val_ds,
            callbacks=[wandb_logger, early_stopping, reduce_lr, checkpoint, TQDMProgressBar()],
            verbose=0,
        )

        test_results = model.evaluate(test_ds, verbose=0, return_dict=True)
        
        for name, value in test_results.items():
            wandb.log({f"test/{name}": value})
            logger.info(f"Test {name}: {value:.3f}")
        
        if SETUP['finetune']:
            logger.info("Fine-tuning the model...")
            
            pretrain = model.layers[2]
            pretrain.trainable = True
            for layer in pretrain.layers[:-SETUP['fine_tune']['layers']]:
                layer.trainable = False

            logger.info(model.summary(show_trainable=True))
            
            model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=SETUP["fine_tune"]["lr"]),
                loss=SETUP['compile']['loss'],
                metrics=SETUP['compile']['metrics'],
            )
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=SETUP['early_stop']["monitor"],
                patience=SETUP['fine_tune']['patience'],
                restore_best_weights=SETUP['early_stop']['restore_best_weights'],
            )
            
            total_epochs = SETUP['train']["epochs"] + SETUP["fine_tune"]["epochs"]
            
            ftun_history = model.fit(
                train_ds,
                epochs=total_epochs,
                initial_epoch=len(history.epoch),
                validation_data=val_ds,
                callbacks=[wandb_logger, early_stopping, reduce_lr, checkpoint, TQDMProgressBar()],
                verbose=0,
            )
            
            ftun_results = model.evaluate(test_ds, verbose=0, return_dict=True)
        
            for name, value in test_results.items():
                wandb.log({f"ftun_test/{name}": value})
                logger.info(f"Fine Tuned Test {name}: {value:.3f}")
            
            if SETUP['model']['save']:
                model.save(f"models/{NICKNAME}/model_fine_tuned.keras")
            
        logger.success(f"Model {NICKNAME} trained and logged to wandb.")
        
        