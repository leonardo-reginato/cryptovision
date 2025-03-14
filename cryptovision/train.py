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
from cryptovision import tools

from tensorflow.keras import Layer, layers, backend                 # type: ignore
from tensorflow.keras import applications as keras_apps             # type: ignore
from tensorflow.keras.saving import register_keras_serializable     # type: ignore

from cryptovision import models
import cryptovision.dataset as dataset

import warnings

SEED = 42

# Python random seed
random.seed(SEED)

# NumPy random seed
np.random.seed(SEED)

# TensorFlow random seed
tf.random.set_seed(SEED)

# Set environment variable for deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Set TensorFlow logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# Enable mixed precision for Apple Silicon
tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
wandb.require("core")
    
if __name__ == '__main__':
    
    SETUP = {
        "seed": SEED,
        "verbose": 0,
        "version": f"v{datetime.datetime.now().strftime('%y%m.%d.%H%M')}",
        "project": 'BEM - Results',
        "pretrain": "RN50v2",
        "finetune": True,
        
        "model": {
            "function": models.basic,
            "args": {
                "dropout_rate": 0.4,
                "shared_units": 1024,
            },
            "save": True,
        },
        
        "image": {
            "size": (128, 128),
            "shape": (128, 128, 3),
        },
       
        "dataset": {
            "version": "v2.0.0",
            "test_size": .15,
            "validation_size": .15,
            "batch_size": 32,
            "class_samples_threshold": 90,
            "stratify_by": 'folder_label',
            "sources": ['fish_functions_v02','web', 'inaturalist_v03',],
        },
        
        "compile": {
            "lr": 1e-4,
            "loss": {
                "family": tf.keras.losses.CategoricalFocalCrossentropy(label_smoothing=0.1),
                "genus": tf.keras.losses.CategoricalFocalCrossentropy(label_smoothing=0.1),
                "species": tf.keras.losses.CategoricalFocalCrossentropy(label_smoothing=0.1),
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
            "patience": 4,
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
            "epochs": 10,
            "layers": 15,
            "lr": 1e-5,
            "pretrain_layer": "resnet50v2",
            "patience": 4
        },
    }
    
    df = dataset.main(min_samples=SETUP['dataset']['class_samples_threshold'])
    
    train_df, val_df, test_df = tools.split_dataframe(
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

    train_ds = tools.tensorflow_dataset(
        train_df, 
        batch_size=SETUP['dataset']['batch_size'], 
        image_size=SETUP['image']['size'],
    )
    
    val_ds = tools.tensorflow_dataset(
        val_df, 
        batch_size=SETUP['dataset']['batch_size'], 
        image_size=SETUP['image']['size'],
    )
    
    test_ds = tools.tensorflow_dataset(
        test_df, 
        batch_size=SETUP['dataset']['batch_size'], 
        image_size=SETUP['image']['size'],
    )
    
    train_ds = train_ds.cache().shuffle(buffer_size=1000, seed=SEED).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
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
    
    NICKNAME = f"{SETUP['model']['function'].__name__}_{SETUP['pretrain']}_{SETUP['image']['size'][0]}_{SETUP['version']}"
    TAGS = [SETUP['pretrain']]
           
    with wandb.init(project=SETUP['project'], name=NICKNAME, config={**SETUP}, tags=TAGS) as run:
        
        logger.info(f"Dataset Size: Train {train_df.shape[0]} ({train_df.shape[0] / df.shape[0] * 100:.2f}%) - Val {val_df.shape[0]} ({val_df.shape[0] / df.shape[0] * 100:.2f}%) - Test {test_df.shape[0]} ({test_df.shape[0] / df.shape[0] * 100:.2f}%)")
        logger.info(f"Levels Amount: Family {len(names['family'])} - Genus {len(names['genus'])} - Species {len(names['species'])}")
        
        logger.info(f"Model: {SETUP['model']['function'].__name__}")
        
        model = SETUP['model']['function'](
            pretrain = pretrain_models[SETUP['pretrain']]['model'],
            preprocess = pretrain_models[SETUP['pretrain']]['preprocess'],
            augmentation = augmentation,
            input_shape = SETUP['image']['shape'],
            shared_units = SETUP['model']['args']['shared_units'],
            output_units = [
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
            callbacks=[wandb_logger, early_stopping, reduce_lr, checkpoint, tools.TQDMProgressBar()],
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
                callbacks=[wandb_logger, early_stopping, reduce_lr, checkpoint, tools.TQDMProgressBar()],
                verbose=0,
            )
            
            ftun_results = model.evaluate(test_ds, verbose=0, return_dict=True)
        
            for name, value in ftun_results.items():
                wandb.log({f"ftun_test/{name}": value})
                logger.info(f"Fine Tuned Test {name}: {value:.3f}")
            
            if SETUP['model']['save']:
                model.save(f"models/{NICKNAME}/model_fine_tuned.keras")
            
        logger.success(f"Model {NICKNAME} trained and logged to wandb.")
        
        