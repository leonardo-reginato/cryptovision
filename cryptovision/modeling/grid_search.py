import os
import json
import yaml
import wandb
import random
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from wandb.integration.keras import WandbMetricsLogger
from cryptovision import tools

from tensorflow.keras import layers                         # type: ignore
from tensorflow.keras import applications as keras_apps     # type: ignore

from cryptovision.models import CryptoVisionModels as cv_models
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

# Enable mixed precision for Apple Silicon
tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
wandb.require("core")

def load_training_settings(settings_file_path: str) -> dict:
    with open(settings_file_path, 'r') as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)
        
    cvision_models = {
        'basic': cv_models.basic,
        #'basicV2': cv_models.basicV2,
        #'hierar': cv_models.hierarchical,
        #'att_hierar': cv_models.attention_hierarchical,
    }
    
    settings['cvision_model'] = cvision_models[settings['cvision_model']]
    
    if settings['version'] == 'auto':
        settings['version'] = datetime.datetime.now().strftime('%y%m.%d.%H%M')
    
    return settings

def create_augmentation_layer (
    seed:int=42, zoom_factor:tuple[int]=(0.05, 0.1), 
    contrast:float=0.2, brightness:float=0.2, rotation:float=0.1, 
    translation:float=0.1, image_size:int=224, gauss_noise:float=0.2,
):
    
    augmentation_layer = tf.keras.Sequential(
        [
           layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(rotation, seed=seed),
            layers.RandomZoom(height_factor=zoom_factor, width_factor=zoom_factor, seed=seed),  # Wider zoom range
            layers.RandomContrast(contrast, seed=seed),
            layers.RandomBrightness(brightness, seed=seed),
            layers.RandomTranslation(translation, translation, seed=seed),
            layers.RandomCrop(image_size, image_size, seed=seed),
            layers.GaussianNoise(gauss_noise, seed=seed), 
        ]
    )
    
    return augmentation_layer

def wandb_training_pipeline(project, name, tags, model, settings, data, save_mode=True):
    
    with wandb.init(project=project, name=name, config=settings, tags=tags) as run:
        
        logger.info(f"Wandb run started:{run.name} | ID:{run.id}")
        
        logger.info(model.summary(show_trainable=True))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=settings['lr']),
            loss={
                'family': settings['loss'],
                'genus': settings['loss'],
                'species': settings['loss'],
            },
            metrics={
                'family': settings['metrics'],
                'genus': settings['metrics'],
                'species': settings['metrics'],
            },
            loss_weights={
                'family': settings['loss_weights'][0],
                'genus': settings['loss_weights'][1],
                'species': settings['loss_weights'][2],
            },
        )
        
        new_model_path = f"models/{project}/{run.name}_{settings['version']}"
        os.makedirs(new_model_path, exist_ok=True)
        
        model.save(f"{new_model_path}/model_pretrain.keras") if save_mode else None
        
        wandb_callback = WandbMetricsLogger()
        
        #logger.info("Calculating pre-training metrics...")
        #pre_train_metrics = model.evaluate(data['test'], verbose=0, return_dict=True)
        #for metric, value in pre_train_metrics.items():
        #    wandb.log({f"pre_train_test/{metric}": value})
        #    logger.info(f"pre_train_test/{metric}: {value:.3f}")
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=settings['early_stop_monitor'],
            patience=settings['patience'],
            restore_best_weights=True,
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=settings['reduce_lr_monitor'],
            factor=settings['lr_factor'],
            patience=settings['lr_patience'],
            min_lr=settings['lr_min'],
        )
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"{new_model_path}/model_trained.keras",
            monitor=settings['checkpoint_monitor'],
            save_best_only=settings['save_best_only'] if save_mode else False,
            mode=settings['checkpoint_mode'],
            verbose=0
        )
        
        history = model.fit(
            data['train'],
            validation_data=data['val'],
            epochs=settings['epochs'],
            callbacks=[
                wandb_callback, 
                early_stopping,
                #taxon_alignment,
                reduce_lr, 
                checkpoint, 
                tools.TQDMProgressBar()
            ],
            verbose=0
        )
        
        # save history as json
        with open(f"{new_model_path}/history.json", "w") as f:
            json.dump(history.history, f)
        
        logger.info("Calculating post-training metrics...")
        post_training_metrics = model.evaluate(data['test'], verbose=0, return_dict=True)
        for name, value in post_training_metrics.items():
            wandb.log({f"post_train_test/{name}": value})
            logger.info(f"post_train_test/{name}: {value:.3f}")
            
        if settings['fine_tune']:

            logger.info("Fine-tuning model...")
            
            pretrain = model.layers[2]
            pretrain.trainable = True
            for layer in pretrain.layers[:-settings['ftune']['layers']]:
                layer.trainable = False
                
            logger.info(model.summary(show_trainable=True))
            
            model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=settings['ftune']['lr']),
                loss={
                    'family': settings['ftune']['loss'],
                    'genus': settings['ftune']['loss'],
                    'species': settings['ftune']['loss'],
                },
                metrics={
                    'family': settings['ftune']['metrics'],
                    'genus': settings['ftune']['metrics'],
                    'species': settings['ftune']['metrics'],
                },
                loss_weights={
                    'family': settings['ftune']['loss_weights'][0],
                    'genus': settings['ftune']['loss_weights'][1],
                    'species': settings['ftune']['loss_weights'][2],
                },
            )

            wandb_callback = WandbMetricsLogger()

            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                f"{new_model_path}/model_ftune.keras",
                monitor=settings['checkpoint_monitor'],
                save_best_only=settings['save_best_only'],
                mode=settings['checkpoint_mode'],
                verbose=0
            )
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=settings['early_stop_monitor'],
                patience=settings['ftune']['patience'],
                restore_best_weights=True,
            )
            
            total_epochs = settings['ftune']['epochs'] + settings['epochs']

            ftun_history = model.fit(
                data['train'],
                validation_data=data['val'],
                epochs=total_epochs,
                initial_epoch=settings['epochs'],
                callbacks=[wandb_callback, early_stopping, reduce_lr, checkpoint, tools.TQDMProgressBar()],
                verbose=0
            )
            
            logger.info("Calculating fine-tuning metrics...")
            ftun_metrics = model.evaluate(data['test'], verbose=0, return_dict=True)
            for name, value in ftun_metrics.items():
                wandb.log({f"ftun_test/{name}": value})
                logger.info(f"ftun_test/{name}: {value:.3f}")
                
            # save history as json
            with open(f"{new_model_path}/ftun_history.json", "w") as f:
                json.dump(ftun_history.history, f)

            logger.success("Fine-tuning finished.")
        
        logger.success("Wandb Training finished.")
        
        
    wandb.finish()


if __name__ == '__main__':

    logger.info("Starting training...")
    
    settings = load_training_settings('/Users/leonardo/Documents/Projects/cryptovision/cryptovision/settings.yaml')
    
    if settings['suppress_warnings']:
        # Set TensorFlow logging level to suppress warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        warnings.filterwarnings("ignore")
          
    # Set grid search parameters
    param_grid = settings['grid_search']
    
    grid_csv_path = 'grid_search_results.csv'
    
    if not os.path.exists(grid_csv_path):
        from itertools import product
        
        combinations = list(product(
            param_grid['features_dropout'],
            param_grid['shared_dropout'],
            param_grid['shared_layer_neurons'],
            param_grid['pooling_type'],
            param_grid['concat'],
            param_grid['loss_weights'],
        ))
        
        df_params = pd.DataFrame(combinations, columns=['features_dropout', 'shared_dropout', 'shared_layer_neurons','pooling_type', 'concat', 'loss_weights'])
        df_params['status'] = 'pending'
        df_params.to_csv(grid_csv_path, index=False)
    else:
        df_params = pd.read_csv(grid_csv_path)
    
    data = {}
    tf_data = {}
    
    data['train'], data['val'], data['test'] = dataset.main(
        min_samples=settings['samples_threshold'],
        return_split=True,
        stratify_by='species',
        test_size=settings['test_size'],
        val_size=settings['validation_size'],
        random_state=SEED
    )
    
    tf_data['train'] = tools.tensorflow_dataset(
        data['train'],
        batch_size=settings['batch_size'],
        image_size=(settings['image_size'], settings['image_size']),
        shuffle=False,
    )
    
    tf_data['val'] = tools.tensorflow_dataset(
        data['val'],
        batch_size=settings['batch_size'],
        image_size=(settings['image_size'], settings['image_size']),
        shuffle=False,
    )
    
    tf_data['test'] = tools.tensorflow_dataset(
        data['test'],
        batch_size=settings['batch_size'],
        image_size=(settings['image_size'], settings['image_size']),
        shuffle=False,
    )
    
    settings['families'] = sorted(data['test']['family'].unique().tolist())
    settings['genera'] = sorted(data['test']['genus'].unique().tolist())
    settings['species'] = sorted(data['test']['species'].unique().tolist())
    
    _, _, _, settings['genus_to_family'], settings['species_to_genus'] = tools.get_taxonomic_mappings_from_dataframe(data['test'])
    
    augmentation_layer = create_augmentation_layer(image_size=settings['image_size'])
    
    # Grid Search Loop
    for idx, row, in df_params.iterrows():
        
        if row['status'] == 'done':
            continue
        
        try:
            logger.info(f"Running Combination: {idx+1}/{len(df_params)}")
            logger.info(f"Combination: \n{row}")
            
            # Update settings with grid search parameters
            settings['features_dropout'] = row['features_dropout']
            settings['shared_dropout'] = row['shared_dropout']
            settings['shared_layer_neurons'] = row['shared_layer_neurons']
            settings['pooling_type'] = row['pooling_type']
            settings['concat'] = row['concat']
            settings['loss_weights'] = row['loss_weights']
            
            # Update settings tags with grid search parameters
            settings['tags'] = [
                f"FEATDP_{settings['features_dropout']}",
                f"SHRDP_{settings['shared_dropout']}",
                f"SHRNEU_{settings['shared_layer_neurons']}",
                f"POOL_{settings['pooling_type']}",
                f"CONCAT_{settings['concat']}",
                f"LOSSW_{settings['loss_weights']}",
            ]
            
            # Train model
            model = settings['cvision_model'](
                imagenet_name=settings['pretrain'],
                augmentation=augmentation_layer,
                input_shape=(settings['image_size'], settings['image_size'], 3),
                shared_dropout=settings['shared_dropout'],
                feat_dropout=settings['features_dropout'],
                shared_layer_neurons=settings['shared_layer_neurons'],
                pooling_type=settings['pooling_type'],
                concatenate=settings['concat'],
                output_neurons=(
                    data['test']['family'].nunique(),
                    data['test']['genus'].nunique(),
                    data['test']['species'].nunique(),
                )
            )
            
            logger.info(model.summary(show_trainable=True))
            
            #wandb_training_pipeline(
            #    project=settings['project'],
            #    name=settings['name'],
            #    model=model,
            #    tags=settings['tags'],
            #    settings=settings,
            #    data=tf_data,
            #    save_mode=False
            #)
            
            # Create a fake loop with tqdm to show progress
            import time
            from tqdm import tqdm
            
            for i in tqdm(range(100), desc="Fake Progress"):
                time.sleep(0.01)
            
            df_params.loc[idx, 'status'] = 'done'
            df_params.to_csv(grid_csv_path, index=False)
        except Exception as e:
            logger.error(f"Error: {e}")
            df_params.loc[idx, 'status'] = 'error'
            df_params.to_csv(grid_csv_path, index=False)
            continue
        
        logger.success("GridSearch Script finished.")