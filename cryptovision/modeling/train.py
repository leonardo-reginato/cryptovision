import os
import json
import yaml
import wandb
import random
import datetime
import numpy as np
from loguru import logger
from wandb.integration.keras import WandbMetricsLogger

import tensorflow as tf
from tensorflow.keras import callbacks                      # type: ignore

from cryptovision import tools
import cryptovision.dataset as dataset
from cryptovision.models import CryptoVisionModels as cv_models

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

def load_settings(settings_file_path: str) -> dict:
    with open(settings_file_path, 'r') as f:
        settings = yaml.load(f, Loader=yaml.SafeLoader)
    
    if settings['version'] == 'auto':
        settings['version'] = datetime.datetime.now().strftime('%y%m.%d.%H%M')
    
    return settings

def train_with_wandb(
    project_name: str,
    experiment_name: str,
    run_tags: list,
    config: dict,
    datasets: dict,
    model: tf.keras.Model,
    save_mode: bool = True,
) -> tf.keras.Model:
    """
    Unified training pipeline using Weights & Biases (wandb) for deep learning experiments.
    
    This function covers both initial training and an optional fine-tuning phase, always expecting a 
    tf.keras.Model as input. It compiles the model, trains it with callbacks, evaluates on a test set, and logs 
    results to wandb.
    
    To maximize flexibility and reusability, the optimizers are expected to be provided in the configuration:
        - 'optimizer': The optimizer for the initial training phase.
        - 'ftune_optimizer': The optimizer for the fine-tuning phase.
    
    Args:
        project_name (str): The wandb project name.
        experiment_name (str): The name for the wandb run.
        run_tags (list): List of tags to help organize the experiment.
        config (dict): Training configuration, including:
            - optimizer (tf.keras.optimizer): Optimizer for pre-training.
            - loss (str): Loss function for pre-training.
            - metrics (str): Metrics for pre-training.
            - loss_weights (list): List of loss weights.
            - early_stop_monitor (str): Metric to monitor for early stopping.
            - patience (int): Patience for early stopping.
            - reduce_lr_monitor (str): Metric to monitor for learning rate reduction.
            - lr_factor (float): Factor for reducing learning rate.
            - lr_patience (int): Patience for learning rate reduction.
            - lr_min (float): Minimum learning rate.
            - checkpoint_monitor (str): Metric to monitor for checkpointing.
            - save_best_only (bool): Whether to save only the best model.
            - checkpoint_mode (str): Mode for checkpointing.
            - epochs (int): Number of epochs for pre-training.
            - fine_tune (bool): Whether to perform fine-tuning.
            - ftune_optimizer (tf.keras.optimizer): Optimizer for fine-tuning.
            - ftune_loss (str): Loss function for fine-tuning.
            - ftune_metrics (str): Metrics for fine-tuning.
            - ftune_loss_weights (list): List of loss weights for fine-tuning.
            - ftune (dict): Contains additional fine-tuning parameters like 'patience' and 'epochs', and\n               'layers' (the number of layers in the pre-trained module to keep trainable).\n            - version (str, optional): Run version identifier.\n        datasets (dict): Dictionary containing the datasets. Expected keys: 'train', 'val', 'test'.\n        model (tf.keras.Model): The model to train.\n        save_mode (bool, optional): Whether to save the model and checkpoints to disk. Defaults to True.\n\n    Returns:\n        tf.keras.Model: The trained (and optionally fine-tuned) model.\n    """
    
    # Prepare wandb initialization arguments
    wandb_init_args = {
        "project": project_name,
        "config": config,
        "tags": run_tags,
    }
    if experiment_name is not None:
        wandb_init_args["name"] = experiment_name
    
    # Log into wandb
    wandb.login()
    with wandb.init(**wandb_init_args) as run:
        
        if experiment_name is None:
            experiment_name = run.name
        logger.info(f"Wandb run started: {experiment_name} | ID: {run.id}")
        logger.info("Model summary before training:")
        logger.info(model.summary(show_trainable=True))
        
        # Create a directory to store model artifacts
        version = config.get('version', datetime.datetime.now().strftime('%y%m.%d.%H%M'))
        output_dir = os.path.join("models", project_name, f"{run.name}_{run.id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # === Pre-Training Phase ===
        # Compile the model with initial training settings using the provided optimizer
        model.compile(
            optimizer=config['optimizer'],
            loss={key: config['loss'] for key in ['family', 'genus', 'species']},
            metrics={key: config['metrics'] for key in ['family', 'genus', 'species']},
            loss_weights={
                'family': config['loss_weights'][0],
                'genus': config['loss_weights'][1],
                'species': config['loss_weights'][2],
            },
        )
        
        # Optionally save the untrained model
        if save_mode:
            model.save(os.path.join(output_dir, "model_pretrain.keras"))
        
        # Define callbacks for training
        wandb_cb = WandbMetricsLogger()
        early_stop = callbacks.EarlyStopping(
            monitor=config['early_stop']['monitor'],
            patience=config['early_stop']['patience'],
            restore_best_weights=config['restore_best_weights'],
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor=config['reduce_lr']['monitor'],
            factor=config['reduce_lr']['factor'],
            patience=config['reduce_lr']['patience'],
            min_lr=config['reduce_lr']['min'],
        )
        checkpoint = callbacks.ModelCheckpoint(
            os.path.join(output_dir, "model_trained.keras"),
            monitor=config['checkpoint']['monitor'],
            save_best_only=config['save_best_only'] if save_mode else False,
            mode=config['checkpoint']['mode'],
            verbose=0
        )
        
        # Train the model
        history = model.fit(
            datasets['train'],
            validation_data=datasets['val'],
            epochs=config['epochs'],
            callbacks=[wandb_cb, early_stop, reduce_lr, checkpoint, tools.TQDMProgressBar()],
            verbose=0
        )
        
        # Save training history
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(history.history, f)
        
        # Evaluate on the test set and log the results
        test_results = model.evaluate(datasets['test'], verbose=0, return_dict=True)
        for metric, value in test_results.items():
            wandb.log({f"test/{metric}": value})
            logger.info(f"test/{metric}: {value:.3f}")
        
        # === Optional Fine-Tuning Phase ===
        if config.get('fine_tune', False):
            logger.info("Fine-tuning phase activated—refining the model!")
            
            # Unfreeze specific layers for fine-tuning (assumes pre-trained module is at model.layers[2])
            pretrain_layer = model.layers[2]
            pretrain_layer.trainable = True
            for layer in pretrain_layer.layers[:-config['ft']['layers']]:
                layer.trainable = False
            
            logger.info("Model summary after unfreezing layers:")
            logger.info(model.summary(show_trainable=True))
            
            # Re-compile the model with fine-tuning settings using the provided ftune_optimizer
            model.compile(
                optimizer=config['ftune_optimizer'],
                loss={key: config['ft']['loss'] for key in ['family', 'genus', 'species']},
                metrics={key: config['ft']['metrics'] for key in ['family', 'genus', 'species']},
                loss_weights={
                    'family': config['ft']['loss_weights'][0],
                    'genus': config['ft']['loss_weights'][1],
                    'species': config['ft']['loss_weights'][2],
                },
            )
            
            # Define callbacks for the fine-tuning phase
            ft_wandb_cb = WandbMetricsLogger()
            ft_checkpoint = callbacks.ModelCheckpoint(
                os.path.join(output_dir, "model_ftune.keras"),
                monitor=config['checkpoint']['monitor'],
                save_best_only=config['save_best_only'] if save_mode else False,
                mode=config['checkpoint']['mode'],
                verbose=0
            )
            ft_early_stop = callbacks.EarlyStopping(
                monitor=config['early_stop']['monitor'],
                patience=config['ft']['patience'],
                restore_best_weights=True,
            )
            
            total_epochs = config['epochs'] + config['ft']['epochs']
            ft_history = model.fit(
                datasets['train'],
                validation_data=datasets['val'],
                epochs=total_epochs,
                initial_epoch=config['epochs'],
                callbacks=[ft_wandb_cb, ft_early_stop, reduce_lr, ft_checkpoint, tools.TQDMProgressBar()],
                verbose=0
            )
            
            # Evaluate fine-tuned model and log the results
            ft_results = model.evaluate(datasets['test'], verbose=0, return_dict=True)
            for metric, value in ft_results.items():
                wandb.log({f"ft_test/{metric}": value})
                logger.info(f"ft_test/{metric}: {value:.3f}")
            
            # Save fine-tuning history
            with open(os.path.join(output_dir, "ftune_history.json"), "w") as f:
                json.dump(ft_history.history, f)
            
            logger.info("Fine-tuning phase completed successfully.")
        
        logger.success("Wandb training pipeline finished successfully.")
    
    wandb.finish()
    return model


if __name__ == '__main__':

    logger.info("Starting training...")
    
    settings = load_settings('cryptovision/settings.yaml')
    
    if settings['suppress_warnings']:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        warnings.filterwarnings("ignore")
        
    data = {}
    tf_data = {}
    
    data['train'], data['val'], data['test'] = dataset.load_dataset(
        src_path=settings['src_path'],
        min_samples=settings['samples_threshold'],
        return_split=True,
        stratify_by='species',
        test_size=settings['test_size'],
        val_size=settings['validation_size'],
        random_state=SEED
    )
    
    def rename_image_path(df, src_path):
        df['image_path'] = df['image_path'].apply(
            lambda x: x.replace(
                '/Volumes/T7_shield/CryptoVision/Data/Sources',
                src_path,
            )
        )
        return df
    
    data['train'] = rename_image_path(data['train'], settings['src_path'])
    data['val'] = rename_image_path(data['val'], settings['src_path'])
    data['test'] = rename_image_path(data['test'], settings['src_path'])
    
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
    
    
    # Deep Learning Model's Setup
    model = cv_models.basic(
        imagenet_name=settings['pretrain'],
        augmentation=cv_models.augmentation_layer(image_size=settings['image_size']),
        input_shape=(settings['image_size'], settings['image_size'], 3),
        shared_dropout=settings['shared_dropout'],
        feat_dropout=settings['features_dropout'],
        shared_layer_neurons=2048,
        pooling_type='max',
        architecture=settings['architecture'],
        output_neurons=(
            data['test']['family'].nunique(),
            data['test']['genus'].nunique(),
            data['test']['species'].nunique(),
        )
    )

    settings['optimizer'] = tf.keras.optimizers.Adam(
        learning_rate=settings['lr']
    )
    settings['ftune_optimizer'] = tf.keras.optimizers.RMSprop(
        learning_rate=settings['ft']['lr']
    )

    train_with_wandb(
        project_name = settings['project_name'],
        experiment_name = None,
        run_tags = settings['tags'],
        config = settings,
        datasets = tf_data,
        model = model,
        save_mode = True,
    )
    
    logger.success("Training Script finished.")
    