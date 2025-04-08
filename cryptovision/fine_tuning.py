import os
import json
import yaml
import wandb
import datetime
import random
import numpy as np
import warnings
from loguru import logger
import tensorflow as tf
from wandb.integration.keras import WandbMetricsLogger

from cryptovision import tools                   

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Enable mixed precision for Apple Silicon (if needed)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Optionally limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def load_settings(settings_file_path: str) -> dict:
    """Load YAML settings and set version if 'auto'."""
    with open(settings_file_path, 'r') as f:
        settings = yaml.safe_load(f)
    if settings.get('version') == 'auto':
        settings['version'] = datetime.datetime.now().strftime('%y%m.%d.%H%M')
    return settings


def finetune_with_wandb(
    project_name: str,
    experiment_name: str,
    tags: list,
    config: dict,
    model: tf.keras.Model,
    datasets: dict,
    save: bool = True,
):
    
    # Prepare wandb initialization arguments
    wandb_init_args = {
        "project": project_name,
        "config": config,
        "tags": tags,
    }
    
    ft_cfg = config['finetune']
    
    # Log into wandb
    wandb.login()
    with wandb.init(**wandb_init_args) as run:
        if experiment_name is None:
            experiment_name = run.name
        logger.info(f"Wandb run started: {run.name} | ID: {run.id}")
        logger.info("Model summary before fine-tuning:")
        logger.info(model.summary(show_trainable=True))
        
        output_dir = os.path.join("models", project_name, f"{config['version']}")
        os.makedirs(output_dir, exist_ok=True)
            
        # Define callbacks
        wandb_cb = WandbMetricsLogger()
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor=ft_cfg['early_stop']['monitor'],
            patience=ft_cfg['early_stop']['patience'],
            restore_best_weights=ft_cfg['early_stop']['best_weights'],
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=ft_cfg['reduce_lr']['monitor'],
            factor=ft_cfg['reduce_lr']['factor'],
            patience=ft_cfg['reduce_lr']['patience'],
            min_lr=ft_cfg['reduce_lr']['min'],
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "model.weights.h5"),
            monitor=ft_cfg['checkpoint']['monitor'],
            save_best_only=ft_cfg['checkpoint']['save_best_only'],
            mode=ft_cfg['checkpoint']['mode'],
            save_weights_only=ft_cfg['checkpoint']['weights_only'],
            verbose=0
        )
        
        history = model.fit(
            datasets['train'],
            validation_data=datasets['val'],
            initial_epoch=config['epochs'],
            epochs=config['epochs'] + config['finetune']['epochs'],
            callbacks=[wandb_cb, early_stop, reduce_lr, checkpoint, tools.TQDMProgressBar()],
            verbose=0,
        )
        
        # Optional save the fine-tuned model
        if save:
            model.save_weights(os.path.join(output_dir, "finetuned.weights.h5"))
        
        # Save training history
        with open(os.path.join(output_dir, "finetune_history.json"), "w") as f:
            json.dump(history.history, f)
        
        # Evaluate on the test set and log the results
        test_results = model.evaluate(datasets['test'], verbose=0, return_dict=True)
        for metric, value in test_results.items():
            wandb.log({f"ft_test/{metric}": value})
            logger.info(f"ft_test - {metric}: {value:.3f}")
            
        logger.success("wandb fine-tuning pipeline completed successfully.")
        
    wandb.finish()
    return model

def main():
    logger.info("Starting training pipeline...")
    
    # Load settings from YAML
    settings = load_settings('cryptovision/settings.yaml')
    if settings.get('suppress_warnings', False):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        warnings.filterwarnings("ignore")
        
    data = {}
    tf_data = {}
    
    # Load dataset
    import cryptovision.dataset as dataset
    data['train'], data['val'], data['test'] = dataset.load_dataset(
        src_path=settings['data_path'],
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
    
    data['train'] = rename_image_path(data['train'], settings['data_path'])
    data['val'] = rename_image_path(data['val'], settings['data_path'])
    data['test'] = rename_image_path(data['test'], settings['data_path'])
    
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
    from cryptovision.models import CryptoVisionModels as cv_models
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
    
    # Load pre-trained weights
    model.load_weights(settings['finetune']['weights_path'])
    
    # Unfreeze the last layers for fine-tuning
    pretrain = model.layers[2]
    pretrain.trainable = True
    for layer in pretrain.layers[:-settings['finetune']['unfreeze_layers']]:
        layer.trainable = False
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=settings['finetune']['lr']),
        loss={key: settings['finetune']['loss'] for key in ['family', 'genus', 'species']},
        metrics={key: settings['finetune']['metrics'] for key in ['family', 'genus', 'species']},
        loss_weights={
            'family': settings['finetune']['loss_weights'][0],
            'genus': settings['finetune']['loss_weights'][1],
            'species': settings['finetune']['loss_weights'][2],
        },
    )
    
    # Re-do version from trained model
    settings['version'] = os.path.basename(os.path.dirname(settings['finetune']['weights_path']))
    
    # Fine-tune the model with wandb 
    finetune_with_wandb(
        project_name = settings['project_name'],
        experiment_name = settings['experiment_name'],
        tags = settings['tags'],
        config = settings,
        datasets = tf_data,
        model = model,
        save = True,
    )
    
    logger.success("Fine Tuning Script finished.")
    
if __name__ == "__main__":
    main()