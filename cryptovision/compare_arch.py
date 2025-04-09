import os
import yaml
import tensorflow as tf
from loguru import logger
from cryptovision import tools
from cryptovision.models import CryptoVisionModels as cv_models
from cryptovision.train import train_with_wandb
from cryptovision.dataset import load_dataset
import datetime
import random
import numpy as np

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

def load_and_update_settings(arch):
    with open('cryptovision/settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)
    
    settings['architecture'] = arch
    settings['tags'] = [f'Arch_{arch.upper()}', 'ArchCompare']
    if settings.get('version') == 'auto':
        settings['version'] = datetime.datetime.now().strftime('%y%m.%d.%H%M')
    return settings

def prepare_data(settings):
    train_df, val_df, test_df = load_dataset(
        src_path=settings['data_path'],
        min_samples=settings['samples_threshold'],
        return_split=True,
        stratify_by='species',
        test_size=settings['test_size'],
        val_size=settings['validation_size'],
        random_state=settings['seed']
    )

    def update_path(df):
        df['image_path'] = df['image_path'].apply(
            lambda x: x.replace('/Volumes/T7_shield/CryptoVision/Data/Sources', settings['data_path'])
        )
        return df

    train_df, val_df, test_df = map(update_path, [train_df, val_df, test_df])

    tf_data = {
        'train': tools.tensorflow_dataset(train_df, settings['batch_size'], (settings['image_size'], settings['image_size']), shuffle=False),
        'val': tools.tensorflow_dataset(val_df, settings['batch_size'], (settings['image_size'], settings['image_size']), shuffle=False),
        'test': tools.tensorflow_dataset(test_df, settings['batch_size'], (settings['image_size'], settings['image_size']), shuffle=False),
    }
    return tf_data, test_df


if __name__ == "__main__":
    
    logger.info("Starting architecture comparison...")
    
    architectures = ['std', 'att', 'gated', 'concat']

    for arch in architectures:
        logger.success(f"🔧 Running experiment for architecture: {arch.upper()}")

        config = load_and_update_settings(arch)
        tf_data, test_df = prepare_data(config)

        model = cv_models.basic(
            imagenet_name=config['pretrain'],
            augmentation=cv_models.augmentation_layer(image_size=config['image_size']),
            input_shape=(config['image_size'], config['image_size'], 3),
            shared_dropout=config['shared_dropout'],
            feat_dropout=config['features_dropout'],
            shared_layer_neurons=config['shared_layer_neurons'],
            pooling_type=config['pooling_type'],
            architecture=config['architecture'],
            output_neurons=(
                test_df['family'].nunique(),
                test_df['genus'].nunique(),
                test_df['species'].nunique()
            )
        )
        
        print(model.summary())
        
        #model.compile(
        #    optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']),
        #    loss={key: config['loss'] for key in ['family', 'genus', 'species']},
        #    metrics={key: config['metrics'] for key in ['family', 'genus', 'species']},
        #    loss_weights={
        #        'family': config['loss_weights'][0],
        #        'genus': config['loss_weights'][1],
        #        'species': config['loss_weights'][2],
        #    },
        #)
#
        #trained_model = train_with_wandb(
        #    project_name=config['project_name'],
        #    experiment_name=f"Compare_{arch}",
        #    tags=config['tags'],
        #    config=config,
        #    model=model,
        #    datasets=tf_data,
        #    save=True
        #)

    logger.success("✅ All architecture runs completed.")
