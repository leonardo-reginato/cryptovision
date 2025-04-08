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
from itertools import product
import warnings

from cryptovision import tools
import cryptovision.dataset as dataset
from cryptovision.models import CryptoVisionModels as cv_models
from cryptovision.train import train_with_wandb
from cryptovision.fine_tuning import finetune_with_wandb

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Optionally limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def load_training_settings(settings_file_path: str) -> dict:
    with open(settings_file_path, 'r') as f:
        settings = yaml.safe_load(f)
    if settings.get('version') == 'auto':
        settings['version'] = datetime.datetime.now().strftime('%y%m.%d.%H%M')
    return settings

def main():
    logger.info('Starting grid search pipeline...')

    # Load training settings from YAML
    settings = load_training_settings('cryptovision/settings.yaml')

    if settings.get('suppress_warnings', False):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        warnings.filterwarnings('ignore')

    # Setup grid search parameters – either from the settings or defaults
    grid_params = settings.get('grid_search', {})
    arch_list = grid_params.get('architecture', ['gated', 'concat', 'std', 'att'])
    shared_dp_list = grid_params.get('shared_dropout', [0, 0.2, 0.4])
    feat_dp_list = grid_params.get('features_dropout', [0, 0.2, 0.4])
    pooling_list = grid_params.get('pooling_type', ['max', 'avg'])

    combinations = list(product(arch_list, shared_dp_list, feat_dp_list, pooling_list))

    grid_csv_path = 'grid_search_results.csv'
    if not os.path.exists(grid_csv_path):
        df_params = pd.DataFrame(combinations, columns=['architecture', 'shared_dropout', 'features_dropout', 'pooling_type'])
        df_params['status'] = 'pending'
        df_params.to_csv(grid_csv_path, index=False)
    else:
        df_params = pd.read_csv(grid_csv_path)

    # Load dataset
    data = {}
    tf_data = {}
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
            lambda x: x.replace('/Volumes/T7_shield/CryptoVision/Data/Sources', src_path)
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

    total_combinations = len(df_params)
    for idx, row in df_params.iterrows():
        if row['status'] == 'done':
            continue
        logger.info(f'Grid search combination {idx+1}/{total_combinations}: {row.to_dict()}')
        try:
            # Update settings with the current grid search parameters
            settings['architecture'] = row['architecture']
            settings['shared_dropout'] = row['shared_dropout']
            settings['features_dropout'] = row['features_dropout']
            settings['pooling_type'] = row['pooling_type']

            # Update tags with grid search information
            settings['tags'] = [
                f"Arch_{settings['architecture']}",
                f"SharedDP_{settings['shared_dropout']}",
                f"FeatDP_{settings['features_dropout']}",
                f"Pool_{settings['pooling_type']}",
                'GridSearch'
            ]

            # Build the model using the updated grid parameters
            model = cv_models.basic(
                imagenet_name=settings['pretrain'],
                augmentation=cv_models.augmentation_layer(image_size=settings['image_size']),
                input_shape=(settings['image_size'], settings['image_size'], 3),
                shared_dropout=settings['shared_dropout'],
                feat_dropout=settings['features_dropout'],
                shared_layer_neurons=settings['shared_layer_neurons'],
                pooling_type=settings['pooling_type'],
                architecture=settings['architecture'],
                output_neurons=(
                    data['test']['family'].nunique(),
                    data['test']['genus'].nunique(),
                    data['test']['species'].nunique(),
                )
            )

            # Initial training phase
            model = train_with_wandb(
                project_name=settings['project_name'],
                experiment_name=settings.get('experiment_name'),
                tags=settings['tags'],
                config=settings,
                model=model,
                datasets=tf_data,
                save=False
            )

        except Exception as e:
            logger.error(f'Error in combination {idx}: {e}')
            df_params.loc[idx, 'status'] = 'error'
            df_params.to_csv(grid_csv_path, index=False)
            continue

    logger.success('Grid search pipeline finished.')

if __name__ == '__main__':
    main()