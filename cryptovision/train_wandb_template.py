import wandb
import typer
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from loguru import logger
from wandb.integration.keras import WandbMetricsLogger
from cryptovision.config import PARAMS, MODELS_DIR, PROCESSED_DATA_DIR
from cryptovision.tools import ( 
    image_directory_to_pandas, 
    split_image_dataframe,  
    build_dataset_from_dataframe,
    plot_training_history,
    get_taxonomic_mappings_from_folders,
    analyze_taxonomic_misclassifications,
)

app = typer.Typer()
wandb.require("core")
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def create_model(PARAMS, base_model, preprocess, augmentation, family_labels, genus_labels, species_labels):
    # Model Building
        base_model = base_model(
            input_shape=PARAMS['img_size'] + (3,),
            include_top=False,
            weights=PARAMS['model']['weights'],
        )
        
        base_model.trainable = False
        
        # Define the inputs and apply augmentation
        inputs = tf.keras.Input(shape=PARAMS['img_size'] + (3,))
        x = augmentation(inputs)
        x = preprocess(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(PARAMS['model']['dropout'])(x)

        # Shared dense layer for better feature learning
        shared_layer = tf.keras.layers.Dense(PARAMS['model']['shared_layer'], activation='relu', name='shared_layer')(x)

        # Define family output
        family_output = tf.keras.layers.Dense(len(family_labels), activation='softmax', name='family')(shared_layer)

        # Concatenate the family output with the base model output
        family_features = tf.keras.layers.Concatenate()([shared_layer, family_output])

        # Define genus output, using family features as additional input
        genus_hidden = tf.keras.layers.Dense(PARAMS['model']['genus_hidden'], activation='relu')(family_features)
        genus_output = tf.keras.layers.Dense(len(genus_labels), activation='softmax', name='genus')(genus_hidden)

        # Concatenate the family and genus outputs with the base model output
        genus_features = tf.keras.layers.Concatenate()([shared_layer, family_output, genus_output])

        # Define species output, using both family and genus features as additional input
        species_hidden = tf.keras.layers.Dense(PARAMS['model']['species_hidden'], activation='relu')(genus_features)
        species_output = tf.keras.layers.Dense(len(species_labels), activation='softmax', name='species')(species_hidden)

        # Create the hierarchical model
        model = tf.keras.Model(inputs, [family_output, genus_output, species_output])
        
        return model  


@app.command()
def main(
  dataset_dir: Path = PROCESSED_DATA_DIR / "cv_images_dataset"  
):
    with wandb.init(
        project="CryptoVision - HACPL Trails",
        name = f"Run-{PARAMS['model']['base_model']}-BaseModel",
        config={
            **PARAMS,
        }
    ) as run:
        
        # Dataset Setup
        image_df = image_directory_to_pandas(dataset_dir)
        train_df, val_df, test_df = split_image_dataframe(
            image_df, 
            test_size=PARAMS['test_size'], 
            val_size=PARAMS['val_size'],
            random_state=PARAMS['random_state'],
            stratify_by=PARAMS['stratify_by'],
        )
        
        # Datasets Building
        train_ds, family_labels, genus_labels, species_labels = build_dataset_from_dataframe(
            train_df, 
            PARAMS['batch_size'], 
            PARAMS['img_size'])  
        val_ds, _, _, _ = build_dataset_from_dataframe(
            val_df, 
            PARAMS['batch_size'], 
            PARAMS['img_size'])
        test_ds, _, _, _ = build_dataset_from_dataframe(
            test_df, 
            PARAMS['batch_size'], 
            PARAMS['img_size'])
        
        # Tensorflow Autotune
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        # Data Augmentation
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip(PARAMS['data_aug']['flip']),
                tf.keras.layers.RandomRotation(PARAMS['data_aug']['rotation']),
                tf.keras.layers.RandomZoom(PARAMS['data_aug']['zoom']),
                tf.keras.layers.RandomTranslation(0.1, 0.1),
                tf.keras.layers.RandomContrast(PARAMS['data_aug']['contrast']),
                tf.keras.layers.RandomBrightness(PARAMS['data_aug']['brightness']),
            ]
        )
        
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        
        model = create_model(
            PARAMS, 
            tf.keras.applications.ResNet50V2, 
            preprocess_input, 
            data_augmentation, 
            family_labels,
            genus_labels,
            species_labels
        )

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=PARAMS['model']['learning_rate']),
            loss = PARAMS['model']['loss'],
            metrics = {
                'family': PARAMS['model']['metrics'],
                'genus': PARAMS['model']['metrics'],
                'species': PARAMS['model']['metrics'],
            },
        )
        
        results_before_training = model.evaluate(test_ds)
        
        wandb.log({
            'family initial accuracy': results_before_training[4],
            'genus initial accuracy': results_before_training[5],
            'species initial accuracy': results_before_training[6],
        })

        logger.info(f"Initial Family Acc: {results_before_training[4]:.3f}")
        logger.info(f"Initial Genus Acc: {results_before_training[5]:.3f}")
        logger.info(f"Initial Species Acc: {results_before_training[6]:.3f}")
        
        wandb_logger = WandbMetricsLogger()
        
        history = model.fit(
            train_ds,
            epochs=PARAMS['model']['epochs'],
            validation_data=val_ds,
            batch_size=PARAMS['batch_size'],
            callbacks=[
                wandb_logger
            ],
            verbose=PARAMS['verbose'],
        )
        
        results = model.evaluate(test_ds)
        
        wandb.log({
            'family accuracy': results[4],
            'genus accuracy': results[5],
            'species accuracy': results[6],
        })
        
        logger.info(f"First Train Family Acc: {results[4]:.3f}")
        logger.info(f"First Train Genus Acc: {results[5]:.3f}")
        logger.info(f"First Train Species Acc: {results[6]:.3f}")
        
        # Fine-tuning
        base_model = model.layers[2]
        base_model.trainable = True
        for layer in base_model.layers[:-PARAMS['model']['ftun_last_layers']]:
            layer.trainable = False
            
        logger.info(f"Unfreezing the last {PARAMS['model']['ftun_last_layers']} layers")

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=PARAMS['model']['ftun_learning_rate']),
            loss=PARAMS['model']['loss'],
            metrics = {
                'family': PARAMS['model']['metrics'],
                'genus': PARAMS['model']['metrics'],
                'species': PARAMS['model']['metrics'],
            },
        )
        
        total_epochs =  PARAMS['model']['epochs'] + PARAMS['model']['ftun_epochs']

        history_fine = model.fit(
            train_ds,
            epochs=total_epochs,
            initial_epoch=len(history.epoch),
            validation_data=val_ds,
            callbacks=[
                wandb_logger
            ],
            verbose=PARAMS['verbose'],
        )
        
        results_ftun = model.evaluate(test_ds)
        
        wandb.log({
            'family final accuracy': results_ftun[4],
            'genus final accuracy': results_ftun[5],
            'species final accuracy': results_ftun[6],
        })

        logger.info(f"Fine Tune Family Acc: {results_ftun[4]:.3f}")
        logger.info(f"Fine Tune Genus Acc: {results_ftun[5]:.3f}")
        logger.info(f"Fine Tune Species Acc: {results_ftun[6]:.3f}")
        
        
        family_labels, genus_labels, species_labels, genus_to_family, species_to_genus = get_taxonomic_mappings_from_folders(dataset_dir)
        
        analyze_taxonomic_misclassifications(
            model,
            test_ds,
            genus_labels,
            species_labels,
            genus_to_family,
            species_to_genus,
        )
        
        # Save the model
        #wandb.log_artifact(
        #    f"CV_{PARAMS['model']['base_model']}_model.keras",
        #    name=PARAMS['model']['base_model'],
        #    type='model',
        #    aliases=[f'ACC{int(results_ftun[4]*10_000)}','ResNet50V2', 'BaseModel']
        #)
        
        logger.success(f"Model {PARAMS['model']['base_model']} trained and logged to wandb.")
        

        
if __name__ == '__main__':
    app()