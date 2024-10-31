import wandb
import typer
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from loguru import logger
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from wandb.integration.keras import WandbMetricsLogger
from cryptovision.config import PARAMS, PROCESSED_DATA_DIR
from cryptovision.tools import ( 
    image_directory_to_pandas, 
    split_image_dataframe,  
    build_dataset_from_dataframe,
    get_taxonomic_mappings_from_folders,
    analyze_taxonomic_misclassifications,
)
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, 
    Dropout, BatchNormalization, 
    Activation, Multiply,
    Concatenate, Add, Input, Reshape
)
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2

app = typer.Typer()
wandb.require("core")
tf.keras.mixed_precision.set_global_policy('mixed_float16')

project_name = "CryptoVision - HACPL Trials"
run_sufix = "Proteon"

# Define a custom SE Block for 4D inputs
def squeeze_excite_block(input, ratio=16):
    filters = input.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(input)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)

    return Multiply()([input, se])

# Define a custom Self-Attention Layer
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.query_dense = Dense(input_shape[-1])
        self.key_dense = Dense(input_shape[-1])
        self.value_dense = Dense(input_shape[-1])
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        q = self.query_dense(inputs)
        k = self.key_dense(inputs)
        v = self.value_dense(inputs)

        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        attention_output = tf.matmul(attention_scores, v)
        return attention_output + inputs  # Residual connection

# Define a custom Hierarchical Attention Classifier
def create_hierarchical_classifier_with_attention_se(
    PARAMS,
    family_labels,
    genus_labels,
    species_labels,
    data_augmentation
):
    preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    base_model = ResNet50V2(
        input_shape=PARAMS['img_size'] + (3,),
        include_top=False,
        weights=PARAMS['model']['weights']
    )
    base_model.trainable = PARAMS['model']['trainable']

    # Define inputs and apply data augmentation
    inputs = Input(shape=PARAMS['img_size'] + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)

    # Apply SE Block to the output of base model
    x = squeeze_excite_block(x)  # SE block at feature extraction level
    x = GlobalAveragePooling2D()(x)
    x = Dropout(PARAMS['model']['dropout'])(x)

    # Shared dense layer with self-attention
    shared_dense = Dense(PARAMS['model']['shared_layer'], activation=None)(x)
    shared_dense = BatchNormalization()(shared_dense)
    shared_dense = Activation('relu')(shared_dense)
    shared_dense = SelfAttention()(shared_dense)  # Self-attention layer on 2D data
    shared_dense = Dropout(0.3)(shared_dense)

    # Family branch
    family_transform = Dense(PARAMS['model']['family_transform'], activation=None, name='family_transform')(shared_dense)
    family_transform = BatchNormalization()(family_transform)
    family_transform = Activation('relu')(family_transform)
    family_transform = Dropout(0.2)(family_transform)
    family_output = Dense(len(family_labels), activation='softmax', name='family')(family_transform)

    # Genus branch with attention
    family_attention = Dense(PARAMS['model']['family_attention'], activation='sigmoid')(family_transform)
    family_features = Multiply()([shared_dense, family_attention])
    family_features = Concatenate()([family_features, family_output])

    genus_transform = Dense(PARAMS['model']['genus_transform'], activation=None)(family_features)
    genus_transform = BatchNormalization()(genus_transform)
    genus_transform = Activation('relu')(genus_transform)
    genus_transform = Dropout(0.2)(genus_transform)

    genus_residual = Dense(PARAMS['model']['genus_residual'], activation='relu')(family_output)
    genus_hidden = Add()([genus_transform, genus_residual])
    genus_output = Dense(len(genus_labels), activation='softmax', name='genus')(genus_hidden)

    # Species branch with attention
    genus_attention = Dense(PARAMS['model']['genus_attention'], activation='sigmoid')(genus_hidden)
    genus_features = Multiply()([shared_dense, genus_attention])
    genus_features = Concatenate()([genus_features, family_output, genus_output])

    species_transform = Dense(PARAMS['model']['species_transform'], activation=None)(genus_features)
    species_transform = BatchNormalization()(species_transform)
    species_transform = Activation('relu')(species_transform)
    species_transform = Dropout(0.2)(species_transform)

    species_residual = Dense(PARAMS['model']['species_residual'], activation='relu')(Concatenate()([family_output, genus_output]))
    species_hidden = Add()([species_transform, species_residual])
    species_output = Dense(len(species_labels), activation='softmax', name='species')(species_hidden)

    # Build and compile model
    model = Model(inputs, [family_output, genus_output, species_output])

    return model

# Focal Loss function
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = -y_true * (alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred))
        return K.sum(loss, axis=1)
    return focal_loss_fixed

# Hierarchical consistency loss
def hierarchical_consistency_loss(y_true_family, y_pred_family, y_true_genus, y_pred_genus, y_true_species, y_pred_species):
    # Calculate family-level loss
    family_loss = focal_loss()(y_true_family, y_pred_family)
    
    # Genus-level loss with hierarchical penalty based on family misclassification
    genus_loss = focal_loss()(y_true_genus, y_pred_genus)
    genus_penalty = tf.reduce_mean(family_loss) * genus_loss  # Apply penalty if family is wrong
    
    # Species-level loss with penalty based on both family and genus misclassification
    species_loss = focal_loss()(y_true_species, y_pred_species)
    species_penalty = (tf.reduce_mean(family_loss) + tf.reduce_mean(genus_loss)) * species_loss
    
    # Total loss as a combination of focal loss for each level and penalties for hierarchical inconsistency
    return family_loss + genus_penalty + species_penalty

# Final custom loss function combining Focal Loss and Hierarchical Consistency Loss
def combined_hierarchical_loss(y_true_family, y_pred_family, y_true_genus, y_pred_genus, y_true_species, y_pred_species):
    family_loss = focal_loss()(y_true_family, y_pred_family)
    genus_loss = focal_loss()(y_true_genus, y_pred_genus)
    species_loss = focal_loss()(y_true_species, y_pred_species)
    
    # Hierarchical consistency loss
    hierarchy_loss = hierarchical_consistency_loss(y_true_family, y_pred_family, y_true_genus, y_pred_genus, y_true_species, y_pred_species)
    
    # Combined loss with hierarchy penalty
    return family_loss + genus_loss + species_loss + 0.1 * hierarchy_loss  # Adjust weighting as needed

# Matrics Logger with Wandb
def log_metrics(model, target_dataset, prefix,):
    
    logger.info(f"Evaluating {prefix} metrics...")
    
    # Evaluate the model
    results = model.evaluate(target_dataset, return_dict=True, verbose=0)
    
    # Log metrics to WandB
    for metric_name, metric_value in results.items():
        wandb.log({f"{prefix}_{metric_name}": metric_value})
        
        # Log metrics to console
        logger.info(f"{prefix} {metric_name}: {metric_value:.3f}")
    
    return True

@app.command()
def main(
  dataset_dir: Path = PROCESSED_DATA_DIR / "cv_images_dataset"  
):
    with wandb.init(
        project=project_name,
        name = f"{PARAMS['model']['base_model_short']} - {run_sufix}",
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
        
        logger.info(f"Train: {len(train_df)} ({len(train_df)/len(image_df) * 100:.2f} %), Val: {len(val_df)} ({len(val_df)/len(image_df) * 100:.2f} %), Test: {len(test_df)} ({len(test_df)/len(image_df) * 100:.2f} %)")
        
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
        
        model = create_hierarchical_classifier_with_attention_se(
            PARAMS, 
            family_labels,
            genus_labels,
            species_labels,
            data_augmentation
        )

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=PARAMS['model']['learning_rate']),
            loss={
                'family': lambda y_true, y_pred: combined_hierarchical_loss(y_true, y_pred, y_true, y_pred, y_true, y_pred),
                'genus': lambda y_true, y_pred: combined_hierarchical_loss(y_true, y_pred, y_true, y_pred, y_true, y_pred),
                'species': lambda y_true, y_pred: combined_hierarchical_loss(y_true, y_pred, y_true, y_pred, y_true, y_pred),
            },
            metrics = {
                        'family': PARAMS['model']['metrics'],
                        'genus': PARAMS['model']['metrics'],
                        'species': PARAMS['model']['metrics'],
                    },
            loss_weights=PARAMS['model']['loss_weights'],
        )
        
        # Log Metrics before training
        log_metrics(model, test_ds, "pre-train")
        
        wandb_logger = WandbMetricsLogger()
        
        history = model.fit(
            train_ds,
            epochs=PARAMS['model']['epochs'],
            validation_data=val_ds,
            batch_size=PARAMS['batch_size'],
            callbacks=[wandb_logger],
            verbose=PARAMS['verbose'],
        )
        
        # Log Metrics after training
        log_metrics(model, test_ds, "trained")
        
        # Fine-tuning
        base_model = model.layers[2]
        base_model.trainable = True
        for layer in base_model.layers[:-PARAMS['model']['ftun_last_layers']]:
            layer.trainable = False
            
        logger.info(f"Unfreezing the last {PARAMS['model']['ftun_last_layers']} layers")

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=PARAMS['model']['ftun_learning_rate']),
            loss={
                'family': lambda y_true, y_pred: combined_hierarchical_loss(y_true, y_pred, y_true, y_pred, y_true, y_pred),
                'genus': lambda y_true, y_pred: combined_hierarchical_loss(y_true, y_pred, y_true, y_pred, y_true, y_pred),
                'species': lambda y_true, y_pred: combined_hierarchical_loss(y_true, y_pred, y_true, y_pred, y_true, y_pred),
            },
            metrics = {
                        'family': PARAMS['model']['metrics'],
                        'genus': PARAMS['model']['metrics'],
                        'species': PARAMS['model']['metrics'],
                    },
            loss_weights=PARAMS['model']['loss_weights'],
        )
        
        total_epochs =  PARAMS['model']['epochs'] + PARAMS['model']['ftun_epochs']

        history_fine = model.fit(
            train_ds,
            epochs=total_epochs,
            initial_epoch=len(history.epoch),
            validation_data=val_ds,
            callbacks=[wandb_logger],
            verbose=PARAMS['verbose'],
        )
        
        # Log Metrics after fine tuning
        log_metrics(model, test_ds, "fine-tuned")
        
        family_labels, genus_labels, species_labels, genus_to_family, species_to_genus = get_taxonomic_mappings_from_folders(dataset_dir)
        
        analyze_taxonomic_misclassifications(
            model,
            test_ds,
            genus_labels,
            species_labels,
            genus_to_family,
            species_to_genus,
        )
        
        today = datetime.now().strftime("%y%m%d%H%M")
        
        model_path_name = f"/path/to/save/model/hacpl_{PARAMS['model']['base_model_short']}_{PARAMS['img_size'][0]}_f{int(results_ftun['family_accuracy'] *100)}_g{int(results_ftun['genus_accuracy'] *100)}_s{int(results_ftun['species_accuracy'] *100)}_{today}.keras"
        
        model.save(model_path_name)
        
        # Save the model
        wandb.log_artifact(
            model_path_name,
            name = f"hacpl_{PARAMS['model']['base_model_short']}_{PARAMS['img_size'][0]}_f{int(results_ftun['family_accuracy'] *100)}_g{int(results_ftun['genus_accuracy'] *100)}_s{int(results_ftun['species_accuracy'] *100)}_{today}",
            type='model',
        )
        
        logger.success(f"Model {PARAMS['model']['base_model']} trained and logged to wandb.")
        

if __name__ == '__main__':
    app()