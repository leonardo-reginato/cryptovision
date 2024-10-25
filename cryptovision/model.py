import logging
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

def create_transfer_learning_model(
    pre_trained_model,
    input_shape,
    num_layers,
    neurons_per_layer,
    unfreeze_layers,
    use_batchnorm=True,
    use_dropout=True,
    dropout_value=0.5,
    l1_value=0.01,
    l2_value=0.001,
    learning_rate=0.0001,
    num_classes=10,
):
    logging.info("Creating transfer learning model...")

    pre_trained = pre_trained_model(
        include_top=False, pooling="avg", input_shape=input_shape + (3,)
    )

    for layer in pre_trained.layers[-unfreeze_layers:]:
        layer.trainable = True

    inp_model = pre_trained.input
    x = pre_trained.output

    for i in range(num_layers):
        x = Dense(
            neurons_per_layer[i],
            activation="relu",
            kernel_regularizer=l1_l2(l1=l1_value, l2=l2_value),
        )(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        if use_dropout:
            x = Dropout(dropout_value)(x)

    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp_model, outputs=output)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    logging.info("Model created and compiled.")
    return model