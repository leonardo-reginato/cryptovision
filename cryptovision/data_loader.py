import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def get_image_data_generators(df, img_gen_params, model_target, image_shape, batch_size):
    img_generator = ImageDataGenerator(**img_gen_params)

    X_gen_train = img_generator.flow_from_dataframe(
        dataframe=df,
        x_col="path",
        y_col=model_target,
        target_size=image_shape,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        subset="training",
    )

    X_gen_valid = img_generator.flow_from_dataframe(
        dataframe=df,
        x_col="path",
        y_col=model_target,
        target_size=image_shape,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        subset="validation",
    )

    return X_gen_train, X_gen_valid