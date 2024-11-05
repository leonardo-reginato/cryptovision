import streamlit as st
import tensorflow as tf

cv_model = tf.keras.models.load_model(
    "/Users/leonardo/Documents/Projects/cryptovision/cryptovision/docker/web_app/models"
)

def path_to_labes(input_path):
    
    pass