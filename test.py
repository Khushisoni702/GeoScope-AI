import tensorflow as tf

print("TF:", tf.__version__)
print("Keras:", tf.keras.__version__)

model = tf.keras.models.load_model(
    "geoscope_fixed.keras",
    compile=False
)

print("MODEL LOADED SUCCESSFULLY")