import tensorflow as tf
import keras

model = keras.models.load_model('./weights/conditional-gan/model.h5')

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('./weights/conditional-gan/tf-lite_model.tflite', 'wb') as f:
  f.write(tflite_model)