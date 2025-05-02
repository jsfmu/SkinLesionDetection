import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('skin_lesion_model.h5')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert the model
tflite_model = converter.convert()

# Save the model
with open('skin_lesion_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite format successfully!") 