import tensorflow as tf

# interpreter = tf.contrib.lite.Interpreter(model_path="tf_files/optimized_graph.lite")
# interpreter = tf.contrib.lite.Interpreter(model_path="tf_files/mobilenet_float_v2_1.0_299.tflite")
interpreter = tf.contrib.lite.Interpreter(model_path="tf_files/mobilenet_quant_v2_1.0_299.tflite")
# interpreter = tf.contrib.lite.Interpreter(model_path="tf_files/inception_v3_quant.tflite")
interpreter.allocate_tensors()

# Print input shape and type
print(interpreter.get_input_details()[0]['shape'])  # Example: [1 299 299 3]
print(interpreter.get_input_details()[0]['dtype'])  # Example: <class 'numpy.float32'>

# Print output shape and type
print(interpreter.get_output_details()[0]['shape'])  # Example: [1 1001]
print(interpreter.get_output_details()[0]['dtype'])  # Example: <class 'numpy.float32'>