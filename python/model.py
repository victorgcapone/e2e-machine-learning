import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_dim=7, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

def _build_keras_model(**sk_params) -> tf.keras.Sequential:
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss=tf.losses.sigmoid_cross_entropy ,optimizer=optimizer, metrics=['accuracy'])
    return model

def build_keras_model(**sk_params) -> tf.keras.wrappers.scikit_learn.KerasClassifier:
    return tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=_build_keras_model, **sk_params)
