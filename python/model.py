import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_dim=10, activation=tf.nn.relu),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

def _build_keras_model(**sk_params):
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(loss=tf.losses.softmax_cross_entropy, optimizer=optimizer, metrics=['accuracy'])
    return model

def build_keras_model(**sk_params):
    return tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=_build_keras_model, **sk_params)
