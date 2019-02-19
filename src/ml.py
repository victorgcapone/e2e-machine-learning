import functools
import numpy as np
import typing as tg
from . import configuration as cfg
import tensorflow as tf

from flask import (
    Blueprint, flash, request, url_for
)


bp = Blueprint("ml", __name__, url_prefix="/ml")
keras_model = None
graph = None

@bp.before_app_first_request
def load_model():
    print("Loading keras model")
    global keras_model
    global graph
    with open(cfg.config["model"]["path"], 'r') as model_file:
        yaml_model = model_file.read()
        keras_model = tf.keras.models.model_from_yaml(yaml_model)
        graph = tf.get_default_graph()
        keras_model.load_weights(cfg.config["model"]["weights"])

@bp.route('/predict', methods=['POST'])
def predict() -> str:
    global graph
    features = np.array([request.get_json()['features']])
    print(features, len(features), features.shape)
    with graph.as_default():
        prediction = keras_model.predict(features)
    print(prediction)
    return "%.2f" % prediction