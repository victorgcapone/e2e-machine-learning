import functools
import numpy as np
import typing as tg
from . import configuration as cfg
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from flask import (
    Blueprint, flash, request, url_for, jsonify
)

bp = Blueprint("ml", __name__, url_prefix="/ml")

def init_model():
    print("Loading ML model")
    with open(cfg.config["model"]["path"], 'r') as model_file:
        yaml_model = model_file.read()
        keras_model = tf.keras.models.model_from_yaml(yaml_model)
        keras_model.load_weights(cfg.config["model"]["weights"])
        keras_model._make_predict_function()
    graph = tf.get_default_graph()
    return keras_model, graph


@bp.route('/predict', methods=['POST'])
def predict() -> str:
    keras_model, graph = init_model()
    features = np.array([request.get_json()['features']])
    print(features, len(features), features.shape)
    with graph.as_default():
        prediction = keras_model.predict(features)
    print(prediction)
    return jsonify({"survival_chance": "%.2f"%prediction})
