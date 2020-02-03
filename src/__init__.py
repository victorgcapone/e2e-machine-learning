import os
from flask import Flask
import tensorflow as tf
from . import configuration as cfg

def create_app(test_config=None):
    app = Flask(__name__)


    @app.route("/healthcheck")
    def healthcheck() -> str:
        return "OK"

    # Registers the machine learning blueprint
    from . import ml
    app.register_blueprint(ml.bp)

    return app
