from flask import Flask
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

sess = tf.Session()

app = Flask(__name__)

@app.route('/')
def index_page() -> str:
    return "Flask is up"
