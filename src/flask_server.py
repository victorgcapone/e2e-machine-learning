from flask import Flask

app = Flask(__name__)

@app.route('/')
def index_page() -> str:
    return "Flask is up"
