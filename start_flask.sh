#!/bin/bash 

export FLASK_APP=$(pwd)/src/flask_server.py
export FLASK_APP=src
export FLASK_ENV=development
flask run
