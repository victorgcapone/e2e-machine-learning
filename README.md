# An E2E exercise in Machine Learning Engineering

## How to run
- Download the repository and cd into the root folder
- run ```pipenv install``` to install the dependencies
- run ```pipenv shell``` to hop into the virtual environment shell
- run ```python -m luigi --module src/luigi_tasks TrainModel --local-scheduler``` to train the model, this will also download and prepare your data
- run ```./start_flask.sh``` to start the flask server
- POST to localhost:5000/ml/predict a JSON like so
```
{
"features":[Age,SibSp,Parch,Fare,Pclass_2,Pclass_3,Sex_male]
}
```
- The server should respond with
```
{
"survival_chance":x.xx
}
```
