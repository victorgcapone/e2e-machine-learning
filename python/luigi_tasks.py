import luigi
import pandas as pd
import typing as tg
from model import build_keras_model
import h5py

"""
This task cleans the data and prepares it so we can train our
machine learning models
"""
class CleanData(luigi.ExternalTask):

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget('../data/clean_titanic_data.csv')


    def run(self) -> None:
        data = pd.read_csv("../datasets/titanic.csv")
        # Fill Age NaNs with mean value
        data["Age"].fillna(data["Age"].mean(), inplace=True)
        # Drop columns that we won't use
        data = data.drop(["Embarked", "Name", "Cabin", "Ticket", "PassengerId"], axis=1)
        # Drop all other NaN value
        data = data.dropna()
        # OneHotEncode this columns
        data = pd.get_dummies(data, columns=["Pclass", "Sex"], drop_first=True)
        with self.output().open('w') as out:
            data.to_csv(out, index=False)

"""
This task splits the data into training and testing data
"""
class TrainAndTestData(luigi.Task):

    def output(self) -> tg.Dict[str, luigi.LocalTarget]:
        # Not recomended, but this is just an example, ideally we would split the task in two and retur a single Target
        return {"train" : luigi.LocalTarget("../data/train_data.csv"),
                "test" : luigi.LocalTarget("../data/test_data.csv")}


    def requires(self) -> luigi.Task:
        return CleanData()


    def run(self) -> None:
        with self.input().open("r") as input_file:
            data = pd.read_csv(input_file)
        train_data = data.sample(frac=0.7)
        test_data = data.drop(train_data.index)
        with self.output()["train"].open("w") as train_out:
            train_data.to_csv(train_out, index=False)
        with self.output()["test"].open("w") as test_out:
            test_data.to_csv(test_out, index=False)



"""
This tasks train the machine learning model and validates it using the training and test data
"""
class TrainModel(luigi.Task):

    def output(self) -> luigi.LocalTarget:
        return {"model" : luigi.LocalTarget("../model/model.yaml"),
                "weights" : luigi.LocalTarget("../model/weights.h5")}


    def requires(self) -> luigi.Task:
        return TrainAndTestData()


    def run(self) -> None:
        model = build_keras_model(epochs=100, batch_size=1)
        with self.input()["train"].open('r') as train:
            training_data = pd.read_csv(train)
        x_train = training_data.drop("Survived", axis=1)
        y_train = training_data.Survived
        model.fit(x_train.values, y_train.values)
        with self.input()["test"].open('r') as test:
            test_data = pd.read_csv(test)
        x_test = test_data.drop("Survived", axis=1)
        y_test = test_data.Survived
        score = model.score(x_test.values, y_test.values)
        print(score)
        with self.output()['model'].open('w') as model_out:
            model_out.write(model.model.to_yaml())
        # Keras won't let us use a buffer as a parameter :c
        model.model.save_weights(self.output()['weights'].path)
