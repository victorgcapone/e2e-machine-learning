import luigi
import pandas as pd
import typing as tg
import tensorflow as tf

"""
This task cleans the data and prepares it so we can train our
machine learning models
"""
class CleanData(luigi.ExternalTask):

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget('data/clean_titanic_data.csv')


    def run(self) -> None:
        data = pd.read_csv("../datasets/titanic.tsv", sep="\t")
        # Fill Age NaNs with mean value
        data["Age"].fillna(data["Age"].mean(), inplace=True)
        # Drop all other NaN value
        data = data.dropna()
        # Drop columns that we won't use
        data = data.drop(["Name", "Cabin", "Ticket", "PassengerId"], axis=1)
        # OneHotEncode this columns
        data = pd.get_dummies(data, columns=["Embarked", "Sex"])
        with self.output().open('w') as out:
            data.to_csv(out)

"""
This task splits the data into training and testing data
"""
class TrainAndTestData(luigi.Task):

    def output(self) -> tg.Dict[str, luigi.LocalTarget]:
        # Not recomended, but this is just an example, ideally we would split the task in two and retur a single Target
        return {"train" : luigi.LocalTarget("data/train_data.csv"),
                "test" : luigi.LocalTarget("data/test_data.csv")}


    def requires(self) -> luigi.Task:
        return CleanData()


    def run(self) -> None:
        with self.input().open("r") as input_file:
            data = pd.read_csv(input_file)
        train_data = data.sample(frac=0.7)
        test_data = data.drop(train_data.index)
        with self.output()["train"].open("w") as train_out:
            train_data.to_csv(train_out)
        with self.output()["test"].open("w") as test_out:
            test_data.to_csv(test_out)
"""
This tasks train the machine learning model and validates it using the training and test data
"""
class TrainModel(luigi.Task):

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget("model/ml_model")


    def requires(self) -> luigi.Task:
        return TrainAndTestData()


    def run(self) -> None:
        model = tf.keras.Sequential([
            tf.layers.Dense(64, activation='relu', input_dims=11),
            tf.layers.Dense(64, activation='relu'),
            tf.layers.Dense(2, activation='softmax')
        ])
