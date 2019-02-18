import luigi
import pandas as pd

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
This tasks train the machine learning model and validates it 
"""
