import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from abc import ABC, abstractmethod


class Dataset(ABC):
    data = None
    X = None
    y = None

    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.prepare_data()
        self.split()

    def split(self):
        self.data = train_test_split(self.X, self.y, test_size=0.2)

    @abstractmethod
    def prepare_data(self):
        pass


class ShoppersDataset(Dataset):

    def prepare_data(self):
        self.df = pd.get_dummies(self.df, columns=['Month', 'VisitorType'])
        self.df.fillna(0, inplace=True)
        self.y = self.df['Revenue']
        self.X = self.df.drop('Revenue', axis=1)


class HeartDataset(Dataset):

    def prepare_data(self):
        self.y = self.df['target']
        self.X = self.df.drop('target', axis=1)


class CreditCardDataset(Dataset):

    def prepare_data(self):
        self.y = self.df['Class']
        self.X = self.df.drop(['Time', 'Class'], axis=1)


class Classifier:
    score = None

    def __init__(self, data, model):
        x_train, x_test, y_train, y_test = data
        model.fit(x_train, y_train)
        self.score = model.score(x_test, y_test)


datasets = {
    'shopper': ShoppersDataset('online_shoppers_intention.csv'),
    'heart': HeartDataset('heart.csv'),
    'credit card': CreditCardDataset('creditcard.csv'),
}

classifiers = {
    'decision tree': DecisionTreeClassifier(),
    'bayes': GaussianNB(),
    'vectors': SVC(),
    'knn': KNeighborsClassifier(),
    'random forest': RandomForestClassifier(),
}

for dataset_name, dataset in datasets.items():
    for classifier_name, classifier in classifiers.items():
        result = Classifier(dataset.data, classifier)
        print('Dataset: {} Classifier: {} Score: {}'.format(
            dataset_name, classifier_name, result.score
        ))
