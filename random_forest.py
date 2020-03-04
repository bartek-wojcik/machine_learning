import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_validate_split(df, train_percent=.6, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:]]
    return train, validate


def get_x_y(data):
    x = data[data.columns[:-1]]
    y = data[data.columns[-1]]
    return x, y


def random_forest(data):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    train, validate = train_validate_split(data)

    train_x, train_y = get_x_y(train)
    validate_x, validate_y = get_x_y(validate)

    clf.fit(train_x, train_y)

    print(clf.score(validate_x, validate_y))


data = pd.read_csv('gender.csv')

mapped_genders = data['Gender'].map({'F': 0, 'M': 1})

data = pd.get_dummies(data, columns=data.columns[:-1])
data['Gender'] = mapped_genders

random_forest(data)

data = pd.read_csv('creditcard.csv')

random_forest(data)
