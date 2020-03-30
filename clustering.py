from abc import ABC
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class Dataset(ABC):

    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.prepare_data()
        self.scale()
        self.normalize()
        self.pca()
        
    def prepare_data(self):
        pass
    
    def scale(self):
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)

    def normalize(self):
        self.data = normalize(self.data)

    def pca(self):
        pca = PCA(n_components=2)
        self.data = pca.fit_transform(self.data)
        self.data = pd.DataFrame(self.data)
        self.data.columns = ['P1', 'P2']


class ShoppersDataset(Dataset):

    def prepare_data(self):
        self.data = pd.get_dummies(self.data, columns=['Month', 'VisitorType'])
        self.data.fillna(0, inplace=True)


class ClusterDBSCAN:
    clustering = None

    def __init__(self, dataset):
        self.data = dataset.data
        self.cluster()

    def cluster(self):
        self.clustering = DBSCAN(eps=0.07, min_samples=50).fit(self.data)
        labels = self.clustering.labels_
        plt.scatter(self.data['P1'], self.data['P2'], c=labels)
        plt.show()


if __name__ == "__main__":
    datasets = {
        'shopper': ShoppersDataset('online_shoppers_intention.csv'),
    }

    ClusterDBSCAN(datasets['shopper'])
