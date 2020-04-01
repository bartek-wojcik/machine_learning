from abc import ABC
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np


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


class HeartDataset(Dataset):
    pass


class CreditCardDataset(Dataset):
    pass


class ClusterDBSCAN:
    clustering = None

    def __init__(self, dataset, eps, min_samples, name):
        self.name = name
        self.data = dataset.data.copy()
        self.cluster(eps, min_samples)

    def cluster(self, eps, min_samples):
        self.clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(self.data)
        groups = self.clustering.labels_
        self.data['COLORS'] = groups
        for group in np.unique(groups):
            label = 'Cluster {}'. format(group) if group != -1 else 'Noise points'
            filtered_group = self.data[self.data['COLORS'] == group]
            plt.scatter(filtered_group['P1'], filtered_group['P2'], label=label)
        n_clusters_ = len(set(groups)) - (1 if -1 in groups else 0)
        n_noise_ = list(groups).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        plt.title(self.name + ' dataset')
        plt.figtext(0.05, 0.8, 'Eps: {}\nMin_samples: {}\nNumber of clusters: {}\nNoise points: {}'.format(
            eps, min_samples, n_clusters_, n_noise_
        ), fontsize=15)
        fig = plt.gcf()
        fig.set_size_inches(14, 8)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    datasets = {
        'shopper': ShoppersDataset('online_shoppers_intention.csv'),
        'heart': HeartDataset('heart.csv'),
        'creditcard': CreditCardDataset('creditcard.csv'),
    }
    ClusterDBSCAN(datasets['heart'], 0.1, 20, name='Heart')
    ClusterDBSCAN(datasets['heart'], 0.15, 20, name='Heart')
    ClusterDBSCAN(datasets['heart'], 0.2, 20, name='Heart')

    ClusterDBSCAN(datasets['heart'], 0.15, 5, name='Heart')
    ClusterDBSCAN(datasets['heart'], 0.15, 10, name='Heart')
    ClusterDBSCAN(datasets['heart'], 0.15, 20, name='Heart')

    ClusterDBSCAN(datasets['shopper'], 0.05, 10, name='Shopper')
    ClusterDBSCAN(datasets['shopper'], 0.05, 20, name='Shopper')
    ClusterDBSCAN(datasets['shopper'], 0.05, 30, name='Shopper')
    ClusterDBSCAN(datasets['shopper'], 0.05, 60, name='Shopper')
    ClusterDBSCAN(datasets['shopper'], 0.05, 100, name='Shopper')

    # ClusterDBSCAN(datasets['creditcard'], 0.1, 100)
