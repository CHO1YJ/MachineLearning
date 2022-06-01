import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn import datasets

def plot_data(dataset, labels, position, title):
    X, Y = dataset
    colors = np.array(['#BDE6F1', '#FFE59D', '#EEB0B0', '#7D1E6A', '#5FD068'])
    
    if len(np.unique(labels)) > len(colors):
        print("Error : k 값을 줄이거나 Color를 추가하세요")
    else:
        plt.subplot(position)
        plt.scatter(X[:, 0], X[:, 1], color=colors[labels])
        plt.title(title)
        plt.tight_layout()
    return True

n = 1500 # 데이터 수
blobs = datasets.make_blobs(n_samples=n, centers=3)
default_label1 = np.zeros(n, dtype='int')
plot_data(blobs, default_label1, 121, 'Blobs')
# blobs_input=blobs[0]
# plt.scatter(blobs_input[:, 0], blobs_input[:, 1])

moons = datasets.make_moons(n_samples=n, noise=0.05)
default_label2 = np.zeros(n, dtype='int')
plot_data(moons, default_label2, 122, 'Moons')
# moons_input=moons[0]
# plt.scatter(moons_input[:, 0], moons_input[:, 1])

#=================== k-means ========================
model = cluster.KMeans(n_clusters=3)
blobs_input=blobs[0]
moons_input=moons[0]

model.fit(blobs_input)
blobs_label = model.predict(blobs_input)

model.fit(moons_input)
moons_label = model.predict(moons_input)

plot_data(blobs, blobs_label, 121, 'Blobs')
plot_data(moons, moons_label, 122, 'Moons')

#=================== Elbow method ========================
from yellowbrick.cluster import KElbowVisualizer
model_elbow = cluster.KMeans()
visualizer = KElbowVisualizer(model_elbow, k=(1, 5))
visualizer.fit(blobs_input)
visualizer.show()

#=================== DBSCAN ========================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
model = cluster.DBSCAN(eps=0.25, min_samples=3, metric='euclidean') # eps : 거리, min_samples : border point
scaler_blobs = scaler.fit_transform(blobs_input)
scaler_moons = scaler.fit_transform(moons_input)

model.fit(scaler_blobs)
blobs_label_DBSCAN = model.fit_predict(scaler_blobs)

model.fit(scaler_moons)
moons_label_DBSCAN = model.fit_predict(scaler_moons)

plot_data(blobs, blobs_label_DBSCAN, 121, 'Blobs')
plot_data(moons, moons_label_DBSCAN, 122, 'Moons')










