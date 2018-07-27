import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from PIL import Image


# print(__doc__)


def From2D2points(img):
    points = list()
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] > 200:
                points.append([i, j])
    points = np.array(points)
    return points


def gt_num_pointInlable(labels):
    classes = list(set(labels))
    num_pointInlable = dict()
    for i in range(0, len(labels)):
        if labels[i] in classes:
            num_pointInlable.setdefault(str(labels[i]), []).append(i)
    return num_pointInlable


def showimage(img, points, pointstoshow, filename, key):
    height, width = img.shape
    img = np.zeros([height, width])
    PointsX = points[:, 0]
    PointsY = points[:, 1]
    for i in range(0, len(pointstoshow)):
        img[PointsX[pointstoshow[i]]][PointsY[pointstoshow[i]]] = 255
    img = Image.fromarray(img)
    img = img.convert('RGB')
    img.save('afterclustering\\clustering' + filename + str(key) + '.jpg')


def dbscanFromIMG(img, eps, min_samples):
    # input: img,eps,min_samples
    # output: dbscan, 2Dpoints from img
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, imgbin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    imgbin = img
    points = From2D2points(imgbin)
    X = points
    X1 = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X1)
    return db, points


'''
def showClustering(points,labels):
    unique_labels = set(labels)
    IfSaveFig = False
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = points[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor=tuple(col), markersize=0.11)
        if IfSaveFig:
            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.savefig('clustering1.jpg')
        xy = points[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=0.1)

    plt.title('Estimated number of mosquitos: %d' % n_clusters_)
    if IfSaveFig:
        plt.savefig('clustering.jpg')
    plt.show()
'''
dectedby2 = list()
num_mos = 0
for filename in os.listdir(r"./current"):  # read pics from folder one by one
    num_mos = num_mos + 1
    Ifdeletenoisy = False
    imgpath = os.path.join('./current', filename)
    img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
    db, points = dbscanFromIMG(img, 0.5, 200)  # Clustering
    X = points
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels))
    print('after 1st clustering, the num of all clusterings: ', n_clusters_)
    num_pointInlable = gt_num_pointInlable(labels)
    for key in num_pointInlable.keys():
        print('cluster: ', key, 'num of points:', len(num_pointInlable[key]))
        showimage(img, points, np.array(num_pointInlable[key]), filename, key)
    ##Covariance
    X2Cov = list()
    Y2Cov = list()
    for a in num_pointInlable[str(-1)]:
        X2Cov.append(X[a][0])
        Y2Cov.append(X[a][1])
    X2Cov = np.array(X2Cov)
    Y2Cov = np.array(Y2Cov)
    NoisyCov = np.cov(X2Cov, Y2Cov)
    print('Covariance:\n', NoisyCov)
    if NoisyCov[0][0] < 100 and NoisyCov[1][1] < 100:
        dectedby2.append(filename)
        print(filename, ':', NoisyCov)
print(len(dectedby2) / num_mos)
'''
# delete noisy
if Ifdeletenoisy:
    print('length of X before delete: ', len(X))
    for i in set(labels):
        if len(num_pointInlable[str(i)])<1000:
            print(i, ',deleting points: ', len(num_pointInlable[str(i)]))
            for a in num_pointInlable[str(i)]:
                imgbin[X[a][0]][X[a][1]] = 0#delete noisy
X2 = From2D2points(imgbin)
X2 = StandardScaler().fit_transform(X2)
db = DBSCAN(eps=0.05, min_samples=200).fit(X2)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('after 2nd clustering, the num of mosq: ', n_clusters_)
# Black removed and is used for noise instead.
unique_labels = set(labels)
'''
'''
#plot
unique_labels = set(labels)
IfSaveFig = False
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    class_member_mask = (labels == k)
    xy = X1[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor=tuple(col), markersize=1)
    if IfSaveFig:
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.savefig('clustering1.jpg')
    xy = X1[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=1)

plt.title('Estimated number of mosquitos: %d' % n_clusters_)
if IfSaveFig:
    plt.savefig('clustering.jpg')
plt.show()
'''
# img = Image.fromarray(imgbin)
# img.show()
