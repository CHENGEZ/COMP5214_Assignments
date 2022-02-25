import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from FileLoader import *

train_images, train_labels = load_train_data()
test_images, test_labels = load_test_data()

knn = KNeighborsClassifier(n_neighbors=1, p=1)
num_error = 0
errors = np.zeros(10)

for K in range(1, 11):
    print("K =", K, end=" :")
    num_error = 0
    knn.n_neighbors = K
    knn.fit(train_images.reshape(60000, 28*28), train_labels)
    prediction = knn.predict(test_images.reshape(10000, 28*28))
    for truth, predict in zip(test_labels, prediction):
        if truth != predict:
            num_error += 1
    errors[K-1] = num_error
    print("the accuracy is", 1-num_error/10000)

fig, axs = plt.subplots(1, 1)
k = np.arange(1, 11, 1)

fig.suptitle(r"error rate for different 'K'", fontsize=20)
axs.plot(k, errors/10000)  # Plot the graph

plt.show()
