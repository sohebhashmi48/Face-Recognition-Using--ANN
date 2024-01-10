import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
import os
import cv2

def plotgallery(images, titles, h, w, n_rows=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_rows))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_rows * n_col):
        plt.subplot(n_rows, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())

# Load faces dataset
dir_name = "datasets/faces/"
y = []
x = []
target_names = []
person_id = 0
h = w = 300
n_sample = 0
class_names = []

for person_name in os.listdir(dir_name):
    dir_path = dir_name + person_name + "/"
    class_names.append(person_name)
    for image_name in os.listdir(dir_path):
        image_path = dir_path + image_name
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, (h, w))
        v = resized_image.flatten()
        x.append(v)  # Corrected line
        n_sample = n_sample + 1
        y.append(person_id)
        target_names.append(person_name)
    person_id = person_id + 1

y = np.array(y)
x = np.array(x)
target_names = np.array(target_names)
n_features = x.shape[1]  # Corrected line
print(y.shape, x.shape, target_names.shape)
print("number of samples:", n_sample)

# PCA
n_components = 150
print("Extracting the %d eigenfaces from %d faces" % (n_components, x.shape[0]))
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(x)

eigenfaces = pca.components_.reshape(n_components, h, w)

# Plot eigenfaces
eigenfaces_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plotgallery(eigenfaces, eigenfaces_titles, h, w)
plt.show()

print("Projecting the input data on the eigenfaces orthonormal basis")
x_pca = pca.transform(x)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.25, random_state=42)

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)

x_train_lda = lda.transform(x_train)
x_test_lda = lda.transform(x_test)

# Neural Network (MLP) Classifier
clf = MLPClassifier(random_state=1, hidden_layer_sizes=(10, 10), max_iter=1000, verbose=True).fit(x_train_lda, y_train)

print("Model weight: ")
model_info = [coef.shape for coef in clf.coefs_]
print(model_info)

# Prediction and Evaluation
y_pred = clf.predict(x_test_lda)
accuracy = np.sum(y_pred == y_test) / len(y_test) * 100
print("Accuracy: {:.2f}%".format(accuracy))

# Plot results
prediction_titles = ["true: {}\npred: {}".format(class_names[true_id], class_names[pred_id]) for true_id, pred_id in zip(y_test, y_pred)]
plotgallery(x_test, prediction_titles, h, w)
plt.show()
