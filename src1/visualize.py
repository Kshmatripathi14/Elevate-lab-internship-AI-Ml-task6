import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

sns.set(style="whitegrid", context="notebook")

def plot_k_accuracy(k_values, accuracies, output_path=None):
    plt.figure(figsize=(8,5))
    plt.plot(k_values, accuracies, marker='o')
    plt.title('K vs Accuracy')
    plt.xlabel('k (n_neighbors)')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_confusion_matrix(cm, labels, output_path=None, title='Confusion Matrix'):
    plt.figure(figsize=(6,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title(title)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_decision_boundary_2d(model, X, y, output_path=None, title='Decision boundary (2D PCA)'):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00AA00', '#0000FF'])

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    scatter = plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=40)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(handles=scatter.legend_elements()[0], labels=np.unique(y).tolist(), title="Classes")
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
