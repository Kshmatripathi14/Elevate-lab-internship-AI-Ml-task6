import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump, load
from sklearn.decomposition import PCA
import os

def load_iris_dataset():
    from sklearn.datasets import load_iris
    d = load_iris(as_frame=True)
    X = d.data
    y = d.target
    feature_names = d.feature_names
    target_names = d.target_names
    return X, y, feature_names, target_names

def load_csv(path, label_col):
    df = pd.read_csv(path)
    y = df[label_col].values
    X = df.drop(columns=[label_col])
    return X, y, list(X.columns), None

def normalize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_knn(X_train, y_train, k=3, metric='minkowski', p=2):
    model = KNeighborsClassifier(n_neighbors=k, metric=metric, p=p)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, target_names=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    return {'accuracy': acc, 'confusion_matrix': cm, 'report': report, 'y_pred': y_pred}

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)

def load_model(path):
    return load(path)

def pca_transform(X, n_components=2):
    pca = PCA(n_components=n_components)
    X2 = pca.fit_transform(X)
    return X2, pca
