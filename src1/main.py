import argparse
import numpy as np
from knn_pipeline import (
    load_iris_dataset, load_csv, normalize_features, train_knn,
    evaluate_model, save_model, pca_transform
)
from visualize import plot_k_accuracy, plot_confusion_matrix, plot_decision_boundary_2d
from sklearn.model_selection import train_test_split
import os
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

def run_pipeline(use_csv=None, label_col=None, test_size=0.25,
                 k=3, k_range=None, k_step=1, plot=True, output_dir='outputs'):
    if use_csv:
        X, y, feature_names, target_names = load_csv(use_csv, label_col)
    else:
        X, y, feature_names, target_names = load_iris_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    os.makedirs(output_dir, exist_ok=True)

    if k_range is None:
        k_values = [k]
    else:
        start, end = k_range
        k_values = list(range(start, end+1, k_step))

    accuracies = []
    best_model = None
    best_acc = -1
    best_k = None

    for kv in k_values:
        model = train_knn(X_train_scaled, y_train, k=kv)
        eval_res = evaluate_model(model, X_test_scaled, y_test, target_names)
        acc = eval_res['accuracy']
        accuracies.append(acc)
        print(f"k={kv} -> accuracy={acc:.4f}")
        print(eval_res['report'])
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_k = kv
            best_cm = eval_res['confusion_matrix']

    model_path = os.path.join(output_dir, f"model_knn_k{best_k}.joblib")
    dump(best_model, model_path)
    print(f"Best model (k={best_k}) saved to {model_path}")

    if plot:
        plot_k_accuracy(k_values, accuracies, output_path=os.path.join(output_dir, "accuracy_plot.png"))
        plot_confusion_matrix(best_cm, labels=[str(i) for i in sorted(set(y))],
                              output_path=os.path.join(output_dir, f"confusion_matrix_k{best_k}.png"),
                              title=f"Confusion Matrix (k={best_k})")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_all_scaled = np.vstack([X_train_scaled, X_test_scaled])
        X2 = pca.fit_transform(X_all_scaled)
        knn_2d = KNeighborsClassifier(n_neighbors=best_k)
        knn_2d.fit(X2, np.concatenate([y_train, y_test]))
        plot_decision_boundary_2d(knn_2d, X2, np.concatenate([y_train, y_test]),
                                 output_path=os.path.join(output_dir, f"decision_boundary_k{best_k}.png"),
                                 title=f"Decision boundary (k={best_k})")

    print("Pipeline finished. Outputs in:", output_dir)
    return {"best_k": best_k, "best_acc": best_acc, "model_path": model_path}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN Classification Pipeline")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV dataset (if omitted uses sklearn Iris)")
    parser.add_argument("--label_col", type=str, default=None, help="Name of label column if using CSV")
    parser.add_argument("--k", type=int, default=3, help="k for KNN (used if --k_range omitted)")
    parser.add_argument("--k_range", nargs=2, type=int, metavar=('START','END'),
                        help="Evaluate k in a range START END inclusive")
    parser.add_argument("--k_step", type=int, default=1, help="Step for k range")
    parser.add_argument("--plot", action="store_true", help="Save plots to outputs/")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for outputs")
    args = parser.parse_args()

    k_range = tuple(args.k_range) if args.k_range else None
    run_pipeline(use_csv=args.csv, label_col=args.label_col, k=args.k,
                 k_range=k_range, k_step=args.k_step, plot=args.plot, output_dir=args.output_dir)
