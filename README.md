# K-Nearest Neighbors (KNN) Classification — Example Project

This repository demonstrates a simple KNN classification workflow using the Iris dataset:
- Data loading
- Feature normalization
- Model training (KNN) across multiple `k` values
- Evaluation with accuracy & confusion matrix
- Decision boundary visualization using 2D PCA projection

## How to run

1. Create a virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate       # on Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the main pipeline:
```bash
python src/main.py --k 3 --plot
```

3. Try range of k values:
```bash
python src/main.py --k_range 1 15 --k_step 2 --plot
```

## Files
- `src/main.py` — runable pipeline/CLI
- `src/knn_pipeline.py` — model training & evaluation utilities
- `src/visualize.py` — plotting helpers
- `outputs/` — generated plots & saved model
