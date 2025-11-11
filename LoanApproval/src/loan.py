import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from lime.lime_tabular import LimeTabularExplainer
from datetime import datetime

# Ignores harmless warnings caused by numerical states
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all='ignore')

# Load dataset and prepares it to run 
def load_data(data_path):
    data = pd.read_csv(data_path)

    for col in ['Gender', 'Married', 'Education']:
        data[col] = LabelEncoder().fit_transform(data[col])

    X = data.drop('Loan_Status', axis=1)
    y = LabelEncoder().fit_transform(data['Loan_Status'])
    return X, y

# Trains model using data and returns model and its splits X for input and y for target output
def train_model(X, y, test_size=0.25, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if len(set(y)) > 1 else None
    )

    model = RandomForestClassifier(random_state=seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Model trained successfully! Accuracy: {acc:.2f}\n")
    print("Classification Report:\n", classification_report(y_test, preds))
    return model, X_train, X_test, y_train, y_test

# Creates a wrapper around model.predict_proba() so that LIME, which passes a NumPy array, can still call the model with proper feature names
def _predict_with_feature(model, X_train_columns):
    def predict_fn(x_numpy):
        df = pd.DataFrame(x_numpy, columns=X_train_columns)
        return model.predict_proba(df)
    return predict_fn

# Generate LIME explanation for a single instance
def explain_with_lime(model, X_train, X_test, index=0, output_dir="../reports"):
    explain = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Rejected', 'Approved'],
        mode='classification'
    )

    predict_fn = _predict_with_feature(model, X_train.columns.tolist())

    exp = explain.explain_instance(
        X_test.iloc[index].values,
        predict_fn,
        num_features=5
    )

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"lime_explanation_{index}.html")
    exp.save_to_file(file_path)
    abs_path = os.path.abspath(file_path)
    print(f"LIME explanation saved: {abs_path}")
    return abs_path

# Generate multiple LIME explanations in batch; here 3 exlanations
def explain_batch(model, X_train, X_test, k=5, output_dir="../reports"):
    os.makedirs(output_dir, exist_ok=True)
    k = max(1, min(k, len(X_test)))

    explain = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Rejected', 'Approved'],
        mode='classification'
    )
    predict_fn = _predict_with_feature(model, X_train.columns.tolist())

    paths = []
    for i in range(k):
        exp = explain.explain_instance(
            X_test.iloc[i].values, predict_fn, num_features=5
        )
        p = os.path.join(output_dir, f"lime_explanation_{i}.html")
        exp.save_to_file(p)
        paths.append(os.path.abspath(p))

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(f"<h2>LIME Explanations ({ts})</h2><ul>")
        for p in paths:
            rel = os.path.basename(p)
            f.write(f'<li><a href="{rel}">{rel}</a></li>')
        f.write("</ul>")

    print(f"Index written: {os.path.abspath(index_path)}")
    return paths
