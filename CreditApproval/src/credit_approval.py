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
np.seterr(all="ignore")

# Load dataset and prepares it to run 
def load_data(path):
    df = pd.read_csv(path, na_values=["?"])

    for col in df.columns[:-1]:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    target = df.columns[-1]
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[target].astype(str))

    X = df.drop(columns=[target])

    print("Dataset loaded with shape:", df.shape)
    print("Target distribution:\n", pd.Series(target_encoder.classes_).value_counts(), "\n")

    return X, y, target_encoder

# Trains model using data and returns model and its splits X for input and y for target output
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("Model trained successfully! Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    return model, X_train, X_test, y_train, y_test

# Creates a wrapper around model.predict_proba() so that LIME, which passes a NumPy array, can still call the model with proper feature names
def _predict_fn(model, feature_names):
    def predict(x_numpy):
        df = pd.DataFrame(x_numpy, columns=feature_names)
        return model.predict_proba(df)
    return predict

# Generate LIME explanation for a single instance
def explain_with_lime(model, X_train, X_test, target_encoder, index=0, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)

    explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=target_encoder.classes_.tolist(),
    mode="classification"
    )

    predict_fn = _predict_fn(model, X_train.columns.tolist())

    exp = explainer.explain_instance(
    X_test.iloc[index].values,
    predict_fn,
    num_features=8,
    top_labels=1
)
    label_to_explain = exp.top_labels[0]
    html_path = os.path.join(output_dir, f"lime_explanation_{index}.html")
    exp.save_to_file(html_path)

    print(f"LIME explanation saved: {os.path.abspath(html_path)}")
    return html_path

# Generate multiple LIME explanations in batch; here 5 exlanations
def explain_batch(model, X_train, X_test, target_encoder, k=5, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=target_encoder.classes_.tolist(),
        mode="classification"
    )

    predict_fn = _predict_fn(model, X_train.columns.tolist())

    paths = []
    k = min(k, len(X_test))

    for i in range(k):
        exp = explainer.explain_instance(
            X_test.iloc[i].values,
            predict_fn,
            num_features=8,
            top_labels=1
        )

        label_to_explain = exp.top_labels[0]
        file_path = os.path.join(output_dir, f"lime_explanation_{i}.html")
        exp.save_to_file(file_path)
        paths.append(file_path)

    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w") as f:
        f.write("<h1>Credit Approval — LIME Explanations</h1><ul>")
        for p in paths:
            f.write(f'<li><a href="{os.path.basename(p)}">{os.path.basename(p)}</a></li>')
        f.write("</ul>")

    print("Batch explanations saved.")
    print("Index:", os.path.abspath(index_path))

    return paths
