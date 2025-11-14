## **Credit Approval Explainable AI — Report**

## *1. Project Overview*

This project demonstrates **Explainable Artificial Intelligence (XAI)** using the
**LIME (Local Interpretable Model-Agnostic Explanations)** framework on the **German Credit Approval dataset**.

The goal is to build a classifier that predicts the **purpose of a customer’s credit request** and then explain *why* the model made each prediction.

## *2. Dataset Summary*

The dataset (`credit_approval.csv`) contains **1,000 customer credit records**, each with demographic and financial attributes.

### **Features Include:**

| Feature          | Description                                             |
| ---------------- | ------------------------------------------------------- |
| Age              | Applicant age                                           |
| Sex              | Male/Female                                             |
| Job              | Job type category                                       |
| Housing          | Own / Free / Rent                                       |
| Saving accounts  | Financial savings                                       |
| Checking account | Checking account balance                                |
| Credit amount    | Requested credit amount                                 |
| Duration         | Loan duration (months)                                  |
| Purpose          | Target label (car, furniture, radio/TV, business, etc.) |

### **Target (Multiclass)**

There are *8 possible outcomes*:

* car
* radio/TV
* furniture/equipment
* business
* education
* repairs
* domestic appliances
* vacation/others

This category of **“Purpose”** is used as the prediction target.

## *3. Model Training*

* **Algorithm:** Random Forest Classifier
* **Train/Test Split:** 80% / 20%
* **Classes:** 8 (multiclass problem)

### **Performance Snapshot**

| Metric                    | Value                                               |
| ------------------------- | --------------------------------------------------- |
| Accuracy                  | ~0.33                                               |
| Model Type                | Multiclass                                          |
| Reason for Lower Accuracy | Many overlapping classes and categorical complexity |

Although accuracy is modest here, this dataset is meant for **explainability and interpretability**, not performance.

## *4. LIME Explainability*

LIME is used to generate **local explanations** for individual predictions.

Because this is a **multiclass** prediction problem:

### LIME explains based on the *actual predicted class*

**Note:**

HTML files are not included in GitHub.

They are intentionally excluded because:
* They contain inline JavaScript and large CSS sections
* They change every run → large diffs
* They increase repo size

**The Screenshots of each output is included in `/reports/output`**

## *5. What Each Explanation Shows*

Each LIME explanation includes:

### *a Prediction Probabilities*

LIME displays the model’s confidence across all 8 classes:

* car
* furniture/equipment
* radio/TV
* business
* education
* repairs
* domestic appliances
* vacation/others

It shows both:

* The predicted class
* The probability distribution over alternatives

### *b Feature Influence Bars*

The central panel shows two sides:

#### **Left — NOT (Predicted Class)**

Features pushing the model *away* from predicting the chosen category.

#### **Right — Predicted Class**

Features pushing the model *toward* the chosen category.

### *c Feature Table*

The rightmost section shows:

Feature          Value Used in Prediction
---------------------------------------
Age              XX
Duration         XX
Credit amount    XX
Job              XX
Saving accounts  XX
Checking account XX
Sex              XX
Housing          XX

## *6. Responsible AI Considerations*

This project supports ethical and explainable AI goals:

* **Transparency** — Every prediction is accompanied by an explanation.
* **Accountability** — Human decision-makers can understand the reasoning.
* **Bias detection** — LIME reveals when demographic attributes influence outcomes.

## *8. Technical Notes*

* **Model:** RandomForestClassifier
* **Explainability Tool:** LIMETabularExplainer
* **Encoding:** LabelEncoder for both features and target
* **Outputs:**

  * Individual HTML explanations not on github but output included.
  * Navigation `index.html` file


## *9. Summary*

* This project demonstrates how **LIME can interpret multiclass predictions**, making machine learning decisions understandable even when the model is complex.
* It provides a transparent view into credit approval decisions and showcases practical XAI techniques for responsible ML deployment.

## *10. Author*

**Tejas Padaliya**
Bachelor of Computer Science
University of New Brunswick, Fredericton (NB)

