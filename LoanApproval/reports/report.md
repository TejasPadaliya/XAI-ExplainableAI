## **Loan Approval Explainable AI — Report**

### *1. Project Overview*

This project demonstrates **Explainable Artificial Intelligence (XAI)** using the **LIME (Local Interpretable Model-Agnostic Explanations)** framework.
The goal is to predict whether a loan application should be approved or rejected, and to explain **why** the model made that decision.

By combining a simple machine learning model (Random Forest Classifier) with LIME, we make model predictions transparent, interpretable, and accountable.

### *2. Dataset Summary*

A small dataset containing 10 loan applications was used.
Each record represents a loan applicant with the following features:

Feature: Discription

Gender: Male/Female
Married: Yes/No
Education: Graduate/Not Graduate
ApplicantIncome: Monthly income
LoanAmount: Requested loan ammount
Credit_History: 1 = good, 0 = bad
Loan_Status: Target label - Y = Approved, N = Rejected

Even though the dataset is small, it effectively demonstrates explainability for understanding the concept.

### *3. Model Training*

* **Algorithm:** Random Forest Classifier
* **Train/Test Split:** 75% / 25%
* **Accuracy:** Almost 100% (high due to limited data)

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 1.00  |
| Precision | 1.00  |
| Recall    | 1.00  |
| F1-Score  | 1.00  |

*Note:* With a larger, more complex dataset accuracy would be more realistic. The second project on German Credit Approval data is an example of this.

### *4. Explainability Results*

LIME explanations were generated for three test samples:

| File                      | Prediction | Confidence | Key Features Driving Decision                             |
| ------------------------- | ---------- | ---------- | --------------------------------------------------------- |
| `lime_explanation_0.html` | Rejected   | 0.91       | Low income, no credit history, small loan                 |
| `lime_explanation_1.html` | Approved   | 0.82       | Positive credit history, moderate income                  |
| `lime_explanation_2.html` | Approved   | 0.99       | High income, strong credit record, big loan amount |

**Note:**

HTML files are not included in GitHub.

They are intentionally excluded because:
* They contain inline JavaScript and large CSS sections
* They change every run → large diffs
* They increase repo size

**The Screenshots of each output is included in `/reports/output`**

The color bars show the direction of influence:

* **Orange = pushes toward approval**
* **Blue = pushes toward rejection**

### *5. Key Insights*

1. **Credit history** is the most influential feature across all predictions.
2. **Income level** plays a large positive role when it exceeds amount 3500–4000.
3. **Loan amount** slightly reduces approval probability when large otherwise not a big factor.
4. **Marital status** and **education** show mild influence but mostly counted as tiebreakers.
5. Each explanation is local, meaning different applicants show different dominant factors.

### *6. Responsible AI Perspective*

This project aligns with **Responsible and Ethical AI** goals:

* *Transparency* meaning each decision can be explained to a human.
* *Accountability* meaning LIME provides interpretable local reasoning.
* *Bias Awareness* — gender and marital status can introduce potential bias.

### *7. Technical Notes*

* **Model:** RandomForestClassifier (sklearn)
* **Explainability Tool:** LIMETabularExplainer
* **Output Files:**

  * Individual HTML explanations not on github but output included.
  * Index page → `/reports/index.html`

### *9. Takeaways*

* LIME helps bridge the gap between **ML models** and **human reasoning**.
* Even small datasets can demonstrate *interpretability concepts* easily.
* Transparent models increase **trust**, **fairness**, and **responsible adoption** in decision-making systems which remains the main goal behind XAI.

### *10. Author*

**Tejas Padaliya**
Bachelor of Computer Science
University of New Brunswick (Fredericton, NB)
