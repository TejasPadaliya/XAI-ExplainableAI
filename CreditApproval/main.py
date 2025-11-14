from src.credit_approval import load_data, train_model, explain_with_lime, explain_batch

def main():
    print("Starting Credit Approval LIME XAI Pipeline:\n")

    X, y, target_encoder = load_data("data/credit_approval.csv")

    model, X_train, X_test, y_train, y_test = train_model(X, y)

    explain_with_lime(model, X_train, X_test, target_encoder, index=0, output_dir="reports")

    explain_batch(model, X_train, X_test, target_encoder, k=5, output_dir="reports")

    print("Open CreditApproval/reports/index.html to view explanations.")

if __name__ == "__main__":
    main()
