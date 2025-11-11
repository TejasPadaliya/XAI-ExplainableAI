from src.loan import load_data, train_model, explain_with_lime, explain_batch

def main():
    print("Starting Loan Approval XAI Pipeline:\n")

    data_path = "LoanApproval/data/loan_data.csv"
    X, y = load_data(data_path)

    model, X_train, X_test, y_train, y_test = train_model(X, y)

    explain_with_lime(model, X_train, X_test, index=2, output_dir="LoanApproval/reports")

    explain_batch(model, X_train, X_test, k=min(5, len(X_test)), output_dir="LoanApproval/reports")

    print("\nOpen LoanApproval/reports/index.html to visualize explanations.")

if __name__ == "__main__":
    main()
