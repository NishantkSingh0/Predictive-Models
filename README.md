# 🏦 Bank Loan Approval - Binary Classifier
A Sequential classification model trained on a large real-world bank loan dataset from Kaggle to predict loan approval.

### 📊 Dataset
Customer financial and loan data  📥  <a href="https://www.kaggle.com/datasets/omkar5/dataset-for-bank-loan-prediction" target="_blank">Dataset for Bank Loan Prediction</a>

### ⚙️ Workflow
* **EDA & Cleaning:** Handled nulls, outliers, patterns
* **Preprocessing:** Encoded categoricals, scaled features
* **Model: Sequential** NN with 32,768 parameters, ReLU layers, sigmoid output
* **Training:** Binary cross-entropy + Adam + early stopping

### 🧠 Features used
* **Loan Status**: Indicates whether the loan was **approved or denied** (target variable for classification).
* **Current Loan Amount**: **The total** amount the client is requesting or currently owes on the loan.
* **Term**: The **length of the loan repayment period**, usually in months (e.g., 36 or 60 months).
* **Credit Score**: **A numerical value** showing the client’s **creditworthiness** based on past credit behavior.
* **Annual Income**: The client's **yearly earnings**, used to assess their ability to repay the loan.
* **Years in Current Job**: The **duration of employment** at the current job, indicating job stability.
* **Home Ownership**: Shows if the client **rents, owns, or has a mortgage**, reflecting financial stability.
* **Purpose**: The **reason** for taking the loan (e.g., debt consolidation, medical, education, etc.).
* **Monthly Debt**: The **total monthly payments** for debts (excluding the new loan), used in debt-to-income ratio.
* **Number of Open Accounts**: Total number of **active credit accounts** (like loans or credit cards) the client currently has.
* **Current Credit Balance** The **outstanding amount** the client owes across all credit accounts.
* **Bankruptcies** Number of **times** the client has declared bankruptcy, signaling past financial distress.
* **Tax Liens**: Number of **legal claims** by the government on the client’s property due to unpaid taxes.

### 📁 Files
* <a href="https://github.com/NishantkSingh0/Predictive-Models/blob/main/LoanPrediction.ipynb" target="_blank">LoanPrediction.ipynb</a> – Full pipeline
* <a href="https://www.kaggle.com/code/nishantsingh96/loanprediction" target="_blank">Kaggle Notebook</a> - Full Explanation

### 🚀 Results
Achieved strong classification performance of 80% on unseen test data with a well-generalized and compact model.
