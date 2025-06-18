# 🏦 Bank Loan Approval - Binary Classifier
A Sequential classification model trained on a large real-world bank loan dataset from Kaggle to predict loan approval.

### 📊 Dataset
Customer financial and loan data  📥  <a href="https://www.kaggle.com/datasets/omkar5/dataset-for-bank-loan-prediction" target="_blank">Dataset for Bank Loan Prediction</a>

### ⚙️ Workflow
* `EDA & Cleaning:` Handled nulls, outliers, patterns
* `Preprocessing:` Encoded categoricals, scaled features
* `Model: Sequential:` NN with 32,768 parameters, ReLU layers, sigmoid output
* `Training:` Binary cross-entropy + Adam + early stopping

### 🧠 Features used
* `Loan Status`: Indicates whether the loan was **approved or denied** (target variable for classification).
* `Current Loan Amount`: **The total** amount the client is requesting or currently owes on the loan.
* `Term**: The `length of the loan repayment period**, usually in months (e.g., 36 or 60 months).
* `Credit Score`: **A numerical value** showing the client’s **creditworthiness** based on past credit behavior.
* `Annual Income`: The client's **yearly earnings**, used to assess their ability to repay the loan.
* `Years in Current Job`: The **duration of employment** at the current job, indicating job stability.
* `Home Ownership`: Shows if the client **rents, owns, or has a mortgage**, reflecting financial stability.
* `Purpose`: The **reason** for taking the loan (e.g., debt consolidation, medical, education, etc.).
* `Monthly Debt`: The **total monthly payments** for debts (excluding the new loan), used in debt-to-income ratio.
* `Number of Open Accounts`: Total number of **active credit accounts** (like loans or credit cards) the client currently has.
* `Current Credit Balance` The **outstanding amount** the client owes across all credit accounts.
* `Bankruptcies` Number of **times** the client has declared bankruptcy, signaling past financial distress.
* `Tax Liens`: Number of **legal claims** by the government on the client’s property due to unpaid taxes.

### 📁 Files
* <a href="https://github.com/NishantkSingh0/Predictive-Models/blob/main/LoanPrediction.ipynb" target="_blank">LoanPrediction.ipynb</a> – Full pipeline
* <a href="https://www.kaggle.com/code/nishantsingh96/loanprediction" target="_blank">Kaggle Notebook</a> - Full Explanation

### 🚀 Results
Achieved strong classification performance of 80% on unseen test data with a well-generalized and compact model.

<br><br>
--- 
<br><br>

# ✈️ Airline Delay & Cancellation Prediction - Binary Classifier  
A Sequential classification model trained on a large real-world airline dataset to predict whether a flight will be delayed by 15+ minutes or canceled.

### 📊 Dataset  
Historical flight data enriched with weather, airport, and carrier features.  
<a href="https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations" target="_blank">2019 airline delays and cancellations</a>  
🎯 Target: `DEP_DEL15` – Binary indicator for delay >15 minutes or cancellation.  

### ⚙️ Workflow  
- **`EDA & Cleaning`**: Handled missing values, outliers, anomalies  
- **`Preprocessing`**:  
  - Encoded categorical features  
  - Applied feature scaling  
  - Log-transformed skewed features  
- **`Custom Class Weights`**: Balanced training for imbalanced label distribution  
- **`Model`**:  
  - Deep Neural Network with multiple Dense layers, Dropout, and BatchNormalization  
  - Total parameters: ~35K  
  - Loss: Binary cross-entropy  
  - Optimizer: Adam  
  - Metrics: Accuracy, Precision, Recall, AUC  

### 🧠 Features Used  
- `MONTH`: Month of flight  
- `DAY_OF_WEEK`: Day of the week  
- `DEP_TIME_BLK`: Departure time block (e.g., 0600–0659)  
- `DISTANCE_GROUP`: Distance bucket of the flight  
- `SEGMENT_NUMBER`: Flight segment order  
- `CONCURRENT_FLIGHTS`: Concurrent departing flights in same time block  
- `NUMBER_OF_SEATS`: Seats available on aircraft  
- `CARRIER_NAME`: Airline carrier  
- `AIRPORT_FLIGHTS_MONTH`: Avg. airport traffic  
- `AIRLINE_FLIGHTS_MONTH`: Avg. flights operated by airline  
- `AIRLINE_AIRPORT_FLIGHTS_MONTH`: Combined airport-airline traffic  
- `PLANE_AGE`: Age of aircraft  
- `LATITUDE`, `LONGITUDE`: Geolocation of airport  
- `PRCP`, `SNOW`, `SNWD`, `TMAX`, `AWND`: Weather conditions (precipitation, snow, temp, wind)

### 📁 Files  
* <a href="https://github.com/NishantkSingh0/Predictive-Models/blob/main/Airline-Delays-Cancellation-Prediction.ipynb" target="_blank">LoanPrediction.ipynb</a> – Full pipeline 
- <a href="https://www.kaggle.com/code/nishantsingh96/airline-delays-cancellation-prediction" target="_blank">Kaggle Notebook</a> - Full Explanation 

### 🚀 Results  
Achieved **`~65% accuracy`** on unseen test data using a custom-weighted loss function and tuned architecture.  
The model shows strong generalization for real-world flight scheduling and operations forecasting.

<br><br>
--- 
<br><br>

# 🏡 House Price Prediction - Regression Model  
A machine learning regression model trained on the well-known California housing dataset to predict house prices based on key demographic and geographic features.

### 📊 Dataset  
California housing dataset from `sklearn.datasets`, derived from the 1990 U.S. Census.  
📥 <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html" target="_blank">California Housing Dataset Documentation</a>

### ⚙️ Workflow  
- **`EDA & Cleaning`**: Inspected and visualized data distribution, identified and handled outliers  
- **`Preprocessing`**: Scaled features using `StandardScaler`  
- **`Model`**: Trained a Linear Regression model and a Tree-based regressor (e.g., RandomForestRegressor)  
- **`Training`**: Used `mean_squared_error` as the loss metric and evaluated performance using MAE, RMSE, and R² score

### 🧠 Features Used  
- `MedInc`: Median income in the block group (normalized)  
- `HouseAge`: Median age of houses in the area  
- `AveRooms`: Average number of rooms per household  
- `AveBedrms`: Average number of bedrooms per household  
- `Population`: Total population in the block group  
- `AveOccup`: Average number of household members  
- `Latitude`: Geographical latitude  
- `Longitude`: Geographical longitude  
- 🎯 `Price`: Median house value (target variable, scaled)

### 📁 Files  
- <a href="https://github.com/NishantkSingh0/Predictive-Models/blob/main/HousePricePredictions.ipynb" target="_blank">HousePricePrediction.ipynb</a> – Full code pipeline with training, evaluation, and visualizations  
- <a href="https://colab.research.google.com/drive/1Zvz7cSAZx8cipfW3fV46U3ckICCgQ0AD?usp=sharing" target="_blank">Colab Notebook</a> – Explained end-to-end

### 🚀 Results  
Achieved strong regression performance on test data:  
- `MAE`: ~0.1582  
- `RMSE`: ~0.2072  
- `R² Score`: ~0.82  
Demonstrated good generalization and accurate predictions with well-preprocessed features and effective model choice.

