# Credit Card Approval Prediction
This project builds a machine learning model to automatically predict the approval of credit card applications. Commercial banks receive a high volume of applications, and automating this process can save time and reduce errors. This notebook demonstrates the end-to-end process of cleaning data, building a predictive model, and optimizing its performance.

The model is built using the Credit Card Approval dataset from the UCI Machine Learning Repository.

# Methodology
The project follows several key steps to ensure the data is ready for modeling and to build an effective predictor:

Data Loading and Inspection: The dataset is loaded using pandas and inspected to understand its structure, which includes a mix of numerical and non-numerical features with anonymized column names.

# Data Preprocessing:

Handling Missing Values: Missing values, marked as '?' in the dataset, are first replaced with NaN. Numerical missing values are then imputed using the mean of their respective columns, while categorical missing values are imputed with the most frequent value.

Categorical Data Encoding: All non-numeric (object-type) columns are converted into a numerical format using scikit-learn's LabelEncoder to make them suitable for the machine learning model.

Feature Scaling: Feature values are scaled to a uniform range of 0 to 1 using MinMaxScaler to improve model performance.

# Model Building and Evaluation:

Train-Test Split: The dataset is split into training (67%) and testing (33%) sets.

Model Selection: A Logistic Regression classifier is chosen for this classification task.

# Evaluation: The model's performance is evaluated based on its accuracy score and confusion matrix.

# Hyperparameter Tuning:

To improve the model's performance, GridSearchCV is used to find the optimal hyperparameters for tol and max_iter.

# Results
The initial Logistic Regression model achieved an accuracy of ~84% on the test set.

After hyperparameter tuning with GridSearchCV, the optimized model achieved a cross-validated accuracy of ~85%.

# Technologies Used
Python

pandas: For data manipulation and analysis.

NumPy: For numerical operations.

scikit-learn: For data preprocessing, model building, and evaluation.
