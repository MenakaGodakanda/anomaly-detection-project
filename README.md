# Anomaly Detection Using Machine Learning

This project implements anomaly detection using the Isolation Forest algorithm. It is designed to identify fraudulent transactions in credit card datasets. The project follows a structured approach that includes data preprocessing, model training, evaluation, and saving the trained model for future use.

## Overview
<img width="1014" alt="Screenshot 2025-02-07 at 2 39 23 pm" src="https://github.com/user-attachments/assets/a0e64c0e-13f2-4270-8af4-58392222f482" />

### Explanation

#### 1. Data Collection
- This is the first step, where we acquire the dataset for training and evaluation.
- The dataset used is the `Credit Card Fraud Detection dataset`.
- The dataset file (`creditcard.csv`) is stored in the `data/raw/` directory.

#### 2. Data Preprocessing
- Clean and transform the raw dataset to prepare it for machine learning.
  - **Feature Selection**: We remove unnecessary columns and extract relevant features.
  - **Feature Scaling**: Normalize numerical data using `StandardScaler` to ensure uniformity.
  - **Train-Test Split**: The dataset is split into:
    - **Training Set**: Used to train the anomaly detection model.
    - **Testing Set**: Used for evaluating the model’s performance.
- Processed data is stored in the `data/processed/` directory.

#### 3. Model Training
- Train the Isolation Forest model to detect anomalies.
  - **Algorithm Choice**: Isolation Forest is selected due to its efficiency in detecting outliers.
  - **Model Training**: The model is trained on the training dataset (X_train, y_train).
  - **Model Saving**: The trained model is saved as `isolation_forest_model.pkl` in the `models/` directory.

#### 4. Model Evaluation
- Test the trained model on unseen data and analyze its performance.
  - **Loading the Model**: The saved model is loaded from `models/isolation_forest_model.pkl`.
  - **Making Predictions**: The model predicts anomalies in the test dataset (`X_test`).
  - **Performance Metrics**: The results are evaluated using:
    - **Precision**: How many detected anomalies were actually frauds?
    - **Recall**: How many actual frauds were detected?
    - **F1-score**: A balance between precision and recall.

## Installation
```bash
git clone https://github.com/MenakaGodakanda/anomaly-detection-project.git
cd anomaly-detection-project
```

## Install Required Tools

### 1. Set up a virtual environment:
```
python3 -m venv anomaly_env
source anomaly_env/bin/activate
```

### 2. Install dependencies:
```
pip install pandas numpy matplotlib seaborn scikit-learn pyod jupyter
```

## Dataset
The project uses the Credit Card Fraud Detection Dataset from Kaggle. The dataset contains transactions made by credit cards in September 2013 by European cardholders. It is highly imbalanced, with only 0.172% of transactions labelled as fraudulent.
- Use an open-source dataset:
  - [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Download the dataset and place it in the `data/raw/` directory.

## Perform Exploratory Data Analysis (EDA)
- Create a Jupyter notebook (`notebooks/eda.ipynb`) for EDA.
```
jupyter notebook
```
### Basic Information
```
import pandas as pd

# Load the dataset
data = pd.read_csv("./data/raw/creditcard.csv")

# Display basic information
print(data.info())
print(data.describe())
```
![Screenshot 2025-02-07 040455](https://github.com/user-attachments/assets/42e3709d-8434-4641-8585-d3df5aa14234)
![Screenshot 2025-02-07 040522](https://github.com/user-attachments/assets/4d704a6d-cf72-4ff7-b688-e0cf5273c72d)
![Screenshot 2025-02-07 040534](https://github.com/user-attachments/assets/bedd6bac-36e5-41fb-9a73-c5cf22ccc00a)
![Screenshot 2025-02-07 040541](https://github.com/user-attachments/assets/63acc763-f81f-4617-aa23-f19b2e32639f)

### Class Imbalance
```
import matplotlib.pyplot as plt

# Visualize class imbalance
sns.countplot(x='Class', data=data)
plt.title("Class Distribution")
plt.show()
```
![Screenshot 2025-02-07 040609](https://github.com/user-attachments/assets/9ddc8899-1fae-467b-9b5c-04b379bdf8ac)
![Screenshot 2025-02-07 040614](https://github.com/user-attachments/assets/ca18291f-a26d-48a7-ae95-46351e4ebe55)

### Correlations
```
import seaborn as sns
import matplotlib.pyplot as plt

# Check correlations
corr_matrix = data.corr()
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```
![Screenshot 2025-02-07 040634](https://github.com/user-attachments/assets/be2ffbd6-1d84-44bd-9fcf-5b6f01f69322)
![Screenshot 2025-02-07 040641](https://github.com/user-attachments/assets/eded4b12-cacd-40a1-bf21-2c21c7e150f7)

## Usage
Preprocess data, train the model and evaluate the model by running `credit_card_fraud.py` script:
```
python src/main.py
```

### 1. Preprocess the Data
- Loads the dataset
- Splits it into training and testing sets
- Standardizes feature values

### 2. Train the Model
- Trains an Isolation Forest model
- Evaluates performance
- Saves the trained model in `models/`

### 3. Evaluate the Model
- Loads the trained model
- Tests it on unseen data
- Displays classification metrics

### 4. Example Output
![Screenshot 2025-02-07 140303](https://github.com/user-attachments/assets/b5d859b3-bee0-44aa-a95d-dd604e480e15)
![Screenshot 2025-02-07 140310](https://github.com/user-attachments/assets/50889272-1c0b-44d2-88c9-a9d54cd5e63e)

## Overview of the Outputs
### 1. Exploratory Data Analysis (EDA)
- **Class Distribution**: A bar chart showing the imbalance in classes (e.g., far more "non-fraudulent" samples than "fraudulent" samples).
- **Correlation Matrix**: A heatmap visualizing the relationships between features.
- **Insights**: Observations from the data, such as which features are highly correlated with anomalies.

### 2. Preprocessed Data
- **Data Splits**: Processed data divided into training and testing datasets (e.g., `X_train`, `X_test`, `y_train`, `y_test`).
- **Feature Scaling**: Features are standardized to have a mean of 0 and a standard deviation of 1.

### 3. Training the Model
- **Training Completion**: A message confirming the training of the model.
- **Model Saved**: Location where the model is saved (`models/isolation_forest_model.pkl`).

### 4. Model Evaluation
- **Classification Report**: Metrics such as Precision, Recall, F1-score, and Support for detecting anomalies.
- **Accuracy**: An overall percentage accuracy for the model.
- **Confusion Matrix (Optional)**: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).
- **Insights**:
  - The model detects anomalies reasonably well given the data imbalance.
  - Precision for the anomaly class (`Class = 1`) may be low due to the class imbalance.

### 4. Summary of Outputs
- **Preprocessing**: The dataset is loaded, scaled, and split into training/testing sets.
- **Training**: Isolation Forest is trained, predictions are made, and performance metrics are displayed.
- **Evaluation**: The saved model is loaded, and the classification report is generated.

## Project Structure
```
anomaly-detection-project/
├── data/
│   ├── raw/          # Raw data files
├── notebooks/        # Jupyter notebooks for EDA and model development
├── src/              # Source code for scripts
│   ├── main.py
├── models/           # Saved models
├── README.md         # Project description
```

## License
This project is open-source under the MIT License.
