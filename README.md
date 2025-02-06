# Anomaly Detection Project

## Overview
This project demonstrates anomaly detection using machine learning.


## Features
- Data preprocessing
- Model training with Isolation Forest
- Model evaluation

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
### Class Imbalance
```
import matplotlib.pyplot as plt

# Visualize class imbalance
sns.countplot(x='Class', data=data)
plt.title("Class Distribution")
plt.show()
```
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

## Usage
Preprocess data, train the model and evaluate the model by running `credit_card_fraud.py` script:
```
python src/credit_card_fraud.py
```

### 1. Exploratory Data Analysis (EDA)
- **Class Distribution**: A bar chart showing the imbalance in classes (e.g., far more "non-fraudulent" samples than "fraudulent" samples).
- **Correlation Matrix**: A heatmap visualizing the relationships between features.
- **Insights**: Observations from the data, such as which features are highly correlated with anomalies.

- Example Output:
  - Class Distribution

  - Correlation Matrix

### 2. Preprocessed Data
- **Data Splits**: Processed data divided into training and testing datasets (e.g., `X_train`, `X_test`, `y_train`, `y_test`).
- **Feature Scaling**: Features are standardized to have a mean of 0 and a standard deviation of 1.
- Example Output:

### 3. Training the Model
- **Training Completion**: A message confirming the training of the model.
- **Model Saved**: Location where the model is saved (`models/isolation_forest_model.pkl`).
- Example Output:

### 4. Model Evaluation
- **Classification Report**: Metrics such as Precision, Recall, F1-score, and Support for detecting anomalies.
- **Accuracy**: An overall percentage accuracy for the model.
- **Confusion Matrix (Optional)**: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).
- Example Output:

- Insights:
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
│   ├── credit_card_fraud.py
├── models/           # Saved models
├── README.md         # Project description
```

## License
This project is open-source under the MIT License.
