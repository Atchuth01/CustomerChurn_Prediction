# CustomerChurn_Prediction
 Customer attrition(churn) prediction on a Bank customer churn dataset using python[Machine Learning].

 Here’s a sample `README.md` file description for your project on customer churn prediction using a Random Forest model:

---

# Bank Customer Churn Prediction

This project aims to predict customer churn (i.e., whether a customer will leave the bank) using a Random Forest classifier. It is built using a bank's customer dataset and leverages machine learning techniques to accurately identify customers who are likely to leave. The project includes data preprocessing, model training, testing, and evaluation.

## Project Overview

Customer churn is a critical business issue for banks, as it reflects a loss of revenue and customer base. This project uses a dataset of bank customers, including features such as customer demographics, bank account details, and customer behavior. The goal is to predict whether a customer will churn or not using Random Forest, a powerful ensemble learning technique.To identify and predict which customers are likely to stop doing business with the bank. This allows the bank to take proactive measures to retain these customers and minimize customer attrition.

### Key Features
- **Data Preprocessing**: One-Hot Encoding is applied to categorical features (`country`, `gender`) to convert them into numerical format.
- **Model**: A Random Forest classifier is used to train on historical data and predict future churn.
- **Model Evaluation**: Accuracy, classification report, and confusion matrix are used to evaluate the model’s performance.
- **Customer Churn Prediction**: After model training, predictions are made on a separate test dataset containing customer information, and a list of customers likely to churn is printed.

- **Accuracy and Update**: This model has 86% accuracy. You can another algorithms like Logistic Regression, Decision trees or any other for classification to increase the model performance and accuracy.and u can tune the parameters for better performance.

- **Results**: This model tested on a test data set and predicted and printed the Customers ID's who are going to leave teh bank and also printed classification report(accuracy, f1_score and so on...).

## Dataset

The project uses two datasets:
- `Customer Churn.csv`: Contains features like country, gender, customer ID, and churn status.
- `test_data[churn].csv`: Contains customer information on which the churn prediction is tested.

Both datasets should be placed in the root directory for the code to work.

### Dataset Features
- **CustomerID**: Unique identifier for each customer.
- **Country**: The country in which the customer resides.
- **Gender**: The gender of the customer.
- **Churn**: Target variable indicating whether the customer churned (1) or not (0).
- **Other features**: Additional customer-related attributes that help in the prediction process.

## Installation and Requirements

To run this project locally, you need the following libraries:

- pandas
- scikit-learn

You can install them using pip:

```bash
pip install pandas scikit-learn
```

## Running the Project

1. Clone this repository to your local machine:

```bash
git clone https://github.com/Atchuth01/customer-churn-prediction.git
```

2. Ensure you have the datasets (`Customer Churn.csv` and `test_data[churn].csv`) in the root directory.

3. Run the script:

```bash
python churn_prediction.py
```

### Output

The script will output:
- **Accuracy** of the model.
- **Classification Report** showing precision, recall, F1-score, and support.
- **Customers likely to churn** from the test dataset.

## Model Explanation

### Random Forest Classifier

Random Forest is an ensemble learning method that fits multiple decision trees on various sub-samples of the dataset and uses averaging to improve the model's accuracy and control over-fitting. In this project, we used 100 trees (`n_estimators=100`) for the Random Forest model.

### Evaluation Metrics

- **Accuracy**: Measures the overall correctness of the model's predictions.
- **Classification Report**: Provides detailed precision, recall, and F1-score for both churned and non-churned customers.
  
## Conclusion

This project demonstrates how a machine learning model can be used to predict customer churn in the banking industry. With the use of a Random Forest classifier, we achieve strong predictive performance, helping banks identify at-risk customers.

---

## Future Improvements

- Fine-tuning the Random Forest model using techniques like hyperparameter tuning.
- Exploring additional machine learning algorithms such as Gradient Boosting or XGBoost for comparison.
- Incorporating feature selection to reduce dimensionality and improve performance.
  
## License

This project is licensed under the MIT License.


