ML Assignment 2
1. Problem Statement
The goal of this project is to develop and compare multiple machine learning classification models to predict the target variable of the chosen dataset.
The assignment requires training six different ML models on the same dataset, evaluating them using standard classification metrics, and deploying a Streamlit web application that allows users to upload test data and view model performance, confusion matrix, and a classification report.
This project demonstrates a complete ML workflow including:

Data preprocessing
Model training
Evaluation
Saving trained models
Building an interactive Streamlit app
Deployment to Streamlit Cloud

2. Dataset Description 

Dataset Name: Customer-Churn
Source: UCI
Total Rows: 3151
Features: 13
Target Column: Churn

The dataset contains only numerical features.
The objective is a binary classification task where the target variable indicates whether a given instance belongs to class 0 or 1.
A standard 80/20 train-test split was used for all models.

3. Models Used & Comparison Table
You are required to evaluate all six models using:
•	Accuracy
•	AUC
•	Precision
•	Recall
•	F1 Score
•	MCC
The following six models were implemented:
1.	Logistic Regression
2.	Decision Tree Classifier
3.	k Nearest Neighbors (kNN)
4.	Gaussian Naive Bayes
5.	Random Forest (Ensemble)
6.	XGBoost (Ensemble)
 
 Comparison Table of All Evaluation Metrics

| Model                   | Accuracy |   AUC   | Precision | Recall |   F1   |   MCC   |
|-------------------------|----------|---------|-----------|--------|--------|---------|
| Logistic Regression     | 0.8968   | 0.9208  | 0.8920    | 0.8968 | 0.8821 | 0.5509  |
| Decision Tree           | 0.9270   | 0.8673  | 0.9264    | 0.9270 | 0.9267 | 0.7221  |
| kNN                     | 0.9556   | 0.9680  | 0.9552    | 0.9556 | 0.9554 | 0.8309  |
| Naive Bayes (Gaussian)  | 0.7381   | 0.8986  | 0.8783    | 0.7381 | 0.7727 | 0.4536  |
| Random Forest           | 0.9651   | 0.9877  | 0.9645    | 0.9651 | 0.9645 | 0.8648  |
| XGBoost                 | 0.9603   | 0.9922  | 0.9605    | 0.9603 | 0.9604 | 0.8508  |
 
4. Observations on Model Performance 

| Model                   | Observation |
|-------------------------|-------------|
| Logistic Regression     | Performs well with a strong AUC (0.9208), indicating good rank-ordering ability. Accuracy and F1 are solid, but MCC (0.5509) shows only moderate balanced performance, meaning it struggles slightly on class separation compared to more complex models. |
| Decision Tree           | Achieves high accuracy (0.9270) and F1 (0.9267), showing it fits the dataset well. However, the lower AUC (0.8673) suggests weaker generalization and possible overfitting. MCC (0.7221) indicates good but not perfect balanced prediction. |
| kNN                     | One of the strongest performers with accuracy (0.9556) and high AUC (0.9680). Strong MCC (0.8309) shows excellent balanced classification. Works well due to clear neighborhood structure in the dataset and effective feature scaling. |
| Naive Bayes (Gaussian)  | Lowest accuracy (0.7381) and MCC (0.4536), meaning its assumptions do not match the dataset. Despite this, AUC (0.8986) remains decent—indicating that while probability ranking is good, classification thresholds are less effective. |
| Random Forest           | Best overall performer with high accuracy (0.9651), AUC (0.9877), and MCC (0.8648). Handles non-linear feature interactions extremely well and generalizes strongly due to ensemble averaging. |
| XGBoost                 | Highest AUC (0.9922), showing exceptional ranking capability. Accuracy (0.9603) and MCC (0.8508) are excellent. Very robust model due to gradient boosting and ability to capture complex patterns. |

