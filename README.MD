# Diabetes Disease Prediction Machine Learning Project

![Header Image](images/01.jpg)
## Overview

This repository contains code for a machine learning project focused on predicting the likelihood of a person having diabetes. The project includes the implementation of various classification models and an Artificial Neural Network (ANN) for classification.

- [Direct Link to Web App](https://desease-prediction.streamlit.app/)
- [Direct Link to Documentation](https://github.com/Arya920/Desease-prediction/blob/main/Documentation/Decease_Prediction.pdf)
## Models Implemented

The following models have been implemented in this project:

- Logistic Regression
- Support Vector Classifier (SVC)
- Gaussian Naive Bayes
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Extra Trees Classifier
- XGBoost Classifier
- LightGBM Classifier (imported as `lgb.LGBMClassifier`)

## Cloning the Repository

1. To clone this repository, run the following command in your terminal:

```bash
   git clone https://github.com/Arya920/Desease-prediction.git
```

2. Install the Required Packages:
```bash
   pip install -r requirements.txt
```

3. Run the web application:
```bash
   streamlit run app.py
```

## Models Details

- Logistic Regression
  - Description: Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that can be used to predict the outcome of a categorical dependent variable. It models the probability that the dependent variable belongs to a particular category.

- Support Vector Classifier (SVC)
  - Description: Support Vector Classifier is a type of support vector machine that is used for classification tasks. It works by finding the hyperplane that best separates the classes in the feature space. SVC aims to maximize the margin between the classes.

- Gaussian Naive Bayes
  - Description: Gaussian Naive Bayes is a probabilistic classifier that makes predictions using the Bayes' theorem, assuming that the features are independent and follow a Gaussian distribution. It is particularly useful for text classification tasks.

- Random Forest Classifier
  - Description: Random Forest is an ensemble learning method that builds a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. It combines the predictions of multiple trees to improve accuracy and control overfitting.

- Gradient Boosting Classifier
  - Description: Gradient Boosting is an ensemble learning method that builds a series of decision trees and combines their predictions. It works by fitting a new tree to the residual errors of the previous tree. Gradient Boosting is known for its high predictive power and flexibility.

- AdaBoost Classifier
  - Description: AdaBoost is an ensemble learning method that combines multiple weak classifiers to form a strong classifier. It assigns weights to each instance and adjusts them at each iteration. AdaBoost focuses more on the misclassified instances in the training set, allowing for better performance.

- Extra Trees Classifier
  - Description: Extra Trees (Extremely Randomized Trees) is an ensemble learning method that builds multiple decision trees and merges their results. It adds randomness to the tree-building process by considering random splits at each node. Extra Trees can lead to faster training times and may reduce overfitting.

- XGBoost Classifier
  - Description: XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed for efficient and accurate large-scale machine learning tasks. It is known for its speed and performance, making it a popular choice in machine learning competitions.

- LightGBM Classifier
  - Description: LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is designed for distributed and efficient training of large datasets. LightGBM is known for its speed and efficiency, making it a suitable choice for large-scale applications.

- Artificial Neural Network (ANN)
  - Description: An Artificial Neural Network has been implemented for classification. It comprises an input layer, multiple hidden layers, and an output layer. The network is trained using backpropagation.

## Model Performance

The following table provides an overview of the error metrics for each model:

|  model                       |  accuracy   |  precision  |  recall   |  f1    |
|------------------------------|-------------|------------|----------|----------|
|  LogisticRegression          |  0.761905   |  0.692308  |  0.5625  |  0.620690|
|  LogisticRegression          |  0.761905   |  0.692308  |  0.5625  |  0.620690|
|  SVC                         |  0.740260   |  0.635135  |  0.5875  |  0.610390|
|  GaussianNB                  |  0.757576   |  0.646341  |  0.6625  |  0.654321|
|  RandomForestClassifier      |  0.753247   |  0.638554  |  0.6625  |  0.650307|
|  GradientBoostingClassifier  |  0.727273   |  0.597701  |  0.6500  |  0.622754|
|  AdaBoostClassifier          |  0.740260   |  0.631579  |  0.6000  |  0.615385|
|  ExtraTreesClassifier        |  0.735931   |  0.626667  |  0.5875  |  0.606452|
|  XGBClassifier               |  0.709957   |  0.571429  |  0.6500  |  0.608187|
|  LGBMClassifier              |  0.761905   |  0.640449  |  0.7125  |  0.674556|



