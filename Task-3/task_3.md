# Task 03

Heart Disease Prediction

## Objectives  

    Build a model to predict whether a person is at risk of heart disease based on their health data. 

## Dataset that is used

    Heart Disease UCI Dataset (available on Kaggle) 

## Instructions

    Clean the dataset (handle missing values if any). 
    Perform Exploratory Data Analysis (EDA) to understand trends. 
    Train a classification model (Logistic Regression or Decision Tree). 
    Evaluate using metrics: accuracy, ROC curve, and confusion matrix. 
    Highlight important features affecting prediction. 

## Models to build

### Linear Regression

This algorithm is used when you want to predict a continuous value based on one or more input features. It assumes a linear relationship between the inputs and the output.

### Decision Tree

This one works like a flowchart: it splits the data into branches based on feature values. It can be used for both classification (e.g., spam vs. not spam) and regression (predicting a number)

## Evaluating models

```python

# Evaluating the models
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay

def evaluate_model(model, X_test, y_test, title=""):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])

    print(f"\n {title} Accuracy: {acc:.2f}")
    print(f" ROC AUC: {roc:.2f}")

    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.title(f"{title} - Confusion Matrix")
    plt.show()

    plt.plot(fpr, tpr, label=f"{title} (AUC = {roc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} - ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

```  

Evaluating model using evaluate() function We:

Print Accuracy

Show Confusion Matrix

Plot ROC Curve with AUC score

It gives us a complete picture of true positives, false alarms, and overall performance.

## Highliting importance features affecting prediction

```python  
#  Step 9: Feature importance
    feat_importance = pd.Series(dt_model.feature_importances_index=X_train.columns)
    feat_importance.nlargest(10).plot(kind='barh', color='coral')
    plt.title('Top Features Affecting Heart Disease Prediction')
    plt.xlabel('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()

```  

We extract top 10 features influencing heart disease prediction (like age, thalach, cp, etc.) using the Decision Tree model.
This gives real-world insights about the factors that can be the real reason of heart disease and these insights can be shared with doctors or domain experts!  

## Summary

This project is all about creating a machine learning model that can help predict the chances of heart disease in patients by looking at their health data. We used the UCI Heart Disease dataset from Kaggle to build a system that identifies people at high risk, based on indicators like age, cholesterol levels, blood pressure, and types of chest pain, among other things.

We kicked things off with some data cleaning to deal with any missing bits and make sure everything was consistent. Then, we did some exploratory data analysis (EDA) to find connections between different features and the target variable, which is whether or not someone has heart disease. We visualized important patterns with histograms, heatmaps showing correlation, and charts that highlight which features matter most.

We trained up two models:

Logistic Regression, since it’s pretty straightforward and easy to explain

Decision Tree Classifier, because it gives us a clear rule-based way to look at data.

To see how well our models were performing, we looked at accuracy, confusion matrices, and ROC-AUC scores. The ROC curve helped us understand how sensitive the models were versus how specific they were. Some of the most important features turned out to be the type of chest pain, the maximum heart rate achieved, and ST depression.

In the end, the models showed some really good predictive power, with ROC-AUC scores over 0.80, which suggests they can classify well. This approach could be a useful tool for helping doctors prevent heart issues.  
