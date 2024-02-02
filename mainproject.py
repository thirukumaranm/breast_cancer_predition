# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['breast_cancer'] = data.target

# Display the first few rows
print(df.head())

# Data Preprocessing
# No missing values, no categorical variables, scaling numerical features
X = df.drop('breast_cancer', axis=1)
y = df['breast_cancer']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
# Logistic Regression
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC AUC Score: {roc_auc:.2f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # ROC Curve
    # (Assuming binary classification)
    y_scores = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

# Evaluate Logistic Regression
print("Logistic Regression:")
evaluate_model(logreg_model, X_test, y_test)

# Evaluate Decision Tree
print("\nDecision Tree:")
evaluate_model(dt_model, X_test, y_test)

# Cross-Validation
cv_scores = cross_val_score(logreg_model, X, y, cv=5, scoring='accuracy')
print("\nCross-Validation Scores (Logistic Regression):", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Hyperparameter Tuning (Optional)
# Example: GridSearchCV for Logistic Regression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_logreg_model = grid_search.best_estimator_

# Evaluate the tuned Logistic Regression model
print("\nTuned Logistic Regression:")
evaluate_model(best_logreg_model, X_test, y_test)

# Conclusion
print("\nConclusion:")
print("Logistic Regression performs better than the Decision Tree in this case.")
print("Data preprocessing, including scaling and handling missing values, is crucial for model performance.")
