import pickle

import joblib
import matplotlib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Force a backend that works (TkAgg, Agg, or Qt5Agg)
matplotlib.use('TkAgg')

import warnings
warnings.filterwarnings('ignore')

# Load the split data from pickle file
with open(r'C:\Users\Najihah Azman\PycharmProjects\pythonProject6\split_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Instantiate the RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)

# Fit the model
rfc.fit(X_train, y_train)

# Predict the Test set results
y_pred = rfc.predict(X_test)

# Check accuracy score
print(f'Model accuracy score with default decision-trees: {accuracy_score(y_test, y_pred):0.4f}')

# Instantiate the classifier with n_estimators = 100
rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)

# Fit the model to the training set
rfc_100.fit(X_train, y_train)

# Predict on the test set results
y_pred_100 = rfc_100.predict(X_test)

# Check accuracy score
print(f'Model accuracy score with 100 decision-trees: {accuracy_score(y_test, y_pred_100):0.4f}')

# 2. Save the trained model
joblib.dump(rfc_100, 'random_forest_model.pkl')  # Saving the model here
print("Model saved!")

# Feature importance
feature_scores = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Create a seaborn bar plot
sns.barplot(x=feature_scores, y=feature_scores.index)

# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')

# Add title to the graph
plt.title("Visualizing Important Features")

# Visualize the graph
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print('Confusion Matrix\n', cm)

# Classification report
print(classification_report(y_test, y_pred))

# Plot confusion matrix using Seaborn heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


import numpy as np
import matplotlib.pyplot as plt


# Updated Classification Report Data
labels = ['Class 0', 'Class 1', 'Class 2']
precision = [0.98, 0.99, 0.99]
recall = [0.94, 1.00, 1.00]
f1_score = [0.96, 0.99, 0.99]

x = np.arange(len(labels))  # Label positions
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width, precision, width, label='Precision', color='royalblue')
rects2 = ax.bar(x, recall, width, label='Recall', color='orange')
rects3 = ax.bar(x + width, f1_score, width, label='F1-score', color='green')

# Labels, title, and legend
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Classification Report Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0.9, 1.02])  # Set y-axis limit for better visualization
ax.legend()

# Show values on top of bars
for rect in rects1 + rects2 + rects3:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')

plt.show()

# --------------------------
# Accuracy, Macro Avg, Weighted Avg Visualization
# --------------------------

# Updated Metrics and Values
metrics = ['Accuracy', 'Macro Avg', 'Weighted Avg']
values = [0.99, 0.98, 0.99]  # Accuracy, macro avg F1-score, weighted avg F1-score

plt.figure(figsize=(6, 4))
plt.bar(metrics, values, color=['blue', 'orange', 'green'])

# Labels and title
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.title("Classification Report Summary")
plt.ylim(0.95, 1.0)  # Adjust y-axis range

# Display values on top of bars
for i, v in enumerate(values):
    plt.text(i, v + 0.005, f"{v:.2f}", ha='center', fontsize=12)

plt.show()


