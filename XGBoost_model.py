

import pickle

import matplotlib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import xgboost as xgb

# Force a backend that works (TkAgg, Agg, or Qt5Agg)
matplotlib.use('TkAgg')

import warnings

warnings.filterwarnings('ignore')

# Load the split data from pickle file
with open('split_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Declare parameters for multi-class classification
num_classes = len(set(y_train))  # Number of unique classes in y_train

params = {
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': num_classes,  # Set the number of classes dynamically
    'max_depth': 4,
    'alpha': 10,
    'learning_rate': 0.1,
    'n_estimators': 100
}


# instantiate the classifier
xgb_clf = XGBClassifier(**params)
(xgb_clf.fit(X_train, y_train))

# make predictions on test data
y_pred = xgb_clf.predict(X_test)

# check accuracy score
from sklearn.metrics import accuracy_score

print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))



# # Cross-validation
# dtrain = xgb.DMatrix(data=X_train, label=y_train)  # Prepare data for cross-validation
# xgb_cv = xgb.cv(dtrain=dtrain, params=params, nfold=3,
#                 num_boost_round=50, early_stopping_rounds=10, metrics="mlogloss", as_pandas=True, seed=123)
#
# print(xgb_cv.head())

# plot graph of the importance
xgb.plot_importance(xgb_clf)
plt.rcParams['figure.figsize'] = [6, 4]
plt.show()

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Compute classification report
report = classification_report(y_test, y_pred)

# Print classification report
print("Classification Report:\n", report)

# Plot confusion matrix using Seaborn heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_train), yticklabels=set(y_train))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Save the XGBoost model
# xgb_clf.save_model('xgboost_model.json')  # Saves as a JSON file for XGBoost

import numpy as np
import matplotlib.pyplot as plt

# Classification report data
labels = ['Class 0', 'Class 1', 'Class 2']
precision = [1.00, 0.99, 0.99]
recall = [0.93, 1.00, 0.99]
f1_score = [0.96, 0.99, 0.99]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-score')

# Labels, title, and legend
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Classification Report Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0, 1.1])  # Set y-axis limit
ax.legend()

# Show the values on top of bars
for rect in rects1 + rects2 + rects3:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')

plt.show()

import matplotlib.pyplot as plt

# Metrics and their values
metrics = ['Accuracy', 'Macro Avg', 'Weighted Avg']
values = [0.99, 0.98, 0.99]  # Accuracy, macro avg F1-score, weighted avg F1-score

# Create a bar chart
plt.figure(figsize=(6, 4))
plt.bar(metrics, values, color=['blue', 'orange', 'green'])

# Labels and title
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.title("Classification Report Summary")
plt.ylim(0.9, 1.0)  # Set y-axis range for better visualization

# Display values on top of bars
for i, v in enumerate(values):
    plt.text(i, v + 0.005, f"{v:.2f}", ha='center', fontsize=12)

plt.show()


# from sklearn.model_selection import RandomizedSearchCV
# from xgboost import XGBClassifier
#
# param_dist = {
#     'max_depth': [4, 6, 8, 10],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'n_estimators': [100, 200, 300],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'gamma': [0, 0.1, 0.3, 0.5],
#     'reg_alpha': [0, 0.1, 1, 10],
#     'reg_lambda': [0, 1, 10]
# }
#
# xgb_model = XGBClassifier(objective='multi:softmax', num_class=num_classes, use_label_encoder=False)
#
# random_search = RandomizedSearchCV(
#     estimator=xgb_model,
#     param_distributions=param_dist,
#     scoring='accuracy',
#     n_iter=50,  # increase if time permits
#     cv=3,
#     verbose=2,
#     n_jobs=-1,
#     random_state=42
# )
#
# random_search.fit(X_train, y_train)
#
# # Evaluate
# print("Best Parameters:", random_search.best_params_)
# print("Best Accuracy:", random_search.best_score_)
#
# best_model = random_search.best_estimator_
# y_pred = best_model.predict(X_test)
# print("Test Accuracy with best params:", accuracy_score(y_test, y_pred))

best_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1
}
from xgboost import XGBClassifier


# Use pre-found best hyperparameters
model = XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    use_label_encoder=False,
    eval_metric='mlogloss',
    **best_params
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Test Accuracy with best params:", accuracy_score(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_train),
            yticklabels=np.unique(y_train))
plt.title('Confusion Matrix - Tuned XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
report_dict = classification_report(y_test, y_pred, output_dict=True)

labels = list(report_dict.keys())[:num_classes]
precision = [report_dict[label]['precision'] for label in labels]
recall = [report_dict[label]['recall'] for label in labels]
f1 = [report_dict[label]['f1-score'] for label in labels]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width, precision, width, label='Precision')
bars2 = ax.bar(x, recall, width, label='Recall')
bars3 = ax.bar(x + width, f1, width, label='F1-score')

ax.set_xticks(x)
ax.set_xticklabels([f'Class {label}' for label in labels])
ax.set_ylim([0, 1.1])
ax.set_title("Classification Report Metrics - Tuned Model")
ax.legend()

# Add value labels on top
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()
accuracy = accuracy_score(y_test, y_pred)
macro_avg = report_dict['macro avg']['f1-score']
weighted_avg = report_dict['weighted avg']['f1-score']

plt.figure(figsize=(6, 4))
plt.bar(['Accuracy', 'Macro Avg', 'Weighted Avg'],
        [accuracy, macro_avg, weighted_avg],
        color=['blue', 'orange', 'green'])

plt.ylim(0.85, 1.0)
plt.title('Tuned XGBoost - Accuracy & Averages')

for i, v in enumerate([accuracy, macro_avg, weighted_avg]):
    plt.text(i, v + 0.005, f"{v:.2f}", ha='center', fontsize=12)

plt.tight_layout()
plt.show()
