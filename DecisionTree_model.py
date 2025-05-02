import pickle
import warnings
import matplotlib
from sklearn.preprocessing import LabelEncoder
# import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



warnings.filterwarnings('ignore')

matplotlib.use('TkAgg')  # Set the backend to TkAgg
# Load the split data from pickle file
with open(r'C:\Users\Najihah Azman\PycharmProjects\pythonProject6\split_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)


# instantiate the DecisionTreeClassifier model with criterion gini index
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

# fit the model
clf_gini.fit(X_train, y_train)

# predict using criterion gini index
y_pred_gini = clf_gini.predict(X_test)

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

y_pred_train_gini = clf_gini.predict(X_train)

y_pred_train_gini

print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))

# instantiate the DecisionTreeClassifier model with criterion entropy
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
# fit the model
clf_en.fit(X_train, y_train)
# predict using criterion entropy
y_pred_en = clf_en.predict(X_test)

print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))

y_pred_train_en = clf_en.predict(X_train)

y_pred_train_en

print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))
#
# from sklearn.model_selection import GridSearchCV
#
# # Define hyperparameter grid
# param_grid = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [3, 5, 10, 15, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# # Create base model
# dt = DecisionTreeClassifier(random_state=0)
#
# # Initialize Grid Search with Cross-Validation
# grid_search = GridSearchCV(estimator=dt,
#                            param_grid=param_grid,
#                            cv=5,
#                            n_jobs=-1,
#                            verbose=1,
#                            scoring='accuracy')
#
# # Fit the model
# grid_search.fit(X_train, y_train)
#
# # Best model details
# print("Best hyperparameters:", grid_search.best_params_)
#
# best_model = grid_search.best_estimator_
#
# # Evaluate tuned model
# y_pred_best = best_model.predict(X_test)
#
# print("Accuracy of best model on test set: {:.4f}".format(accuracy_score(y_test, y_pred_best)))
#

# Print the Confusion Matrix for Entropy
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_en)

print('Confusion matrix\n\n', cm)

# Classification report for Entropy
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_en))


# Encode target variable (only needed if not done before)
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Updated Classification report data
labels = ['Class 0', 'Class 1', 'Class 2']
precision = [1.00, 0.97, 0.99]
recall = [0.84, 1.00, 1.00]
f1_score = [0.91, 0.99, 0.99]

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
ax.set_ylim([0, 1.1])  # Set y-axis limit
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

# Updated metrics and values
metrics = ['Accuracy', 'Macro Avg', 'Weighted Avg']
values = [0.97, 0.94, 0.96]  # Accuracy, macro avg F1-score, weighted avg F1-score

plt.figure(figsize=(6, 4))
plt.bar(metrics, values, color=['blue', 'orange', 'green'])

# Labels and title
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.title("Classification Report Summary")
plt.ylim(0.85, 1.0)  # Adjust y-axis range

# Display values on top of bars
for i, v in enumerate(values):
    plt.text(i, v + 0.005, f"{v:.2f}", ha='center', fontsize=12)

plt.show()


# Compute confusion matrix for Gini-based model
cm_gini = confusion_matrix(y_test, y_pred_gini)

# Display confusion matrix
print('Confusion matrix (Gini Criterion)\n\n', cm_gini)
print(classification_report(y_test, y_pred_gini))

# Encode target variable (only needed if not done before)
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

# Plot Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_gini, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Gini Criterion)')
plt.show()

# Classification Report Metrics (Precision, Recall, F1-score)
# Extracting report data
report_gini = classification_report(y_test, y_pred_gini, output_dict=True)
labels = list(report_gini.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
precision = [report_gini[label]['precision'] for label in labels]
recall = [report_gini[label]['recall'] for label in labels]
f1_score = [report_gini[label]['f1-score'] for label in labels]

x = np.arange(len(labels))  # Label positions
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width, precision, width, label='Precision', color='royalblue')
rects2 = ax.bar(x, recall, width, label='Recall', color='orange')
rects3 = ax.bar(x + width, f1_score, width, label='F1-score', color='green')

# Labels, title, and legend
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Classification Report Metrics (Gini Criterion)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0, 1.1])  # Set y-axis limit
ax.legend()

# Show values on top of bars
for rect in rects1 + rects2 + rects3:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')

plt.show()

# Accuracy, Macro Avg, Weighted Avg Visualization
metrics = ['Accuracy', 'Macro Avg', 'Weighted Avg']
values = [report_gini['accuracy'], report_gini['macro avg']['f1-score'], report_gini['weighted avg']['f1-score']]

plt.figure(figsize=(6, 4))
plt.bar(metrics, values, color=['blue', 'orange', 'green'])

# Labels and title
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.title("Classification Report Summary (Gini Criterion)")
plt.ylim(0.85, 1.0)  # Adjust y-axis range

# Display values on top of bars
for i, v in enumerate(values):
    plt.text(i, v + 0.005, f"{v:.2f}", ha='center', fontsize=12)

plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Define the best model with tuned hyperparameters
best_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=0
)

# Train the model
best_model.fit(X_train, y_train)

# Predict on the test set
y_pred_best = best_model.predict(X_test)

# Evaluate accuracy
print("Accuracy of best model on test set: {:.4f}".format(accuracy_score(y_test, y_pred_best)))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Confusion matrix
cm_best = confusion_matrix(y_test, y_pred_best)

# Label encoding (for heatmap labels)
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Tuned Model)')
plt.show()

from sklearn.metrics import classification_report
import numpy as np

# Classification report
report_best = classification_report(y_test, y_pred_best, output_dict=True)

# Extract class names
labels = list(report_best.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
precision = [report_best[label]['precision'] for label in labels]
recall = [report_best[label]['recall'] for label in labels]
f1_score = [report_best[label]['f1-score'] for label in labels]

# Bar plot
x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width, precision, width, label='Precision', color='royalblue')
rects2 = ax.bar(x, recall, width, label='Recall', color='orange')
rects3 = ax.bar(x + width, f1_score, width, label='F1-score', color='green')

ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Classification Metrics (Tuned Model)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0, 1.1])
ax.legend()

# Annotate scores
for rect in rects1 + rects2 + rects3:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')

plt.show()
# Summary metrics
metrics = ['Accuracy', 'Macro Avg', 'Weighted Avg']
values = [
    report_best['accuracy'],
    report_best['macro avg']['f1-score'],
    report_best['weighted avg']['f1-score']
]

# Bar chart
plt.figure(figsize=(6, 4))
plt.bar(metrics, values, color=['blue', 'orange', 'green'])

plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.title("Classification Summary (Tuned Model)")
plt.ylim(0.85, 1.0)

# Annotate
for i, v in enumerate(values):
    plt.text(i, v + 0.005, f"{v:.2f}", ha='center', fontsize=12)

plt.show()
